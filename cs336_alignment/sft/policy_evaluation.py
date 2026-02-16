import numpy as np
import torch
from vllm import SamplingParams, LLM
import random, json
from unittest.mock import patch
from cs336_alignment.sft import sft_helper
import os


def init_vllm(model_id: str, device: str, seed: int, max_model_len: int, gpu_memory_utilization: float=0.85, enforce_eager=True):
    """
    Start the inference process, use vLLM to hold a model separate from the policy
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None
    )
    with world_size_patch, profiling_patch:
        return LLM(
            model=model_id,
            device=device,
            dtype="bfloat16",
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            enforce_eager=enforce_eager
        )

def load_policy_into_vllm_instance(policy, llm: LLM):
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())

def evaluate_vllm(vllm_model: LLM,
                  policy_model, # 训练中的 HF 模型
                  eval_model,# 留在cuda1上的副本模型
                  problems: list[str],
                  solutions: list[str],
                  reward_fn,
                  eval_sampling_params: SamplingParams,
                  output_path: str):
    """
    使用 vLLM 生成回复，并使用 HF 模型以 micro-batch 方式计算 entropy。
    适配 Qwen 2.5 大词表模型，防止显存爆炸和索引错误。
    vllm_model, eval_model @ cuda:1
    policy_model @ cuda:0
    """
    if output_path:
        dir_name = os.path.dirname(output_path)
        os.makedirs(dir_name, exist_ok=True)

    prompts = []
    # 1. 构造推理 Prompt
    for problem in problems:
        pt = f"""
            A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.
            User: {problem}
            Assistant: <think>
            """
        prompts.append(pt)

    # 2. vLLM 生成 (在 GPU 1 上高效运行)
    outputs = vllm_model.generate(prompts, eval_sampling_params)
    response = [output.outputs[0].text for output in outputs]

    # 把训练模型的权重同步到cuda1上
    eval_model.load_state_dict(policy_model.state_dict())

    # 3. 初始化评价指标
    # 使用 micro-batch 主要是为了保护GPU被大词表的 Logits 撑爆
    micro_batch_size = 4  # 针对 Qwen 2.5 (152k 词表)
    total_sum_entropy = 0.0
    total_valid_tokens = 0

    total_format_reward = 0.0
    total_reward = 0.0

    correct_lengths = []
    incorrect_lengths = []
    results_to_save = []

    tokenizer = vllm_model.get_tokenizer()
    device = eval_model.device

    # 4. Micro-batch 循环处理：同步计算 Entropy 和 Reward
    with torch.no_grad():
        for i in range(0, len(prompts), micro_batch_size):

            # 获取当前批次切片
            batch_prompts = prompts[i: i + micro_batch_size]
            batch_responses = response[i: i + micro_batch_size]
            batch_solutions = solutions[i: i + micro_batch_size]

            # A. 计算 Entropy (在 GPU 1 上)
            # 这一步通过 tokenize_prompt_and_output 动态生成当前 batch 的 tensor
            val_batch_ids = sft_helper.tokenize_prompt_and_output(batch_prompts, batch_responses, tokenizer)
            input_ids = val_batch_ids["input_ids"].to(device)
            labels = val_batch_ids["labels"].to(device)
            response_mask = val_batch_ids["response_mask"].to(device)

            token_entropy = sft_helper.get_response_log_probs(
                eval_model,
                input_ids,
                labels,
                return_token_entropy=True
            )["token_entropy"]

            torch.cuda.empty_cache()

            total_sum_entropy += (token_entropy * response_mask).sum().item()
            total_valid_tokens += response_mask.sum().item()

            # B. 计算 Reward 并统计长度
            # 注意：内部循环 j 永远相对于当前 micro_batch 寻址，解决 IndexError
            for j in range(len(batch_responses)):
                current_resp = batch_responses[j]
                current_sol = batch_solutions[j]
                grades = reward_fn(current_resp, current_sol)

                f_rew = grades.get("format_reward", 0.0)
                r_rew = grades.get("reward", 0.0)

                total_format_reward += f_rew
                total_reward += r_rew

                # 使用当前 batch 内的 mask 获取长度
                resp_len = torch.sum(response_mask[j]).item()

                if r_rew > 0:
                    correct_lengths.append(resp_len)
                else:
                    incorrect_lengths.append(resp_len)

                # 收集结果用于磁盘持久化
                results_to_save.append({
                    "problem": batch_prompts[j],
                    "ground_truth": current_sol,
                    "model_generation": current_resp,
                    "format_reward": f_rew,
                    "reward": r_rew,
                    "response_length": resp_len
                })

    # 5. 汇总计算最终指标
    avg_token_entropy = total_sum_entropy / total_valid_tokens if total_valid_tokens > 0 else 0.0
    num_samples = len(problems)

    def safe_avg(data_list):
        return sum(data_list) / len(data_list) if len(data_list) > 0 else 0.0

    metrics = {
        "eval/format_reward": total_format_reward / num_samples,
        "eval/avg_reward": total_reward / num_samples,
        "eval/avg_correct_response_length": safe_avg(correct_lengths),
        "eval/avg_incorrect_response_length": safe_avg(incorrect_lengths),
        "eval/avg_token_entropy": avg_token_entropy
    }


    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in results_to_save:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"Evaluation results saved to {output_path}")

    return metrics