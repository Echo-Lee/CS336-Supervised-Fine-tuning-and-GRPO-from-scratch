import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import SamplingParams, LLM

import sft_helper, get_cosine_lr, policy_evaluation
from torch.utils.data import TensorDataset, DataLoader

import json, yaml
import wandb

def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def train(
    model,
    optimizer,
    tokenizer,
    train_prompt_strs: list[str],
    train_output_strs: list[str],
    val_problems: list[str],
    val_solutions: list[str],
    reward_fn,
    config: dict,
    eval_model = None,
    vllm_engine: LLM = None,
):

    train_conf = config["training"]
    eval_conf = config["evaluation"]

    # Tokenizer setting and data loading
    dataset = sft_helper.tokenize_prompt_and_output(train_prompt_strs, train_output_strs, tokenizer)
    tensor_ds = TensorDataset(dataset["input_ids"], dataset["labels"], dataset["response_mask"])

    loader = DataLoader(tensor_ds, batch_size=train_conf["batch_size"], shuffle=True, drop_last=True)

    # vLLM sample config
    sampling_params = SamplingParams(
        temperature=eval_conf["temperature"],
        top_p=1.0,
        max_tokens=eval_conf["max_tokens"],
        stop=["</answer>"],
        include_stop_str_in_output=True
    )

    wandb.define_metric("train_step")
    wandb.define_metric("eval_step")
    wandb.define_metric("train/*", step_metric="train_step")
    wandb.define_metric("eval/*", step_metric="eval_step")

    print("Starting training loop...")
    model.train()

    global_step = 1
    total_loss = 0.0
    grad_accum_steps = train_conf["gradient_accumulation_steps"]
    max_iter = train_conf["max_iter"]


    while global_step <= max_iter:
        for micro_step, (input_ids, labels, response_mask) in enumerate(loader):
            if global_step > max_iter:
                break

            # Move to GPU
            device = model.device
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            response_mask = response_mask.to(device)

            log_probs_dict = sft_helper.get_response_log_probs(
                model,
                input_ids,
                labels,
                return_token_entropy=False # Config setting
            )
            log_probs = log_probs_dict["log_probs"]

            loss, metadata = sft_helper.sft_microbatch_train_step(
                log_probs,
                response_mask,
                gradient_accumulation_steps=grad_accum_steps,
                normalize_constant=1.0)

            total_loss += loss.item()

            # 积累到一定梯度后执行更新
            if (micro_step + 1) % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=train_conf["clip_grad"]
                )

                lr = get_cosine_lr.learning_rate_schedule(t=global_step,
                                                          alpha_max=float(train_conf["learning_rate_max"]),
                                                          alpha_min=float(train_conf["learning_rate_min"]),
                                                          t_warm=train_conf["warmup_steps"],
                                                          t_cold=train_conf["max_iter"]
                                                          )

                for group in optimizer.param_groups:
                    group["lr"] = lr

                optimizer.step()
                optimizer.zero_grad()

                global_step += 1

                # 打印日志
                if global_step % 2 == 0:
                    print(f"Step {global_step}: Loss = {total_loss:.4f} | LR = {lr:.2e}")

                wandb.log({
                    "train/loss": total_loss,
                    "train/lr": optimizer.param_groups[0]['lr'],
                    "train_step": global_step  # 显式指定 x 轴
                })

                total_loss = 0.0  # 重置 Loss

                # 周期性评测：
                if global_step % eval_conf["eval_interval"] == 0 and vllm_engine is not None:
                    policy_evaluation.load_policy_into_vllm_instance(model, vllm_engine)
                    save_path = f"{config['model']['save_dir']}/training_samples-{config['training']['max_samples']}-checkpoint/eval_step_{global_step}.jsonl"
                    metrics = policy_evaluation.evaluate_vllm(
                        vllm_engine,
                        model,
                        eval_model,
                        val_problems[:eval_conf["eval_batch_size"]],
                        val_solutions[:eval_conf["eval_batch_size"]],
                        reward_fn,
                        sampling_params,
                        output_path=save_path
                    )

                    metrics["eval_step"] = global_step
                    wandb.log(metrics)
                    model.train()


    ckpt_path = f"{config['model']['save_dir']}/training_samples-{config['training']['max_samples']}-checkpoint"
    model.save_pretrained(ckpt_path)
    tokenizer.save_pretrained(ckpt_path)
    print(f"Saved checkpoint to {ckpt_path}")

    print("Training and Evaluation finished. Exiting...")
    wandb.finish()
    import os
    os._exit(0)  # 强制杀掉所有子进程退出

if __name__ == "__main__":
    # 加载config
    config = load_config("/root/autodl-tmp/cs336/assignment5-alignment-main/cs336_alignment/sft/config.yaml")
    print("Config loaded:", config)

    wandb.init(
        project=config["logging"]["project_name"],
        name=config["logging"]["run_name"],
        config=config
    )

    model_conf = config["model"]
    train_conf = config["training"]

    # 使用AutoModel这个库加载policy_model, tokenizer
    print("Loading SFT Training Model...")
    policy_model = AutoModelForCausalLM.from_pretrained(
        model_conf['model_id'],
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        attn_implementation="flash_attention_2"
    )

    policy_model.gradient_checkpointing_enable()

    # 在cuda1上加载一份模型副本
    eval_model = AutoModelForCausalLM.from_pretrained(
        model_conf['model_id'],
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=config['evaluation']['vllm_device']
    )
    eval_model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_conf['tokenizer_path'])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 初始化优化器
    print(f"Initializing Optimizer with LR: {train_conf['learning_rate_min']}...")
    optimizer = torch.optim.AdamW(
        policy_model.parameters(),
        lr=float(train_conf['learning_rate_max']),
        weight_decay=float(train_conf['weight_decay']),
        betas=(0.9, 0.95),
    )

    print("Model and Optimizer ready.")

    # 加载训练数据，注意原始数据中只有原始问题，需要加上prompt
    print(f"Loading training data from {config['data']['train_path']}...")
    with open(config['data']['train_path'], 'r', encoding='utf-8') as f:
        train_data_list = json.load(f)

    max_samples = config['training'].get('max_samples', None)
    if max_samples is not None and max_samples > 0:
        if max_samples < len(train_data_list):
            print(f"Truncating training data from {len(train_data_list)} to {max_samples} for experiment.")
            train_data_list = train_data_list[:max_samples]
        else:
            print(f"Requested max_samples ({max_samples}) > dataset size, using full dataset.")

    train_prompts = []
    train_responses = []
    for item in train_data_list:
        problem = item["problem"]
        reasoning_trace = item["reasoning_trace"]
        pt = f"""
            A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.
            User: {problem}
            Assistant: <think>
            """
        train_prompts.append(pt)
        train_responses.append(reasoning_trace)

    # 加载验证数据集
    print(f"Loading validation data from {config['data']['val_path']}...")
    with open(config['data']['val_path'], 'r', encoding='utf-8') as f:
        data_list = json.load(f)

    val_problems = [item['problem'] for item in data_list]
    val_solutions = [str(item['expected_answer']) for item in data_list]

    # 加载用于评测的vllm
    print("Initializing vLLM for evaluation...")
    try:
        # 使用 policy_evaluation 中的 init_vllm 辅助函数
        vllm_engine = policy_evaluation.init_vllm(
            model_id=config['model']['model_id'],
            device=config['evaluation']['vllm_device'],
            seed=config['training']['seed'],
            gpu_memory_utilization=config['evaluation']['gpu_memory_utilization'],
            max_model_len=config['evaluation']['max_tokens']
        )
    except Exception as e:
        print(f"Warning: Failed to initialize vLLM ({e}). Evaluation will be skipped.")
        vllm_engine = None

    from cs336_alignment.drgrpo_grader import r1_zero_reward_fn

    train(
        model=policy_model,
        optimizer=optimizer,
        eval_model=eval_model,
        tokenizer=tokenizer,
        train_prompt_strs=train_prompts,
        train_output_strs=train_responses,
        val_problems=val_problems,
        val_solutions=val_solutions,
        reward_fn=r1_zero_reward_fn,
        config=config,
        vllm_engine=vllm_engine
    )