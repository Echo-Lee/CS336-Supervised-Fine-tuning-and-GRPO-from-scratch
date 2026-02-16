from vllm import LLM, SamplingParams
import json
from cs336_alignment.sft import policy_evaluation, sft_helper
from cs336_alignment import drgrpo_grader
import torch
import argparse
import os


def evaluate_vllm(
    vllm_model: LLM,
    problems: list[str],
    solutions: list[str],
    reward_fn,
    eval_sampling_params: SamplingParams,
    output_path
):
    """
    Using a vLLM model to evaluate model generated answers reward to solutions via grader reward_fn.
    Save detailed record for each question-answer
    """
    # Get prompts
    prompts = []
    with open("/root/autodl-tmp/cs336/assignment5-alignment-main/cs336_alignment/prompts/r1_zero.prompt", "r", encoding="utf-8") as f:
        prompt_template = f.read().strip()

    for problem in problems:
        pt = prompt_template.format(question=problem.strip())
        prompts.append(pt)

    # Generate response and extract texts
    outputs = vllm_model.generate(prompts, eval_sampling_params)
    responses = [output.outputs[0].text for output in outputs]

    # Record reward
    total_reward = 0.0
    total_format_reward = 0.0
    results = [] # Record each response

    # Loop each response and use parser to get grades
    for i, generated_text in enumerate(responses):
        grades = reward_fn(generated_text, solutions[i])

        total_reward += grades.get("reward", 0)
        total_format_reward += grades.get("format_reward", 0)

        results.append({
            "problem": prompts[i],
            "ground_truth": solutions[i],
            "model_generation": generated_text,
            "format_reward": grades.get("format_reward", 0),
            "reward": grades.get("reward", 0)
        })

    n = len(problems)

    avg_reward = total_reward / n
    avg_format_reward = total_format_reward / n

    metrics = {
        "avg_reward": avg_reward,
        "avg_format_reward": avg_format_reward
    }


    print("\n========== Evaluation Results ==========")
    print(f"Average Reward        : {avg_reward:.4f}")
    print(f"Average Format Reward : {avg_format_reward:.4f}")
    print("========================================\n")

    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in results:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"Evaluation results saved to {output_path}")

    return metrics

if __name__ == "__main__":
    vllm_engine = LLM(
        model_id="/root/autodl-tmp/cs336/checkpoints/rl_qwen_1.5b_full/qwen1.5b-rl-sweep-lr2e-5/best",
        device="cuda:0",
        gpu_memory_utilization=0.85,
        max_model_len=1536,
        enforce_eager=False,
        dtype="bfloat16",
        enable_prefix_caching=True
    )

    with open("/root/autodl-tmp/cs336/assignment5-alignment-main/data/math/data/sft-reason/val.jsonl", 'r', encoding='utf-8') as f:
        data_list = json.load(f)

    val_problems = [item['problem'] for item in data_list]
    val_solutions = [str(item['expected_answer']) for item in data_list]

    eval_sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        min_tokens=4,
        max_tokens=1024,
        stop=["</answer>"],
        include_stop_str_in_output=True
    )

    savepath = "/root/autodl-tmp/cs336/assignment5-alignment-main/data/math/evaluation/qwen1.5b-rl-sweep-lr2e-5.jsonl"

    metrics = evaluate_vllm(vllm_engine, val_problems[:1024], val_solutions[:1024], drgrpo_grader.r1_zero_reward_fn, eval_sampling_params, savepath)