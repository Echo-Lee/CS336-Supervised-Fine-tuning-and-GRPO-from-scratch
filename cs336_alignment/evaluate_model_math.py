from vllm import LLM, SamplingParams
import json
import torch
import argparse
import os
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn


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

    for problem in problems:
        pt = "A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.\n" + \
             f"User: {problem}\n" + \
             "Assistant: <think>\n"
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


    print("\n========== Evaluation Results ==========")
    print(f"Average Reward        : {avg_reward:.4f}")
    print(f"Average Format Reward : {avg_format_reward:.4f}")
    print("========================================\n")

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            for item in results:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"Evaluation results saved to {output_path}")

    metrics_path = output_path.replace('.jsonl', '') + '_metrics.json'
    metrics = {
        "avg_reward": avg_reward,
        "avg_format_reward": avg_format_reward,
        "sample_count": n
    }

    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=4)

    print(f"Metrics saved to {metrics_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="vLLM 自动评估脚本")
    parser.add_argument("--model_path", type=str, required=True, help="模型路径")
    parser.add_argument("--data_path", type=str, required=True, help="数据集路径 (.json 或 .jsonl)")
    parser.add_argument("--output_path", type=str, default="eval_result.jsonl", help="结果保存路径")
    args = parser.parse_args()

    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=1024,
        stop=["</answer>"]
    )

    sampling_params.include_stop_str_in_output = True

    Qwen_Math = LLM(model=args.model_path, enable_prefix_caching=True, max_model_len=1536, dtype="bfloat16")

    with open(args.data_path, 'r', encoding='utf-8') as f:
        data_list = json.load(f)

    problems = [item['problem'] for item in data_list]
    solutions = [str(item['expected_answer']) for item in data_list]

    evaluate_vllm(
        Qwen_Math,
        problems[:100],
        solutions[:100],
        r1_zero_reward_fn,
        sampling_params,
        args.output_path,
    )
