"""
step = 1, ..., n_grpo_steps:
    sample 1 rollout batch
    step = 1, ..., n_train_steps_per_rollout_batch:
        for on-policy update, there is only 1 step, which means that train_batch_size = rollout_batch_size
"""
import torch
import json, yaml
import os
import wandb

import grpo_helper, grpo_evaluator
from transformers import AutoModelForCausalLM, AutoTokenizer
from cs336_alignment.sft import policy_evaluation, sft_helper
from cs336_alignment import drgrpo_grader
from vllm import SamplingParams, LLM
from torch.utils.data import TensorDataset, DataLoader

def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def train_loop(
    policy_model,
    vllm_engine,
    optimizer,
    tokenizer,
    config,
    train_prompts,
    val_problems,
    train_solutions,
    val_solutions
):
    train_conf = config["training"]
    eval_conf = config["evaluation"]
    rollout_batch_size = train_conf["rollout_batch_size"]
    group_size = train_conf["group_size"]
    train_batch_size = train_conf["train_batch_size"]
    gradient_accumulation_steps = train_conf["gradient_accumulation_steps"]
    epochs_per_rollout_batch = train_conf["epochs_per_rollout_batch"]
    cliprange = train_conf["cliprange"]
    length_norm = train_conf.get("length_norm", "masked_mean")
    # some sanity check
    assert train_batch_size % gradient_accumulation_steps == 0, (
        "train_batch_size must be divisible by gradient_accumulation_steps"
    )
    micro_batch_size = train_batch_size // gradient_accumulation_steps

    assert rollout_batch_size % group_size == 0, (
        "rollout_batch_size must be divisible by group_size"
    )
    n_prompts_per_rollout_batch = rollout_batch_size // group_size

    assert train_batch_size >= group_size, (
        "train_batch_size must be greater than or equal to group_size"
    )

    device = policy_model.device

    # Loading training prompts and answers into dataloader:
    dataset = grpo_helper.PromptDataset(train_prompts, train_solutions)


    prompt_loader = DataLoader(
        dataset,
        batch_size=n_prompts_per_rollout_batch,  # draw D answers-solutions pairs each time
        shuffle=True,
        drop_last=False
    )
    prompt_iter = iter(prompt_loader)

    # vllm sampling config, note that we generate group size answers for each prompts in training.
    training_sampling_params = SamplingParams(
        n=group_size,
        temperature=eval_conf["sampling_temperature"],
        top_p=1.0,
        min_tokens=eval_conf["sampling_min_tokens"],
        max_tokens=eval_conf["sampling_max_tokens"],
        stop=["</answer>"],
        include_stop_str_in_output=True
    )

    eval_sampling_params = SamplingParams(
        temperature=eval_conf["sampling_temperature"],
        top_p=1.0,
        min_tokens=eval_conf["sampling_min_tokens"],
        max_tokens=eval_conf["sampling_max_tokens"],
        stop=["</answer>"],
        include_stop_str_in_output=True
    )

    wandb.define_metric("train_step")
    wandb.define_metric("eval_step")
    wandb.define_metric("train/*", step_metric="train_step")
    wandb.define_metric("eval/*", step_metric="eval_step")

    # start training
    print("Starting training loop...")
    policy_model.train()
    best_metric = 0

    def save_model(path):
        os.makedirs(path, exist_ok=True)
        policy_model.save_pretrained(path)
        tokenizer.save_pretrained(path)
        print(f"model saved to {path}.")

    for step in range(train_conf["n_grpo_steps"]):
        try:
            batch_prompts, batch_solutions = next(prompt_iter)
        except StopIteration:
            prompt_iter = iter(prompt_loader)
            batch_prompts, batch_solutions = next(prompt_iter)

        # feed in all batched prompts to get (group_size) generated answers each, so the len should be group_size * n_prompts_per_rollout_batch = rollout batch size
        outputs = vllm_engine.generate(batch_prompts, training_sampling_params)
        # flatten the responses
        rollout_batch_responses = []
        rollout_batch_prompts_repeated = []
        rollout_batch_solutions_repeated = []  # Record expected answers

        for i, output in enumerate(outputs):
            prompt_text = output.prompt
            solution = batch_solutions[i]

            # each output.outputs contains group_size results
            for completion in output.outputs:
                rollout_batch_responses.append(completion.text)
                rollout_batch_prompts_repeated.append(prompt_text)
                rollout_batch_solutions_repeated.append(solution)

        # Calculate advantages and raw rewards for generated text against ground truth.
        advantages, raw_rewards, metadata = grpo_helper.compute_group_normalized_rewards(
            drgrpo_grader.r1_zero_reward_fn,
            rollout_batch_responses,
            rollout_batch_solutions_repeated,
            group_size,
            float(train_conf["advantage_eps"]),
            train_conf["use_std_normalization"]
        )

        rewards_mean = metadata["rewards_mean"]
        format_reward_mean = metadata["format_reward_mean"]

        # Here we need to compute old_log_probs
        # First let's tokenize prompts and answers
        encode_dict = sft_helper.tokenize_prompt_and_output(rollout_batch_prompts_repeated, rollout_batch_responses, tokenizer)

        with torch.no_grad():
            # sum() 计算所有 response token 的总数
            # size(0) 是 rollout_batch_size
            mean_token_counts = encode_dict["response_mask"].sum().float() / encode_dict["response_mask"].size(0)
            mean_token_counts = mean_token_counts.item()  # 转为 python float

        # Then get log probs from old model, here we chunk the inputs into batches in case of OOM.
        old_log_probs_list = []
        need_fixed_old_probs = (epochs_per_rollout_batch * rollout_batch_size) > train_batch_size

        if train_conf["loss_type"] in ["grpo_clip"] and need_fixed_old_probs:
            with torch.no_grad():
                chunk_size = train_conf["chunk_size"]
                for i in range(0, len(rollout_batch_responses), chunk_size):
                    chunk_input_ids = encode_dict["input_ids"][i: i + chunk_size].to(device)
                    chunk_labels = encode_dict["labels"][i: i + chunk_size].to(device)

                    chunk_log_probs = sft_helper.get_response_log_probs(
                        policy_model, chunk_input_ids, chunk_labels, return_token_entropy=False
                    )["log_probs"]

                    old_log_probs_list.append(chunk_log_probs.cpu())
                    del chunk_input_ids, chunk_labels, chunk_log_probs  # 释放引用
                    torch.cuda.empty_cache()

            old_log_probs = torch.cat(old_log_probs_list, dim=0).to(device)
            del old_log_probs_list
            torch.cuda.empty_cache()



        for epoch_idx in range(epochs_per_rollout_batch):
            # for on-policy, that's 1 step only, 1 step = gradient_accumulation_steps of micro training
            # Here we need to compute current log probs
            total_token_counts = 0
            for batch_idx in range(rollout_batch_size // train_batch_size):
                # train per train_batch_size
                total_loss = 0.0
                optimizer.zero_grad()

                for micro_idx in range(gradient_accumulation_steps):
                    # for each micro step, we calculate each micro batch's loss
                    start_idx = micro_idx * micro_batch_size + batch_idx * train_batch_size
                    end_idx = (micro_idx + 1) * micro_batch_size + batch_idx * train_batch_size

                    # Here we need to compute current log probs
                    micro_input_ids = encode_dict["input_ids"][start_idx: end_idx].to(device)
                    micro_labels = encode_dict["labels"][start_idx: end_idx].to(device)
                    micro_mask = encode_dict["response_mask"][start_idx: end_idx].to(device)
                    micro_advantages = advantages[start_idx: end_idx].to(device)
                    micro_raw_rewards = raw_rewards[start_idx: end_idx].to(device)

                    total_token_counts += torch.sum(micro_mask).item()

                    log_probs_dict = sft_helper.get_response_log_probs(
                        policy_model,
                        micro_input_ids,
                        micro_labels,
                        return_token_entropy=False
                    )
                    micro_policy_log_probs = log_probs_dict["log_probs"]
                    if train_conf["loss_type"] in ["grpo_clip"] and need_fixed_old_probs:
                        micro_old_log_probs = old_log_probs[start_idx: end_idx]
                    else:
                        micro_old_log_probs = micro_policy_log_probs.detach()

                    loss, metadata = grpo_helper.grpo_microbatch_train_step(micro_policy_log_probs,
                                                           micro_mask,
                                                           gradient_accumulation_steps,
                                                           train_conf["loss_type"],
                                                           micro_raw_rewards,
                                                           micro_advantages,
                                                           micro_old_log_probs,
                                                           cliprange,
                                                           length_norm
                                                           )
                    total_loss += loss.item()

                # update parameters
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    policy_model.parameters(),
                    max_norm=train_conf["clip_grad"]
                )

                optimizer.step()
                optimizer.zero_grad()

        # The finish of one n-grpo-step
        if step % 2 == 0:
            print(f"Step {step}: Loss = {total_loss:.4f}")
        wandb.log({
            "train/loss": total_loss,
            "train/lr": optimizer.param_groups[0]['lr'],
            "train_rewards_mean": rewards_mean,
            "train_format_reward_mean": format_reward_mean,
            "train/grad_norm": grad_norm.item(),
            "mean_token_counts": mean_token_counts,
            "train_step": step
        })

        # Periodically evaluation
        if step % eval_conf["eval_interval"] == 0:
            policy_evaluation.load_policy_into_vllm_instance(policy_model, vllm_engine)

            eval_dir = f"{config['model']['save_dir']}/{config['logging']['run_name']}/checkpoints"
            if not os.path.exists(eval_dir):
                print(f"Creating evaluation directory: {eval_dir}")
                os.makedirs(eval_dir, exist_ok=True)

            save_path = f"{eval_dir}/eval_step_{step}.jsonl"

            metrics = grpo_evaluator.evaluate_vllm(
                vllm_engine,
                val_problems,
                val_solutions,
                drgrpo_grader.r1_zero_reward_fn,
                eval_sampling_params,
                output_path=save_path
            )

            metrics["eval_step"] = step
            wandb.log(metrics)
            policy_model.train()

            # ===================== save latest =====================
            latest_dir = f"{config['model']['save_dir']}/{config['logging']['run_name']}/latest"
            # save_model(latest_dir)

            # ===================== save best =====================
            cur_metric = metrics.get("avg_reward", None)

            if cur_metric is not None and cur_metric > best_metric:
                best_metric = cur_metric
                best_dir = f"{config['model']['save_dir']}/{config['logging']['run_name']}/best"
                # save_model(best_dir)
                print(f"New best model saved: {best_metric:.4f}")

    ckpt_path = f"{config['model']['save_dir']}/{config['logging']['run_name']}/final"
    save_model(ckpt_path)

    print("Training and Evaluation finished. Exiting...")
    wandb.finish()
    os._exit(0)


if __name__ == "__main__":
    config = load_config("/root/autodl-tmp/cs336/assignment5-alignment-main/cs336_alignment/grpo/grpo_train.yaml")
    print("Config loaded:", config)

    wandb.init(
        project=config["logging"]["project_name"],
        name=config["logging"]["run_name"],
        config=config
    )

    # Loading policy model and tokenizer on cuda 0
    model_conf = config["model"]
    train_conf = config["training"]

    print("Loading Training Model...")
    policy_model = AutoModelForCausalLM.from_pretrained(
        model_conf['model_id'],
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        attn_implementation="flash_attention_2",
    )

    policy_model.config.use_cache = False
    policy_model.gradient_checkpointing_enable()


    tokenizer = AutoTokenizer.from_pretrained(model_conf['tokenizer_path'])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Loading eval model on cuda 1
    # we are not doing this for this task.
    # eval_model = AutoModelForCausalLM.from_pretrained(
    #     model_conf['model_id'],
    #     torch_dtype=torch.bfloat16,
    #     attn_implementation="flash_attention_2",
    #     device_map=config['evaluation']['vllm_device']
    # )
    # eval_model.eval()

    print(f"Initializing Optimizer with LR: {train_conf['learning_rate']}...")
    optimizer = torch.optim.AdamW(
        policy_model.parameters(),
        lr=float(train_conf['learning_rate']),
        weight_decay=0.0,
        betas=(0.9, 0.95),
    )

    print("Model and Optimizer ready.")

    print("Initializing vLLM for evaluation...")
    try:
        vllm_engine = policy_evaluation.init_vllm(
            model_id=config['model']['model_id'],
            device=config["evaluation"]["vllm_device"],
            seed=config['training']['seed'],
            gpu_memory_utilization=config['evaluation']['gpu_memory_utilization'],
            max_model_len=config['evaluation']['max_model_len'],
            enforce_eager = False
        )
    except Exception as e:
        print(f"Warning: Failed to initialize vLLM ({e}). Evaluation will be skipped.")
        vllm_engine = None

    # Load train prompts and ground-truth
    print(f"Loading training data from {config['data']['train_path']}...")
    with open(config['data']['train_path'], 'r', encoding='utf-8') as f:
        train_data_list = json.load(f)

    train_prompts = []
    train_solutions = []

    with open("/root/autodl-tmp/cs336/assignment5-alignment-main/cs336_alignment/prompts/r1_zero.prompt", "r", encoding="utf-8") as f:
        prompt_template = f.read().strip()

    for item in train_data_list:
        pt = prompt_template.format(question=item["problem"])
        train_prompts.append(pt)
        train_solutions.append(str(item['expected_answer']))

    # Load val problems and ground_truth
    print(f"Loading validation data from {config['data']['val_path']}...")
    with open(config['data']['val_path'], 'r', encoding='utf-8') as f:
        data_list = json.load(f)

    val_problems = [item['problem'] for item in data_list]
    val_solutions = [str(item['expected_answer']) for item in data_list]

    train_loop(policy_model, vllm_engine, optimizer, tokenizer, config, train_prompts, val_problems, train_solutions, val_solutions)
