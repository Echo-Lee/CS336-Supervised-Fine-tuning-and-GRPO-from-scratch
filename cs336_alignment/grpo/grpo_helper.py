import torch


def compute_group_normalized_rewards(
        reward_fn,
        rollout_responses,
        repeated_ground_truths,
        group_size,
        advantage_eps,
        normalize_by_std
):
    """
    Args:
        reward_fn: Callable[[str, str], dict[str, float]] Scores the rollout responses against the ground truths, producing
        a dict with keys: "reward", "format reward" and "answer reward".
        rollout_responses: list[str], length = rollout_batch_size = n_prompts_per_rollout_batch * group_size. e.g. ['A1', 'A2', 'A3', 'B1', 'B2', 'B3'].
        repeated_ground_truths: corresponds to rollout_responses.
        normalize_by_std: bool If True,divide by the per-group standard deviation; otherwise subtract only the group mean.

    Returns:
        tuple[torch.Tensor, torch.Tensor, dict[str, float]].
        advantages (rollout_batch_size,). Group Normed rewards.
        raw_rewards (rollout_batch_size,). Unnormed rewards.
        metadata (mean, std, max, min, etc. of rewards).
    """
    raw_rewards = []
    format_raw_rewards = []
    for idx, response in enumerate(rollout_responses):
        grades = reward_fn(response, repeated_ground_truths[idx])
        raw_rewards.append(grades["reward"])
        format_raw_rewards.append(grades["format_reward"])

    raw_rewards = torch.tensor(raw_rewards, dtype=torch.float32)
    raw_rewards_reshaped = raw_rewards.view(-1, group_size)
    format_raw_rewards = torch.tensor(format_raw_rewards, dtype=torch.float32)
    rewards_mean = raw_rewards_reshaped.mean(dim=1, keepdim=True)
    rewards_std = raw_rewards_reshaped.std(dim=1, keepdim=True)

    if normalize_by_std:
        advantages = (raw_rewards_reshaped - rewards_mean) / (rewards_std + advantage_eps)
        advantages = advantages.reshape(-1)
        # advantages = advantages.to(torch.bfloat16)

    else:
        advantages = (raw_rewards_reshaped - rewards_mean)
        advantages = advantages.reshape(-1)
        # advantages = advantages.to(torch.bfloat16)

    # raw_rewards = raw_rewards.to(torch.bfloat16)

    metadata = {"rewards_mean": raw_rewards.mean(),
                "format_reward_mean": format_raw_rewards.mean()}
    return advantages, raw_rewards, metadata


def compute_naive_policy_gradient_loss(
        raw_rewards_or_advantages: torch.Tensor,
        policy_log_probs: torch.Tensor,
) -> torch.Tensor:
    """
    raw_rewards_or_advantages: (batch_size, 1), scaler reward/advantage for each rollout response
    policy_log_probs: Shape(batch_size,sequence_length)

    return Shape(batch_size, sequence_length), per-token policy-gradient loss to be aggregated across the batch and sequence dims in training loop.
    """
    return -raw_rewards_or_advantages * policy_log_probs


def compute_grpo_clip_loss(
        advantages: torch.Tensor,
        policy_log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        cliprange: float
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Args:
        advantages: torch.Tensor (batch_size, 1), per-example advantages A.
        policy_log_probs / old_log_probs: Tensor (batch_size, sequence_length).
        cliprange: clip parameters.
    Returns:
        loss: (batch_size, sequence_length), per-token clipped loss
        metadata: is_clipped: whether the RHS is smaller than LHS. Note: when clipped, the value always falls on RHS.
    """

    original_dtype = policy_log_probs.dtype
    policy_log_probs = policy_log_probs.to(torch.float32)
    old_log_probs = old_log_probs.to(torch.float32)

    ratio = torch.exp(policy_log_probs - old_log_probs)

    surr1 = ratio * advantages
    surr2 = ratio.clamp(1.0 - cliprange, 1.0 + cliprange) * advantages

    loss = -torch.minimum(surr1, surr2)
    metadata = {"is_clipped": (surr1 > surr2).detach(), "clipped_ratio": (surr1 > surr2).to(torch.float16).mean().detach()}

    policy_log_probs = policy_log_probs.to(original_dtype)
    old_log_probs = old_log_probs.to(original_dtype)

    return loss.to(original_dtype), metadata

def compute_policy_gradient_loss(
        policy_log_probs,
        loss_type,
        raw_rewards,
        advantages,
        old_log_probs,
        cliprange
) ->  tuple[torch.Tensor, dict[str, torch.Tensor]]:
    assert loss_type in ["no_baseline", "reinforce_with_baseline", "grpo_clip"]
    if advantages.dim() == 1:
        advantages = advantages.unsqueeze(1)
    if raw_rewards.dim() == 1:
        raw_rewards = raw_rewards.unsqueeze(1)
    if loss_type == "no_baseline":
        loss = compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs)
        metadata = {}
    elif loss_type == "reinforce_with_baseline":
        loss = compute_naive_policy_gradient_loss(advantages, policy_log_probs)
        metadata = {}
    else:
        loss, metadata = compute_grpo_clip_loss(advantages, policy_log_probs, old_log_probs, cliprange)
    return loss, metadata

def masked_mean(
        tensor,
        mask,
        dim
) -> torch.Tensor:
    """
    Returns: The masked mean with mean(dim), over masked == 1 elements
    """
    if dim is None:
        return torch.sum(tensor * mask) / torch.sum(mask)
    else:
        return torch.sum(tensor * mask, dim=dim) / torch.sum(mask, dim=dim)

def masked_normalize(tensor: torch.Tensor,
                     mask: torch.Tensor,
                     normalize_constant: float=1.0,
                     dim: int | None = None,) -> torch.Tensor:
    """
    tensor: the tensor to sun and normalize.
    mask: same shape with tensor, positions with 1 are included in the sum.
    normalize_constant: the constant to divide by for norm.
    dim: the dimension to sum along before norm, if None, sum over all dimensions.

    Returns: the masked normalized sum
    """
    masked_tensor = tensor * mask
    if dim is not None:
        return torch.sum(masked_tensor, dim=dim) / normalize_constant
    else:
        return torch.sum(masked_tensor) / normalize_constant

def grpo_microbatch_train_step(
        policy_log_probs,
        response_mask,
        gradient_accumulation_steps,
        loss_type,
        raw_rewards,
        advantages,
        old_log_probs,
        cliprange,
        length_norm = "masked_mean"
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Execute a forward and backward pass on a microbatch.

    Return: loss, scalar tensor, the microbatch adjusted for gradient accumulation.
    metadata.
    """
    assert length_norm in ["masked_mean", "masked_normalize"]
    loss, metadata = compute_policy_gradient_loss(policy_log_probs, loss_type, raw_rewards, advantages, old_log_probs, cliprange)
    if length_norm == "masked_mean":
        loss = masked_mean(loss, response_mask, dim=None) # scalar loss
    elif length_norm == "masked_normalize":
        batch_total_unmasked_tokens = response_mask.sum()
        loss = masked_normalize(loss, response_mask, normalize_constant=batch_total_unmasked_tokens, dim=None)

    loss = loss / gradient_accumulation_steps
    loss.backward()

    return loss, metadata

from torch.utils.data import DataLoader, Dataset

class PromptDataset(Dataset):
    def __init__(self, prompts, answers):
        self.prompts = prompts
        self.answers = answers

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return self.prompts[idx], self.answers[idx]