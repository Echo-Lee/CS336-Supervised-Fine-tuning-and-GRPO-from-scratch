import torch


def tokenize_prompt_and_output(prompt_strs, output_strs, tokenizer):
    """
    Tokenize the prompt and output strings, list[str]
    and construct a mask that is 1 for the 'response' tokens and 0 for 'prompt or paddings'

    Return: dict[str, torch.Tensor]
        input_ids: torch.Tensor(batch_size, max(prompt_and_output_lens) - 1), tokenized prompt and output strings with final token sliced off

        labels: torch.Tensor(batch_size, max(prompt_and_output_lens) - 1), shifted tokenized prompt and output strings without the first token

        response_mask: torch.Tensor(batch_size, max(prompt_and_output_lens) - 1), a boolean mask, True for tokens in response. False for padding and prompts.
    """
    # 对prompt和output分别encode
    prompt_encodings = tokenizer(prompt_strs, padding=False) # return a list
    output_encodings = tokenizer(output_strs, padding=False)

    prompt_ids_list = prompt_encodings["input_ids"]
    output_ids_list = output_encodings["input_ids"]


    pair_lens = [len(p) + len(o) for p, o in zip(prompt_ids_list, output_ids_list)]
    max_prompt_and_output_lens = max(pair_lens)
    batch_size = len(prompt_strs)

    input_ids = torch.full((batch_size, max_prompt_and_output_lens), tokenizer.pad_token_id, dtype=torch.long)
    response_mask = torch.zeros((batch_size, max_prompt_and_output_lens), dtype=torch.bool)

    for i in range(batch_size):
        p_ids = prompt_ids_list[i]
        o_ids = output_ids_list[i]
        p_len = len(p_ids)
        o_len = len(o_ids)

        input_ids[i, :p_len] = torch.tensor(p_ids)
        input_ids[i, p_len:p_len+o_len] = torch.tensor(o_ids)

        response_mask[i, p_len:p_len+o_len] = True

    return {
        "input_ids": input_ids[:, :-1],
        "labels": input_ids[:, 1:],
        "response_mask": response_mask[:, 1:]
    }


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """
    logits: torch.Tensor(Batch_size, sequence_length, vocab_size), unnormalized.
    return: torch.Tensor(Batch_size, sequence_length).
    Using numerically stable methods.
    Pay attention to dtype.
    """
    original_dtype = logits.dtype
    logits = logits.to(torch.float32)

    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

    del logits

    probs = torch.exp(log_probs)

    entropy = -torch.sum(probs * log_probs, dim=-1)
    return entropy.to(original_dtype)

def get_response_log_probs(model,
                           input_ids: torch.Tensor,
                           labels: torch.Tensor,
                           return_token_entropy: bool=False) -> dict[str, torch.Tensor]:
    """
    input_ids / labels: torch.Tensor (batch_size, sequence_length), prompt + response by tokenization method before
    return_token_entropy: if True, return per-token entropy
    Returns:
        "log_probs": shape (batch_Size, sequence length)
        "token_entropy": shape (batch_size, sequence length)
    """
    logits = model(input_ids).logits # (batch, seq, vocab)

    # 转换类型并保证数值稳定
    log_probs = -torch.nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
        reduction='none'
    ).view(labels.size())

    if return_token_entropy:
        # 确保 compute_entropy 也使用了数值稳定的方法
        entropy = compute_entropy(logits)
        return {"log_probs": log_probs, "token_entropy": entropy}

    return {"log_probs": log_probs}

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

def sft_microbatch_train_step(
        policy_log_probs: torch.Tensor,
        response_mask: torch.Tensor,
        gradient_accumulation_steps: int,
        normalize_constant: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    policy_log_probs: (batch_size, sequence_length), including prompts and response. per-token log-probs from SFT policy model being trained
    response_mask: (batch_size, sequence_length)
    gradient_accumulation_steps: number of microbatches per opt step

    Return:
        tuple [torch.Tensor, dict[str, torch.Tensor]]
        loss: scaler tensor.
        metadata: Dict with metadata
    """
    num_tokens = response_mask.sum()
    if num_tokens == 0:
        num_tokens = torch.tensor(1.0, device=response_mask.device)

    loss = -masked_normalize(policy_log_probs, response_mask, normalize_constant=normalize_constant, dim=None) / gradient_accumulation_steps / num_tokens
    loss.backward()

    metadata = {
        "loss": loss.detach(),
        "num_tokens": response_mask.sum().detach()
    }

    return (loss.detach(), metadata)
