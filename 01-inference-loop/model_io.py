"""Tokenizer + model loading."""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

EOS_TOKENS = ("<|im_end|>", "<|endoftext|>")


def default_dtype(device: torch.device) -> torch.dtype:
    """fp16 on accelerators (matches typical serving), fp32 on CPU (fp16 is slow/unsupported there)."""
    return torch.float32 if device.type == "cpu" else torch.float16


def load(model_id: str, device: torch.device, dtype: torch.dtype | None = None):
    """Return `(tokenizer, model)` with the model in eval mode on `device`."""
    dtype = dtype or default_dtype(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(device).eval()
    return tokenizer, model


def eos_token_ids(tokenizer) -> tuple[int, ...]:
    """Ids that end a Qwen chat turn (<|im_end|>) or document (<|endoftext|>)."""
    ids = tokenizer.convert_tokens_to_ids(list(EOS_TOKENS))
    unknown = getattr(tokenizer, "unk_token_id", None)
    if any(i is None or i == unknown for i in ids):
        raise ValueError(f"tokenizer is missing one of the EOS tokens {EOS_TOKENS}: got {ids}")
    return tuple(ids)
