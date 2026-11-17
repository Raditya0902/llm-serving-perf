"""Model loading with an explicit attention implementation (Day 1's `load` has no such knob)."""

from __future__ import annotations

import _day1  # noqa: F401
import torch
from generate import DEFAULT_MODEL_ID
from model_io import default_dtype
from transformers import AutoModelForCausalLM, AutoTokenizer

ATTN_IMPLS = ("sdpa", "eager")


def validate_attn_impl(attn_impl: str) -> str:
    if attn_impl not in ATTN_IMPLS:
        raise ValueError(f"attn_impl must be one of {ATTN_IMPLS}, got {attn_impl!r}")
    return attn_impl


def load_model(
    device: torch.device,
    attn_impl: str = "sdpa",
    model_id: str = DEFAULT_MODEL_ID,
):
    """Return `(tokenizer, model)`; same dtype policy as Day 1 (fp16 on accelerators, fp32 on CPU)."""
    validate_attn_impl(attn_impl)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = (
        AutoModelForCausalLM.from_pretrained(
            model_id, dtype=default_dtype(device), attn_implementation=attn_impl
        )
        .to(device)
        .eval()
    )
    return tokenizer, model
