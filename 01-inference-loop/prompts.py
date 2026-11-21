"""Synthetic prompts with an exact token length, for reproducible benchmarks."""

from __future__ import annotations

import torch

FILLER_PASSAGE = (
    "Large language models generate text one token at a time. Each step reads the "
    "tokens produced so far, runs them through a stack of transformer layers, and "
    "predicts a distribution over the next token. "
)


def make_prompt_ids(tokenizer, n_tokens: int) -> torch.Tensor:
    """Tile a fixed passage and truncate to exactly `n_tokens` ids, shape [1, n_tokens]."""
    if n_tokens <= 0:
        raise ValueError("n_tokens must be positive")
    base = tokenizer(FILLER_PASSAGE, add_special_tokens=False, return_tensors="pt").input_ids[0]
    repeats = -(-n_tokens // base.numel())  # ceil division
    return base.repeat(repeats)[:n_tokens].unsqueeze(0)
