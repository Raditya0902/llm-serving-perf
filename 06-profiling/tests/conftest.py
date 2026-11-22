"""Shared fixtures: a tiny random Qwen2 model on CPU (no downloads) and the `06-profiling` import path."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

TINY_VOCAB = 128
TINY_EOS = (1, 2)
SYNTHETIC_NOTE = "SYNTHETIC TEST DATA - not a measurement"


@pytest.fixture(scope="session")
def tiny_model() -> Qwen2ForCausalLM:
    torch.manual_seed(0)
    config = Qwen2Config(
        vocab_size=TINY_VOCAB,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=512,
    )
    return Qwen2ForCausalLM(config).eval()


@pytest.fixture()
def prompt_ids() -> torch.Tensor:
    generator = torch.Generator().manual_seed(0)
    # Avoid the EOS ids so runs are not cut short.
    return torch.randint(3, TINY_VOCAB, (1, 12), generator=generator)
