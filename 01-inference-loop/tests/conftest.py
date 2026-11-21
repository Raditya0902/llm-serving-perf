import sys
from pathlib import Path

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

TINY_VOCAB = 128


@pytest.fixture(scope="session")
def tiny_model():
    """Small random Qwen2 (fp32, CPU): fast, deterministic, same architecture family."""
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
def prompt_ids():
    generator = torch.Generator().manual_seed(1)
    return torch.randint(0, TINY_VOCAB, (1, 12), generator=generator)
