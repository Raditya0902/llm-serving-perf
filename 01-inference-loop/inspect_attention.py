"""Print the real tensor shapes inside one Qwen2 attention layer, for prefill and decode.

Uses forward hooks on layer 0's attention sub-modules and the KV cache itself.
"""

from __future__ import annotations

import torch

from device import pick_device
from generate import DEFAULT_MODEL_ID
from model_io import load

PROMPT_LEN = 10


def _shape(t: torch.Tensor) -> list[int]:
    return list(t.shape)


def _capture(model, ids: torch.Tensor, cache=None):
    """Run one forward pass; return (shapes recorded by hooks, new cache)."""
    attn = model.model.layers[0].self_attn
    shapes: dict[str, list[int]] = {}

    def hook_for(name: str):
        def hook(_module, inputs, output):
            tensor = output[0] if isinstance(output, tuple) else output
            shapes[name] = _shape(tensor)
            if name == "q_proj":
                shapes["input hidden_states"] = _shape(inputs[0])
        return hook

    handles = [
        getattr(attn, name).register_forward_hook(hook_for(name))
        for name in ("q_proj", "k_proj", "v_proj", "o_proj")
    ]
    try:
        with torch.no_grad():
            out = model(input_ids=ids, past_key_values=cache, use_cache=True)
    finally:
        for handle in handles:
            handle.remove()
    layer_cache = out.past_key_values.layers[0]
    shapes["cached K (layer 0)"] = _shape(layer_cache.keys)
    shapes["cached V (layer 0)"] = _shape(layer_cache.values)
    shapes["logits (all positions)"] = _shape(out.logits)
    return shapes, out.past_key_values


def _print(title: str, shapes: dict[str, list[int]]) -> None:
    print(f"\n{title}")
    for name, shape in shapes.items():
        print(f"  {name:<24} {shape}")


def _main() -> None:
    device = pick_device()
    _, model = load(DEFAULT_MODEL_ID, device)
    cfg = model.config
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    print(
        f"hidden={cfg.hidden_size} q_heads={cfg.num_attention_heads} "
        f"kv_heads={cfg.num_key_value_heads} head_dim={head_dim} "
        f"layers={cfg.num_hidden_layers} vocab={cfg.vocab_size}"
    )

    prompt = torch.randint(0, cfg.vocab_size, (1, PROMPT_LEN), device=device)
    prefill_shapes, cache = _capture(model, prompt)
    _print(f"PREFILL (B=1, T={PROMPT_LEN})", prefill_shapes)

    next_id = torch.randint(0, cfg.vocab_size, (1, 1), device=device)
    decode_shapes, _ = _capture(model, next_id, cache)
    _print(f"DECODE (B=1, 1 new token, {PROMPT_LEN} cached)", decode_shapes)


if __name__ == "__main__":
    _main()
