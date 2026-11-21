"""Hand-written greedy autoregressive decoding, with and without a KV cache.

Both loops share the same token-selection and stopping logic; the *only*
difference is what is fed to the model each step:

  uncached: the whole sequence so far          -> O(n) work per step, O(n^2) total
  cached:   one new token + the stored K/V     -> O(1) new projections per step
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Callable, Sequence

import torch

from device import MemoryTracker, sync

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"


@dataclass(frozen=True)
class GenerationResult:
    token_ids: tuple[int, ...]
    prefill_s: float  # first forward pass + first token (i.e. TTFT)
    step_times_s: tuple[float, ...]  # one entry per token after the first
    peak_mem_bytes: int
    stopped_on_eos: bool

    @property
    def total_s(self) -> float:
        return self.prefill_s + sum(self.step_times_s)

    @property
    def decode_s_per_token(self) -> float:
        if not self.step_times_s:
            return 0.0
        return sum(self.step_times_s) / len(self.step_times_s)


def _validate(input_ids: torch.Tensor, max_new_tokens: int) -> None:
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be >= 1")
    if input_ids.dim() != 2 or input_ids.shape[0] != 1:
        raise ValueError("input_ids must have shape [1, seq_len] (batch size 1)")


def _select_next_token(
    logits: torch.Tensor, suppressed_ids: torch.Tensor | None
) -> torch.Tensor:
    """Greedy pick from the last position's logits. [1, 1, V] -> [1, 1].

    `suppressed_ids` (prebuilt once per run, outside the timed loop) are masked to -inf.
    """
    next_logits = logits[:, -1, :]
    if suppressed_ids is not None:
        next_logits = next_logits.index_fill(-1, suppressed_ids, float("-inf"))
    return next_logits.argmax(dim=-1, keepdim=True)


def _run(
    model,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    eos_ids: Sequence[int],
    suppress_eos: bool,
    use_cache: bool,
    tracker: MemoryTracker | None,
    on_step: Callable[[int], None] | None,
) -> GenerationResult:
    _validate(input_ids, max_new_tokens)
    device = input_ids.device
    eos_set = frozenset(eos_ids)
    suppressed_ids = (
        torch.tensor(sorted(eos_set), device=device) if suppress_eos and eos_set else None
    )
    tracker = tracker or MemoryTracker(device)
    tracker.reset()

    tokens: list[int] = []
    step_times: list[float] = []
    prefill_s = 0.0
    sequence = input_ids
    cache = None
    next_token = input_ids  # fed to the model on the first (prefill) step

    with torch.no_grad():
        for step in range(max_new_tokens):
            if on_step is not None and step > 0 and use_cache:
                on_step(cache.get_seq_length())

            sync(device)
            start = time.perf_counter()

            if use_cache:
                out = model(
                    input_ids=next_token,
                    past_key_values=cache,
                    use_cache=True,
                    logits_to_keep=1,
                )
                cache = out.past_key_values
            else:
                out = model(input_ids=sequence, use_cache=False, logits_to_keep=1)

            next_token = _select_next_token(out.logits, suppressed_ids)
            token_id = int(next_token.item())  # .item() also forces a host sync

            sync(device)
            elapsed = time.perf_counter() - start
            tracker.sample()

            tokens.append(token_id)
            if step == 0:
                prefill_s = elapsed
            else:
                step_times.append(elapsed)
            if not use_cache:
                sequence = torch.cat([sequence, next_token], dim=1)
            if token_id in eos_set:
                break

    return GenerationResult(
        token_ids=tuple(tokens),
        prefill_s=prefill_s,
        step_times_s=tuple(step_times),
        peak_mem_bytes=tracker.peak_bytes,
        stopped_on_eos=bool(tokens) and tokens[-1] in eos_set,
    )


def generate_uncached(
    model,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    eos_ids: Sequence[int],
    suppress_eos: bool = False,
    tracker: MemoryTracker | None = None,
) -> GenerationResult:
    """Greedy decode, recomputing the full sequence every step (no KV cache)."""
    return _run(model, input_ids, max_new_tokens, eos_ids, suppress_eos, False, tracker, None)


def generate_cached(
    model,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    eos_ids: Sequence[int],
    suppress_eos: bool = False,
    tracker: MemoryTracker | None = None,
    on_step: Callable[[int], None] | None = None,
) -> GenerationResult:
    """Greedy decode with `past_key_values`: prefill once, then feed one token per step.

    `on_step(cache_len)` is called before each decode step (test/inspection hook).
    """
    return _run(model, input_ids, max_new_tokens, eos_ids, suppress_eos, True, tracker, on_step)


def _main() -> None:
    from device import pick_device
    from model_io import eos_token_ids, load

    parser = argparse.ArgumentParser(description="Greedy generation demo")
    parser.add_argument("--prompt", default="Explain the KV cache in one sentence.")
    parser.add_argument("--mode", choices=["cached", "uncached"], default="cached")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = pick_device(args.device)
    tokenizer, model = load(DEFAULT_MODEL_ID, device)
    chat = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    input_ids = tokenizer(chat, return_tensors="pt").input_ids.to(device)

    fn = generate_cached if args.mode == "cached" else generate_uncached
    result = fn(model, input_ids, args.max_new_tokens, eos_token_ids(tokenizer))

    print(tokenizer.decode(result.token_ids, skip_special_tokens=True))
    print(
        f"[{args.mode} on {device}] {len(result.token_ids)} tokens | "
        f"prefill {result.prefill_s * 1e3:.1f} ms | "
        f"decode {result.decode_s_per_token * 1e3:.1f} ms/token | "
        f"total {result.total_s:.2f} s"
    )


if __name__ == "__main__":
    _main()
