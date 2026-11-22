"""Greedy decode variants that share arithmetic, so outputs can be compared token for token.

  day1           - `01-inference-loop`'s `generate_cached` unchanged (batch 1, sync + .item() every step)
  batched        - same loop shape as `01-inference-loop` but for B equal-length prompts (one host sync/step)
  deferred_sync  - batch 1, tokens stay on the device; ONE sync after the last step

All variants run a fixed number of tokens (EOS is suppressed, as in `01-inference-loop`'s experiments),
because the deferred variant cannot see a stop token without syncing.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Sequence

import _day1  # noqa: F401
import torch
from device import MemoryTracker, sync
from generate import _select_next_token, generate_cached  # shared arithmetic, see module doc


@dataclass(frozen=True)
class DecodeRun:
    tokens: tuple[tuple[int, ...], ...]  # [batch][generated token]
    prefill_s: float  # first forward + first token on host (TTFT)
    decode_s: float  # wall time of all decode steps
    decode_steps: int
    step_times_s: tuple[float, ...]  # per step; empty when steps are not individually timed
    peak_mem_bytes: int

    @property
    def batch_size(self) -> int:
        return len(self.tokens)

    @property
    def output_tokens(self) -> int:
        return sum(len(row) for row in self.tokens)

    @property
    def total_s(self) -> float:
        return self.prefill_s + self.decode_s

    @property
    def tpot_s(self) -> float:
        """Mean time per output token for each request: one step produces one token per row."""
        return self.decode_s / self.decode_steps if self.decode_steps else 0.0


def make_batch_ids(single_ids: torch.Tensor, batch_size: int) -> torch.Tensor:
    """B distinct, equal-length prompts from one [1, T] prompt (row i is rolled by 7*i tokens)."""
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    rows = [torch.roll(single_ids[0], shifts=7 * i) for i in range(batch_size)]
    return torch.stack(rows)


def _suppressed(eos_ids: Sequence[int], device: torch.device) -> torch.Tensor:
    return torch.tensor(sorted(set(eos_ids)), device=device)


def run_day1(model, input_ids: torch.Tensor, n_new: int, eos_ids: Sequence[int]) -> DecodeRun:
    result = generate_cached(model, input_ids, n_new, eos_ids, suppress_eos=True)
    return DecodeRun(
        tokens=(result.token_ids,),
        prefill_s=result.prefill_s,
        decode_s=sum(result.step_times_s),
        decode_steps=len(result.step_times_s),
        step_times_s=result.step_times_s,
        peak_mem_bytes=result.peak_mem_bytes,
    )


def run_batched(model, input_ids: torch.Tensor, n_new: int, eos_ids: Sequence[int]) -> DecodeRun:
    """Cached greedy decode for [B, T] prompts; each step syncs once via `.tolist()`."""
    device = input_ids.device
    suppressed = _suppressed(eos_ids, device)
    tracker = MemoryTracker(device)
    tracker.reset()
    rows: list[list[int]] = [[] for _ in range(input_ids.shape[0])]
    step_times: list[float] = []
    prefill_s = 0.0
    cache = None
    next_token = input_ids

    with torch.no_grad():
        for step in range(n_new):
            sync(device)
            start = time.perf_counter()
            out = model(input_ids=next_token, past_key_values=cache, use_cache=True, logits_to_keep=1)
            cache = out.past_key_values
            next_token = _select_next_token(out.logits, suppressed)
            for row, token in zip(rows, next_token[:, 0].tolist()):  # host sync
                row.append(token)
            sync(device)
            elapsed = time.perf_counter() - start
            tracker.sample()
            if step == 0:
                prefill_s = elapsed
            else:
                step_times.append(elapsed)

    return DecodeRun(
        tokens=tuple(tuple(r) for r in rows),
        prefill_s=prefill_s,
        decode_s=sum(step_times),
        decode_steps=len(step_times),
        step_times_s=tuple(step_times),
        peak_mem_bytes=tracker.peak_bytes,
    )


def run_deferred_sync(
    model, input_ids: torch.Tensor, n_new: int, eos_ids: Sequence[int]
) -> DecodeRun:
    """Batch-1 cached decode with no per-step host sync; only prefill and the end are synced."""
    device = input_ids.device
    suppressed = _suppressed(eos_ids, device)
    tracker = MemoryTracker(device)
    tracker.reset()
    generated: list[torch.Tensor] = []

    with torch.no_grad():
        sync(device)
        start = time.perf_counter()
        out = model(input_ids=input_ids, use_cache=True, logits_to_keep=1)
        cache = out.past_key_values
        next_token = _select_next_token(out.logits, suppressed)
        generated.append(next_token)
        first = int(next_token.item())  # TTFT needs the first token on the host
        sync(device)
        prefill_s = time.perf_counter() - start
        tracker.sample()

        decode_start = time.perf_counter()
        for _ in range(n_new - 1):
            out = model(input_ids=next_token, past_key_values=cache, use_cache=True, logits_to_keep=1)
            cache = out.past_key_values
            next_token = _select_next_token(out.logits, suppressed)
            generated.append(next_token)
            tracker.sample()
        tokens = torch.cat(generated, dim=1)[0].tolist()  # the one deferred host sync
        sync(device)
        decode_s = time.perf_counter() - decode_start

    if tokens[0] != first:
        raise RuntimeError("first token changed between prefill readback and final readback")
    return DecodeRun(
        tokens=(tuple(tokens),),
        prefill_s=prefill_s,
        decode_s=decode_s,
        decode_steps=n_new - 1,
        step_times_s=(),
        peak_mem_bytes=tracker.peak_bytes,
    )
