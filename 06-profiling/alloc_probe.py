"""Are device allocations happening while we generate?

Reads the allocator's live-bytes counter after every decode step. With a KV cache the
expected growth is exactly one token's K/V per step (`kv_bytes_per_token`); anything
beyond that is churn (temporaries, cache re-allocation). CPU has no such counter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import _day1  # noqa: F401
import torch
from device import sync
from generate import _select_next_token
from run_experiments import kv_bytes_per_token  # `01-inference-loop`: 2 * layers * kv_heads * head_dim * bytes

KV_MATCH_TOLERANCE = 0.25  # a step "matches" if its growth is within 25% of one token's KV
KV_MATCH_MIN_FRACTION = 0.9


@dataclass(frozen=True)
class AllocReport:
    available: bool
    reason: str
    deltas: tuple[int, ...] = ()
    steps_with_growth: int = 0
    mean_growth_per_step: float = 0.0
    expected_kv_bytes_per_step: int = 0
    growth_matches_kv: bool = False


def allocated_bytes_reader(device: torch.device) -> Callable[[], int] | None:
    """Live-tensor-bytes counter for `device`, or None where no such counter exists."""
    if device.type == "cuda":
        return lambda: int(torch.cuda.memory_allocated(device))
    if device.type == "mps":
        return lambda: int(torch.mps.current_allocated_memory())
    return None


def build_alloc_report(
    allocated_per_step: Sequence[int], expected_kv_bytes_per_step: int
) -> AllocReport:
    if len(allocated_per_step) < 2:
        raise ValueError("need at least two samples to compute a delta")
    deltas = tuple(b - a for a, b in zip(allocated_per_step, allocated_per_step[1:]))
    close = [
        abs(d - expected_kv_bytes_per_step) <= KV_MATCH_TOLERANCE * expected_kv_bytes_per_step
        for d in deltas
    ]
    return AllocReport(
        available=True,
        reason="",
        deltas=deltas,
        steps_with_growth=sum(1 for d in deltas if d > 0),
        mean_growth_per_step=sum(deltas) / len(deltas),
        expected_kv_bytes_per_step=expected_kv_bytes_per_step,
        growth_matches_kv=sum(close) / len(close) >= KV_MATCH_MIN_FRACTION,
    )


def measure_allocations(model, input_ids: torch.Tensor, n_steps: int) -> AllocReport:
    device = input_ids.device
    reader = allocated_bytes_reader(device)
    if reader is None:
        return AllocReport(False, "no allocator counter on cpu; use the profiler's [memory] events")
    samples: list[int] = []
    with torch.no_grad():
        out = model(input_ids=input_ids, use_cache=True, logits_to_keep=1)
        cache = out.past_key_values
        next_token = _select_next_token(out.logits, None)
        sync(device)
        samples.append(reader())
        for _ in range(n_steps):
            out = model(input_ids=next_token, past_key_values=cache, use_cache=True, logits_to_keep=1)
            cache = out.past_key_values
            next_token = _select_next_token(out.logits, None)
            sync(device)
            samples.append(reader())
    return build_alloc_report(samples, kv_bytes_per_token(model))
