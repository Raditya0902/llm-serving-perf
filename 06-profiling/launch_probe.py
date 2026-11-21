"""Is the CPU delaying the device? Compare enqueue time with synced step time.

Per decode step we time two things on the same forward pass:
  enqueue - from the call to the moment the Python-side forward returns (kernel launches
            queued, no wait for the device)
  step    - the same call plus a device sync (enqueue + whatever the device still had to do)

enqueue / step near 1 means the device finished almost as soon as the CPU stopped issuing
work, i.e. the CPU (Python + dispatch + launch) is the bottleneck. Near 0 means the
device is the bottleneck and the CPU runs far ahead of it.

On a CPU "device" the two are identical by construction, so the verdict is meaningless there.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass

import _day1  # noqa: F401
import torch
from device import sync
from generate import _select_next_token

CPU_BOUND_RATIO = 0.9
DEVICE_BOUND_RATIO = 0.5


@dataclass(frozen=True)
class LaunchReport:
    n_steps: int
    enqueue_ms_median: float
    step_ms_median: float
    enqueue_fraction: float
    verdict: str


def launch_verdict(enqueue_fraction: float) -> str:
    if enqueue_fraction >= CPU_BOUND_RATIO:
        return "cpu-bound"
    if enqueue_fraction < DEVICE_BOUND_RATIO:
        return "device-bound"
    return "mixed"


def measure_launch(model, input_ids: torch.Tensor, n_steps: int) -> LaunchReport:
    """Prefill once, then time `n_steps` cached decode steps (greedy, EOS-agnostic)."""
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1")
    device = input_ids.device
    enqueue_s: list[float] = []
    step_s: list[float] = []
    with torch.no_grad():
        out = model(input_ids=input_ids, use_cache=True, logits_to_keep=1)
        cache = out.past_key_values
        next_token = _select_next_token(out.logits, None)
        sync(device)
        for _ in range(n_steps):
            sync(device)
            start = time.perf_counter()
            out = model(input_ids=next_token, past_key_values=cache, use_cache=True, logits_to_keep=1)
            next_token = _select_next_token(out.logits, None)
            enqueued = time.perf_counter()
            sync(device)
            done = time.perf_counter()
            cache = out.past_key_values
            enqueue_s.append(enqueued - start)
            step_s.append(done - start)
    enqueue_ms = statistics.median(enqueue_s) * 1e3
    step_ms = statistics.median(step_s) * 1e3
    fraction = enqueue_ms / step_ms
    return LaunchReport(n_steps, enqueue_ms, step_ms, fraction, launch_verdict(fraction))
