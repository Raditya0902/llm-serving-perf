"""Synthetic, seeded workloads.

`generate_workload` builds the main 100-request mix used by experiment A:

* prompts : 70% short U[32, 256], 30% long U[1024, 2048]      (drawn independently of outputs)
* outputs : 60% short U[16, 64],  40% long U[200, 512]
* arrivals: most requests are a Poisson process (exponential gaps); `burst_frac`
            of them land together inside a short window mid-way, so the batch
            fills and queues form
* priority: 20% of requests are priority 0 (interactive), the rest priority 1;
            lower number = more important

`long_prompt_scenario` is the targeted experiment: steady decoders plus a few
very long prompts arriving while they are mid-generation.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from request import Request


BURST_POSITION_FRAC = 0.4  # the burst lands 40% of the way through the Poisson arrivals


@dataclass(frozen=True)
class WorkloadConfig:
    num_requests: int = 100
    seed: int = 0
    short_prompt: tuple[int, int] = (32, 256)
    long_prompt: tuple[int, int] = (1024, 2048)
    long_prompt_frac: float = 0.3
    short_output: tuple[int, int] = (16, 64)
    long_output: tuple[int, int] = (200, 512)
    long_output_frac: float = 0.4
    mean_gap_ms: float = 60.0  # Poisson part
    burst_frac: float = 0.3
    burst_window_ms: float = 300.0
    high_priority_frac: float = 0.2

    @property
    def num_burst(self) -> int:
        return round(self.num_requests * self.burst_frac)

    def burst_start_ms(self) -> float:
        return BURST_POSITION_FRAC * (self.num_requests - self.num_burst) * self.mean_gap_ms


def generate_workload(cfg: WorkloadConfig) -> list[Request]:
    rng = random.Random(cfg.seed)
    arrivals: list[float] = []
    clock = 0.0
    for _ in range(cfg.num_requests - cfg.num_burst):
        clock += rng.expovariate(1.0 / cfg.mean_gap_ms)
        arrivals.append(clock)
    start = cfg.burst_start_ms()
    arrivals += [start + rng.uniform(0.0, cfg.burst_window_ms) for _ in range(cfg.num_burst)]
    arrivals.sort()

    requests = []
    for i, arrival in enumerate(arrivals):
        p_range = cfg.long_prompt if rng.random() < cfg.long_prompt_frac else cfg.short_prompt
        o_range = cfg.long_output if rng.random() < cfg.long_output_frac else cfg.short_output
        prio = 0 if rng.random() < cfg.high_priority_frac else 1
        requests.append(Request(i, arrival, rng.randint(*p_range), rng.randint(*o_range), prio))
    return requests


def clone_requests(requests: list[Request]) -> list[Request]:
    """Fresh, un-run copies (the simulator mutates requests)."""
    return [
        Request(r.request_id, r.arrival_time, r.prompt_tokens, r.max_output_tokens, r.priority)
        for r in requests
    ]


def long_prompt_scenario(
    num_decoders: int = 16,
    decoder_prompt: int = 64,
    decoder_output: int = 400,
    long_prompt_tokens: int = 2048,
    long_output: int = 16,
    long_arrivals_ms: tuple[float, ...] = (400.0, 1200.0),
) -> list[Request]:
    """Steady decoders (ids 0..n-1) plus long prompts that arrive mid-decode."""
    requests = [Request(i, i * 2.0, decoder_prompt, decoder_output) for i in range(num_decoders)]
    for j, t in enumerate(long_arrivals_ms):
        requests.append(Request(num_decoders + j, t, long_prompt_tokens, long_output))
    return requests
