"""Discrete-time serving simulation used by Experiment C.

Requests with varied prompt/output lengths arrive over time and compete for one fixed KV
pool. Each step: admit waiting requests (FIFO, head-of-line blocking), then decode one
token for every running request. The same seeded workload is replayed against any
allocator that implements `paged_allocator.KVAllocator`, so differences in the results
come from the allocation policy alone. Pure Python/NumPy: no GPU involved.

If a running request cannot get a new block mid-decode it is *preempted*: its blocks are
freed and it goes back to the front of the queue to restart (recompute-style preemption).
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import asdict, dataclass

import numpy as np

from paged_allocator import AllocatorStats, InsufficientBlocksError, KVAllocator

DEFAULT_MAX_STEPS = 200_000


@dataclass(frozen=True)
class WorkloadConfig:
    num_requests: int
    max_seq_len: int
    arrivals_per_step: float
    prompt_median: int
    prompt_sigma: float
    output_median: int
    output_sigma: float
    seed: int = 0


@dataclass(frozen=True)
class Request:
    req_id: str
    arrival_step: int
    prompt_len: int
    output_len: int

    @property
    def total_len(self) -> int:
        return self.prompt_len + self.output_len


@dataclass(frozen=True)
class SimResult:
    allocator: str
    block_size: int | None
    steps: int
    completed: int
    unfinished: int
    preemptions: int
    admission_failures: int
    mean_running: float
    peak_running: int
    mean_queue_wait_steps: float  # arrival -> first admission (ignores re-queues after preemption)
    mean_latency_steps: float  # arrival -> completion, includes every re-queue
    mean_utilization: float
    peak_utilization: float
    mean_allocated_fraction: float
    mean_unused_reserved_slots: float
    unused_reserved_fraction: float
    throughput_tokens_per_step: float

    def to_row(self) -> dict[str, object]:
        return asdict(self)


def _lognormal_int(rng: np.random.Generator, median: int, sigma: float) -> int:
    return max(1, int(rng.lognormal(math.log(median), sigma)))


def generate_requests(cfg: WorkloadConfig) -> tuple[Request, ...]:
    """Deterministic workload: Poisson-ish arrivals, lognormal prompt/output lengths.

    Prompt + output never exceeds `max_seq_len`, so every request is satisfiable by an
    allocator that reserves `max_seq_len` per request.
    """
    if cfg.num_requests <= 0 or cfg.max_seq_len < 2 or cfg.arrivals_per_step <= 0:
        raise ValueError("num_requests>0, max_seq_len>=2 and arrivals_per_step>0 are required")
    rng = np.random.default_rng(cfg.seed)
    gaps = rng.exponential(1.0 / cfg.arrivals_per_step, size=cfg.num_requests)
    arrivals = np.floor(np.cumsum(gaps)).astype(int)
    requests = []
    for i, step in enumerate(arrivals):
        prompt = min(_lognormal_int(rng, cfg.prompt_median, cfg.prompt_sigma), cfg.max_seq_len - 1)
        output = min(_lognormal_int(rng, cfg.output_median, cfg.output_sigma), cfg.max_seq_len - prompt)
        requests.append(Request(f"req{i}", int(step), prompt, output))
    return tuple(requests)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def simulate(
    name: str,
    allocator: KVAllocator,
    requests: tuple[Request, ...],
    block_size: int | None = None,
    max_steps: int = DEFAULT_MAX_STEPS,
) -> SimResult:
    """Replay `requests` against `allocator` and summarise memory behaviour."""
    if max_steps <= 0:
        raise ValueError(f"max_steps must be positive, got {max_steps}")
    capacity = allocator.stats().capacity_slots
    too_big = [r.req_id for r in requests if r.total_len > capacity]
    if too_big:
        raise ValueError(
            f"{len(too_big)} request(s) (e.g. {too_big[0]}) need more than the pool's {capacity} slots "
            "and could never finish"
        )
    by_id = {r.req_id: r for r in requests}
    pending = deque(sorted(requests, key=lambda r: r.arrival_step))
    waiting: deque[Request] = deque()
    generated: dict[str, int] = {}
    first_admit: dict[str, int] = {}
    finished_at: dict[str, int] = {}
    samples: list[tuple[AllocatorStats, bool, int]] = []  # (stats, backlog, running)
    completed = preemptions = useful_tokens = 0
    step = 0

    for step in range(max_steps):
        while pending and pending[0].arrival_step <= step:
            waiting.append(pending.popleft())

        while waiting:  # FIFO admission with head-of-line blocking
            req = waiting[0]
            try:
                allocator.allocate(req.req_id, req.prompt_len)
            except InsufficientBlocksError:
                break
            waiting.popleft()
            generated[req.req_id] = 0
            first_admit.setdefault(req.req_id, step)

        for rid in list(generated):  # oldest admission first
            if rid not in generated:  # preempted earlier this step
                continue
            req = by_id[rid]
            while True:
                try:
                    allocator.extend(rid)
                    break
                except InsufficientBlocksError:
                    # Evict the youngest running request (possibly rid itself) so the
                    # oldest always makes progress; evicting the failing one instead can
                    # livelock when the pool is smaller than the combined demand.
                    victim = next(reversed(generated))
                    allocator.free(victim)
                    del generated[victim]
                    waiting.appendleft(by_id[victim])
                    preemptions += 1
                    if victim == rid:
                        break
            if rid not in generated:
                continue
            generated[rid] += 1
            if generated[rid] == req.output_len:
                allocator.free(rid)
                del generated[rid]
                completed += 1
                finished_at[rid] = step + 1
                useful_tokens += req.output_len

        samples.append((allocator.stats(), bool(waiting), len(generated)))
        if not pending and not waiting and not generated:
            break

    final = allocator.stats()
    steps = step + 1
    loaded = [s for s in samples if s[1]] or samples  # steps with a non-empty wait queue: pool is contended
    loaded_stats = [s[0] for s in loaded]
    alloc_total = sum(s.allocated_slots for s in loaded_stats)
    frag_total = sum(s.internal_fragmentation for s in loaded_stats)
    waits = [first_admit[r.req_id] - r.arrival_step for r in requests if r.req_id in first_admit]
    latencies = [finished_at[r.req_id] - r.arrival_step for r in requests if r.req_id in finished_at]

    return SimResult(
        allocator=name,
        block_size=block_size,
        steps=steps,
        completed=completed,
        unfinished=len(requests) - completed,
        preemptions=preemptions,
        admission_failures=final.failures - preemptions,
        mean_running=_mean([s[2] for s in samples]),
        peak_running=max(s[2] for s in samples),
        mean_queue_wait_steps=_mean(waits),
        mean_latency_steps=_mean(latencies),
        mean_utilization=_mean([s.utilization for s in loaded_stats]),
        peak_utilization=max(s[0].utilization for s in samples),
        mean_allocated_fraction=_mean([s.allocated_fraction for s in loaded_stats]),
        mean_unused_reserved_slots=_mean([s.internal_fragmentation for s in loaded_stats]),
        unused_reserved_fraction=frag_total / alloc_total if alloc_total else 0.0,
        throughput_tokens_per_step=useful_tokens / steps,
    )
