"""Measurement plumbing shared by the experiments: interleaved repeats and request-level metrics.

Workload model: R requests all arrive at t=0 and are served by sequential static batches.
A request's TTFT is therefore its batch's prefill *plus the time spent waiting for earlier
batches* (queueing), which is what makes max concurrency show up in P95 TTFT.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence, TypeVar

import numpy as np

from variants import DecodeRun

T = TypeVar("T")
BYTES_PER_MB = 1024 * 1024
DRIFT_WARN_FRACTION = 0.10
MIN_REPEATS_FOR_DRIFT = 4


@dataclass(frozen=True)
class RequestMetrics:
    wall_s: float
    ttfts_s: tuple[float, ...]  # one per request
    tpots_s: tuple[float, ...]  # one per request
    output_tokens: int
    peak_mem_bytes: int


def interleaved(
    cases: Mapping[str, Callable[[], T]],
    warmup: int,
    repeats: int,
    cooldown_s: float = 0.0,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, list[T]]:
    """Warm each case up, then run timed rounds, rotating the start of each round (ABC, BCA, CAB...).

    Interleaving makes slow drift hit every case; rotating the order removes *position* bias.
    A case that always runs first in a round pays for any idle->active ramp (clocks, power
    state) after the previous round, which showed up as a fake TTFT penalty in a first
    version of this harness that ran day1 first every time.
    """
    if warmup < 0:
        raise ValueError("warmup must be >= 0")
    if repeats < 1:
        raise ValueError("repeats must be >= 1")
    for fn in cases.values():
        for _ in range(warmup):
            fn()
    collected: dict[str, list[T]] = {label: [] for label in cases}
    labels = list(cases)
    for round_index in range(repeats):
        shift = round_index % len(labels)
        for label in labels[shift:] + labels[:shift]:
            collected[label].append(cases[label]())
        if cooldown_s > 0 and round_index < repeats - 1:
            sleep(cooldown_s)
    return collected


def balanced_repeats(repeats: int, n_cases: int) -> int:
    """Round `repeats` up to a multiple of `n_cases` so every case holds every round position equally."""
    if repeats < 1 or n_cases < 1:
        raise ValueError("repeats and n_cases must be >= 1")
    return -(-repeats // n_cases) * n_cases


def request_metrics(batch_runs: Sequence[DecodeRun]) -> RequestMetrics:
    if not batch_runs:
        raise ValueError("need at least one batch run")
    clock = 0.0
    ttfts: list[float] = []
    tpots: list[float] = []
    for batch in batch_runs:
        ttfts.extend([clock + batch.prefill_s] * batch.batch_size)
        tpots.extend([batch.tpot_s] * batch.batch_size)
        clock += batch.total_s
    return RequestMetrics(
        wall_s=clock,
        ttfts_s=tuple(ttfts),
        tpots_s=tuple(tpots),
        output_tokens=sum(b.output_tokens for b in batch_runs),
        peak_mem_bytes=max(b.peak_mem_bytes for b in batch_runs),
    )


def percentile(values: Sequence[float], q: float) -> float:
    if not len(values):
        raise ValueError("cannot take a percentile of an empty sequence")
    return float(np.percentile(np.asarray(values, dtype=float), q))


def match_fraction(tokens: Sequence[Sequence[int]], reference: Sequence[Sequence[int]]) -> float:
    if len(tokens) != len(reference):
        raise ValueError("token rows and reference rows differ in number")
    return sum(1 for a, b in zip(tokens, reference) if tuple(a) == tuple(b)) / len(reference)


def drift_fraction(values: Sequence[float]) -> float:
    """|median(second half) - median(first half)| / median(all) over repeats of one case."""
    if len(values) < MIN_REPEATS_FOR_DRIFT:
        return 0.0
    half = len(values) // 2
    first, second = statistics.median(values[:half]), statistics.median(values[-half:])
    return abs(second - first) / statistics.median(values)


def flatten_tokens(batch_runs: Sequence[DecodeRun]) -> tuple[tuple[int, ...], ...]:
    return tuple(row for batch in batch_runs for row in batch.tokens)


def summarize_case(label: str, reps: Sequence[Sequence[DecodeRun]], n_requests: int) -> dict:
    """One result row from the repeats of one case (each repeat = the batch runs serving R requests)."""
    per_rep = [request_metrics(rep) for rep in reps]
    rps = [n_requests / m.wall_s for m in per_rep]
    tok_s = [m.output_tokens / m.wall_s for m in per_rep]
    ttfts = [t * 1e3 for m in per_rep for t in m.ttfts_s]
    tpots = [t * 1e3 for m in per_rep for t in m.tpots_s]
    walls = [m.wall_s for m in per_rep]
    return {
        "label": label,
        "n_reps": len(per_rep),
        "n_requests": n_requests,
        "requests_per_s": statistics.median(rps),
        "requests_per_s_min": min(rps),
        "requests_per_s_max": max(rps),
        "output_tok_s": statistics.median(tok_s),
        "ttft_ms_p50": percentile(ttfts, 50),
        "ttft_ms_p95": percentile(ttfts, 95),
        "tpot_ms_p50": percentile(tpots, 50),
        "tpot_ms_p95": percentile(tpots, 95),
        "wall_s": statistics.median(walls),
        "wall_spread_fraction": (max(walls) - min(walls)) / statistics.median(walls),
        "peak_mem_mb": max(m.peak_mem_bytes for m in per_rep) / BYTES_PER_MB,
        "wall_drift_fraction": drift_fraction(walls),
    }


DELTA_KEYS = ("requests_per_s", "output_tok_s", "ttft_ms_p95", "tpot_ms_p95", "peak_mem_mb", "wall_s")


def with_baseline_deltas(rows: Sequence[dict], baseline_label: str) -> list[dict]:
    """Return new rows with `<metric>_vs_baseline_pct` and whether the throughput ranges overlap.

    `rps_range_overlaps_baseline` is the honest noise check: if the min..max requests/s of a
    row overlaps the baseline's min..max across repeats, the difference is not clearly above
    run-to-run variation and must not be reported as a speedup.
    """
    base = next((r for r in rows if r["label"] == baseline_label), None)
    if base is None:
        raise ValueError(f"baseline {baseline_label!r} not among rows")
    result = []
    for row in rows:
        extra = {f"{k}_vs_baseline_pct": 100.0 * (row[k] / base[k] - 1.0) for k in DELTA_KEYS}
        overlaps = (
            row["requests_per_s_min"] <= base["requests_per_s_max"]
            and base["requests_per_s_min"] <= row["requests_per_s_max"]
        )
        result.append(
            {**row, **extra, "is_baseline": row["label"] == baseline_label,
             "rps_range_overlaps_baseline": overlaps}
        )
    return result
