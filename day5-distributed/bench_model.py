"""Pure math for the Day 5 communication benchmark (no torch, no processes).

* ``fit_alpha_beta``   -- latency/bandwidth model  t(bytes) = alpha + beta * bytes
* ``bus_bandwidth_factor`` -- nccl-tests definition of "bus bandwidth" per collective
* ``merge_ranks``      -- a collective is only as fast as its slowest rank
* ``predict_tp_speedup`` -- roofline-style model of one tensor-parallel MLP
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

MIN_ITERS = 3
ITER_BUDGET_BYTES = 512 * 1024 * 1024  # cap total traffic per (collective, size) case


def fit_alpha_beta(sizes_bytes: Sequence[float], times_s: Sequence[float]) -> tuple[float, float]:
    """Least-squares line through (bytes, seconds): returns (alpha seconds, beta seconds/byte)."""
    if len(sizes_bytes) != len(times_s):
        raise ValueError("sizes and times must have the same length")
    if len(sizes_bytes) < 2:
        raise ValueError("need at least two measurements to fit alpha/beta")
    if len(set(sizes_bytes)) < 2:
        raise ValueError("need at least two distinct sizes to fit alpha/beta")
    times = np.asarray(times_s, float)
    # Weight each point by 1/t so the fit minimises *relative* error. Unweighted least squares
    # is dominated by the largest messages and can return a negative (meaningless) latency.
    beta, alpha = np.polyfit(np.asarray(sizes_bytes, float), times, 1, w=1.0 / times)
    return float(alpha), float(beta)


def fit_max_rel_error(
    sizes_bytes: Sequence[float], times_s: Sequence[float], alpha: float, beta: float
) -> float:
    """Worst |model - measured| / measured over the points: how far to trust alpha/beta."""
    return max(abs(alpha + beta * s - t) / t for s, t in zip(sizes_bytes, times_s))


def bus_bandwidth_factor(collective: str, world: int) -> float:
    """Multiply algorithmic bandwidth by this to get 'bus bandwidth' (nccl-tests convention).

    all_reduce: 2(n-1)/n with bytes = message size.
    all_gather: (n-1)/n with bytes = total gathered size (n x per-rank shard).
    broadcast : 1.
    """
    if collective == "all_reduce":
        return 2 * (world - 1) / world
    if collective == "all_gather":
        return (world - 1) / world
    if collective == "broadcast":
        return 1.0
    raise ValueError(f"unknown collective {collective!r}")


def iters_for(size_bytes: int, base_iters: int) -> int:
    """Fewer timed iterations for very large messages, never above ``base_iters``."""
    return min(base_iters, max(MIN_ITERS, ITER_BUDGET_BYTES // max(size_bytes, 1)))


def merge_ranks(
    per_rank: Sequence[Sequence[dict]], keys: tuple[str, ...], max_fields: tuple[str, ...]
) -> list[dict]:
    """Combine each rank's rows into one row per case; ``max_fields`` take the slowest rank.

    Other fields are taken from rank 0. Inputs are not modified.
    """
    def case_of(row: dict) -> tuple:
        return tuple(row[k] for k in keys)

    reference_cases = [case_of(r) for r in per_rank[0]]
    for rows in per_rank[1:]:
        if [case_of(r) for r in rows] != reference_cases:
            raise ValueError("ranks reported different cases; they must run the same cases")

    merged = []
    for i, base in enumerate(per_rank[0]):
        row = dict(base)
        for field in max_fields:
            row[field] = max(rows[i][field] for rows in per_rank)
        merged.append(row)
    return merged


def median_across_runs(
    runs: Sequence[Sequence[dict]],
    keys: tuple[str, ...],
    fields: tuple[str, ...],
    max_fields: tuple[str, ...] = (),
) -> list[dict]:
    """Collapse repeated runs of the same cases: median of ``fields`` plus a relative spread
    ``<field>_spread`` = (max - min) / median; ``max_fields`` keep the worst value seen.
    """
    if not runs:
        raise ValueError("need at least one run")

    def case_of(row: dict) -> tuple:
        return tuple(row[k] for k in keys)

    reference_cases = [case_of(r) for r in runs[0]]
    if any([case_of(r) for r in run] != reference_cases for run in runs[1:]):
        raise ValueError("runs must cover the same cases in the same order")

    out = []
    for i, base in enumerate(runs[0]):
        row = dict(base)
        for field in fields:
            values = [run[i][field] for run in runs]
            median = float(np.median(values))
            row[field] = median
            row[f"{field}_spread"] = (max(values) - min(values)) / median if median else 0.0
        for field in max_fields:
            row[field] = max(run[i][field] for run in runs)
        out.append(row)
    return out


def mlp_cost(
    tokens: int, d_model: int, d_ff: int, n_matrices: int = 2, bytes_per_elem: int = 4
) -> tuple[float, float, float]:
    """(FLOPs, weight bytes, AllReduce message bytes) of one MLP over ``tokens`` tokens."""
    flops = 2.0 * tokens * d_model * d_ff * n_matrices
    weight_bytes = float(d_model * d_ff * n_matrices * bytes_per_elem)
    comm_bytes = float(tokens * d_model * bytes_per_elem)
    return flops, weight_bytes, comm_bytes


@dataclass(frozen=True)
class TPPrediction:
    t_single: float
    t_tp_compute: float
    t_comm: float
    t_tp: float
    speedup: float
    comm_fraction: float


def predict_tp_speedup(
    *,
    flops: float,
    weight_bytes: float,
    comm_bytes: float,
    device_flops: float,
    device_bw: float,
    comm_alpha_s: float,
    comm_beta_s_per_byte: float,
    world: int,
) -> TPPrediction:
    """Roofline estimate: compute = max(flops/F, weight_bytes/BW); TP divides both by ``world``
    and adds one AllReduce, alpha + beta * comm_bytes. ``device_bw=inf`` ignores the memory term.
    """
    if world < 1:
        raise ValueError(f"world must be >= 1, got {world}")
    if device_flops <= 0:
        raise ValueError("device_flops must be positive")
    memory_time = 0.0 if math.isinf(device_bw) else weight_bytes / device_bw
    t_single = max(flops / device_flops, memory_time)
    t_tp_compute = t_single / world
    t_comm = 0.0 if world == 1 else comm_alpha_s + comm_beta_s_per_byte * comm_bytes
    t_tp = t_tp_compute + t_comm
    return TPPrediction(
        t_single=t_single,
        t_tp_compute=t_tp_compute,
        t_comm=t_comm,
        t_tp=t_tp,
        speedup=t_single / t_tp,
        comm_fraction=t_comm / t_tp,
    )
