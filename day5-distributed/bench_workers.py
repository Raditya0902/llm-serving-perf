"""Code that runs *inside* each rank of a benchmark group (launched via ``spawn_group``).

Every worker pins itself to one intra-op thread so N ranks on one CPU do not fight over
cores, and so "1 process" vs "N processes" compares like with like. Workers return plain
lists of dicts; ``bench_model.merge_ranks`` then takes the slowest rank per case.
"""
from __future__ import annotations

import statistics
import time
from typing import Callable

import torch
import torch.distributed as dist
import torch.nn.functional as F

from bench_model import iters_for
from collectives import all_reduce_sum, synchronize
from dist_utils import DistContext
from tensor_parallel import ParallelMLP

COLLECTIVES = ("all_reduce", "all_gather", "broadcast")
FLOAT32_BYTES = 4


def time_op(
    op: Callable[[], object], iters: int, warmup: int, active: bool = True
) -> list[float]:
    """Wall-clock seconds for each of ``iters`` calls, after ``warmup`` untimed calls.

    Ranks barrier before every timed call so they start together; Gloo CPU collectives are
    synchronous, so returning from ``op`` means this rank's part is done. With
    ``active=False`` the rank skips ``op`` but still joins every barrier and returns ``[]``:
    this lets one rank time a compute-only baseline while the others stay idle, so the
    baseline is not slowed by its neighbours competing for memory bandwidth.
    """
    if iters < 1:
        raise ValueError(f"iters must be >= 1, got {iters}")
    for _ in range(warmup):
        if active:
            op()
    synchronize()
    samples = []
    for _ in range(iters):
        synchronize()
        if not active:
            continue
        start = time.perf_counter()
        op()
        samples.append(time.perf_counter() - start)
    return samples


def _randn(shape: tuple[int, ...], seed: int) -> torch.Tensor:
    return torch.randn(*shape, generator=torch.Generator().manual_seed(seed))


def prepare_collective_op(name: str, size_bytes: int, ctx: DistContext | None) -> Callable[[], object]:
    """A zero-argument callable running the *bare* collective on preallocated buffers.

    The library wrappers (collectives.py) clone / allocate / concatenate on every call; at
    16 MB that copy made AllReduce measure ~2.5x slower than the collective itself (checked),
    so the benchmark calls ``torch.distributed`` directly. Buffers are zeros so repeated
    in-place AllReduce cannot overflow; Gloo's speed does not depend on the values.
    """
    if name not in COLLECTIVES:
        raise ValueError(f"unknown collective {name!r}; expected one of {COLLECTIVES}")
    device = ctx.device if ctx is not None else torch.device("cpu")
    buf = torch.zeros(max(size_bytes // FLOAT32_BYTES, 1), device=device)
    if name == "all_reduce":
        return lambda: dist.all_reduce(buf)
    if name == "broadcast":
        return lambda: dist.broadcast(buf, src=0)
    pieces = [torch.empty_like(buf) for _ in range(ctx.world_size)]
    return lambda: dist.all_gather(pieces, buf)


def bench_collectives_worker(
    ctx: DistContext, sizes_bytes: tuple[int, ...], base_iters: int, warmup: int
) -> list[dict]:
    """Time each bare collective at each per-rank message size (float32 payload)."""
    torch.set_num_threads(1)
    rows = []
    for name in COLLECTIVES:
        for size in sizes_bytes:
            op = prepare_collective_op(name, size, ctx)
            samples = time_op(op, iters_for(size, base_iters), warmup)
            rows.append(
                {
                    "collective": name,
                    "size_bytes": size,
                    "median_s": statistics.median(samples),
                    "min_s": min(samples),
                }
            )
    return rows


def bench_tp_mlp_worker(
    ctx: DistContext,
    cases: tuple[tuple[int, int], ...],
    ff_mult: int,
    iters: int,
    warmup: int,
    seed: int,
) -> list[dict]:
    """For each (d_model, tokens): time the unsharded MLP, the TP MLP, and its two halves."""
    torch.set_num_threads(1)
    rows = []
    for d_model, tokens in cases:
        d_ff = ff_mult * d_model
        x = _randn((tokens, d_model), seed)
        w1, b1 = _randn((d_model, d_ff), seed + 1) * 0.05, _randn((d_ff,), seed + 2)
        w2, b2 = _randn((d_ff, d_model), seed + 3) * 0.05, _randn((d_model,), seed + 4)

        def reference() -> torch.Tensor:
            return F.gelu(x @ w1 + b1) @ w2 + b2

        mlp = ParallelMLP(w1, b1, w2, b2, ctx.rank, ctx.world_size)
        partial_sum = torch.zeros(tokens, d_model)

        def compute_only() -> torch.Tensor:  # the local shard's work, no communication
            return F.gelu(mlp.up(x)) @ mlp.down.weight

        # Baseline = one lone process (what TP=1 would run); other ranks idle meanwhile.
        ref_samples = time_op(reference, iters, warmup, active=ctx.rank == 0)
        total_samples = time_op(lambda: mlp(x), iters, warmup)
        compute_samples = time_op(compute_only, iters, warmup)
        comm_samples = time_op(lambda: all_reduce_sum(partial_sum), iters, warmup)

        full_bytes = (w1.numel() + w2.numel()) * FLOAT32_BYTES
        shard_bytes = (mlp.up.weight.numel() + mlp.down.weight.numel()) * FLOAT32_BYTES
        rows.append(
            {
                "d_model": d_model,
                "tokens": tokens,
                "ref_s": statistics.median(ref_samples) if ref_samples else 0.0,
                "tp_total_s": statistics.median(total_samples),
                "tp_compute_s": statistics.median(compute_samples),
                "tp_comm_s": statistics.median(comm_samples),
                "max_err": float((mlp(x) - reference()).abs().max()),
                "shard_weight_bytes": shard_bytes,
                "full_weight_bytes": full_bytes,
            }
        )
    return rows
