from __future__ import annotations

import pytest
import torch

from bench_workers import (
    COLLECTIVES,
    bench_collectives_worker,
    bench_tp_mlp_worker,
    time_op,
)
from dist_utils import spawn_group


def test_time_op_runs_warmup_plus_timed_iterations_and_returns_only_timed() -> None:
    calls: list[int] = []

    samples = time_op(lambda: calls.append(1), iters=4, warmup=2)

    assert len(calls) == 6
    assert len(samples) == 4
    assert all(s >= 0 for s in samples)


def test_time_op_inactive_rank_skips_the_op_but_still_returns_no_samples() -> None:
    calls: list[int] = []

    samples = time_op(lambda: calls.append(1), iters=3, warmup=1, active=False)

    assert calls == [] and samples == []


def test_time_op_rejects_nonpositive_iters() -> None:
    with pytest.raises(ValueError, match="iters"):
        time_op(lambda: None, iters=0, warmup=0)


def test_collective_benchmark_reports_every_collective_and_size_per_rank() -> None:
    sizes = (1024, 8192)

    per_rank = spawn_group(bench_collectives_worker, 2, sizes, 3, 1)

    assert len(per_rank) == 2
    for rows in per_rank:
        assert [(r["collective"], r["size_bytes"]) for r in rows] == [
            (c, s) for c in COLLECTIVES for s in sizes
        ]
        assert all(0 < r["min_s"] <= r["median_s"] for r in rows)


def test_collective_benchmark_restores_thread_setting_is_per_process() -> None:
    before = torch.get_num_threads()

    spawn_group(bench_collectives_worker, 2, (1024,), 2, 0)

    assert torch.get_num_threads() == before  # the parent is untouched


def test_tp_mlp_benchmark_is_numerically_correct_and_shards_weights() -> None:
    cases = ((32, 4), (32, 16))

    per_rank = spawn_group(bench_tp_mlp_worker, 2, cases, 4, 3, 1, 0)

    for rank, rows in enumerate(per_rank):
        assert [(r["d_model"], r["tokens"]) for r in rows] == list(cases)
        for r in rows:
            assert r["max_err"] < 1e-4
            assert r["tp_total_s"] > 0 and r["tp_comm_s"] > 0
            assert r["shard_weight_bytes"] * 2 == r["full_weight_bytes"]
            # the single-process baseline is timed on rank 0 only, with the others idle
            assert (r["ref_s"] > 0) == (rank == 0)


def test_prepared_collective_ops_are_repeatable_on_preallocated_buffers() -> None:
    from helpers import worker_prepared_ops

    results = spawn_group(worker_prepared_ops, 4, 4096, 40)

    assert all(r == {"all_reduce": True, "all_gather": True, "broadcast": True} for r in results)


def test_prepare_collective_op_rejects_unknown_collective() -> None:
    from bench_workers import prepare_collective_op

    with pytest.raises(ValueError, match="collective"):
        prepare_collective_op("reduce_scatter", 1024, ctx=None)
