from __future__ import annotations

import pytest

import helpers
from collectives import all_gather_cat, all_reduce_sum, broadcast_from, synchronize
from dist_utils import spawn_group

WORLD = 2


def test_all_reduce_sums_across_ranks_and_leaves_input_untouched() -> None:
    results = spawn_group(helpers.worker_all_reduce, WORLD)

    for rank, (local, reduced) in enumerate(results):
        assert local == [float(rank + 1)] * 3  # 1 and 2, unmodified
        assert reduced == [3.0] * 3  # 1 + 2 on every rank


def test_all_reduce_with_three_ranks() -> None:
    results = spawn_group(helpers.worker_all_reduce, 3)

    assert all(reduced == [6.0] * 3 for _, reduced in results)  # 1 + 2 + 3


@pytest.mark.parametrize("src", [0, 1])
def test_broadcast_copies_source_value_to_every_rank(src: int) -> None:
    results = spawn_group(helpers.worker_broadcast, WORLD, src)

    for rank, (local, out) in enumerate(results):
        assert out == [float(src * 10)] * 2
        assert local == [float(rank * 10)] * 2  # caller's tensor not mutated


def test_all_gather_concatenates_in_rank_order_along_dim0() -> None:
    results = spawn_group(helpers.worker_all_gather, WORLD, 0)

    expected = [[0.0, 0.0], [0.0, 0.0], [1.0, 1.0], [1.0, 1.0]]
    assert results == [expected, expected]


def test_all_gather_concatenates_along_last_dim() -> None:
    results = spawn_group(helpers.worker_all_gather, WORLD, 1)

    expected = [[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0]]
    assert results == [expected, expected]


def test_synchronize_blocks_fast_rank_until_slow_rank_arrives() -> None:
    delay = 0.6

    (start0, end0), (start1, end1) = spawn_group(helpers.worker_synchronize, WORLD, delay)

    assert end1 - start1 >= delay * 0.9  # rank 1 waited for rank 0's sleep
    assert end0 >= start0 + delay


def test_demo_runs_every_collective_and_reports_pass() -> None:
    reports = spawn_group(helpers.worker_demo, WORLD)

    for rank, lines in enumerate(reports):
        text = "\n".join(lines)
        for name in ("all_reduce", "broadcast", "all_gather", "barrier"):
            assert f"[rank {rank}] {name}" in text
        assert "FAIL" not in text


def test_single_process_fallback_when_group_not_initialised() -> None:
    import torch

    t = torch.arange(4.0)
    assert all_reduce_sum(t).tolist() == t.tolist()
    assert broadcast_from(t, src=0).tolist() == t.tolist()
    assert all_gather_cat(t).tolist() == t.tolist()
    synchronize()  # no-op, must not raise
