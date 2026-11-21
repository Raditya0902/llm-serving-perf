from __future__ import annotations

import pytest
import torch

import helpers
from dist_utils import spawn_group
from tensor_parallel import (
    ColumnParallelLinear,
    ParallelMLP,
    RowParallelLinear,
    split_columns,
    split_last_dim,
    split_rows,
)

TOL = 1e-5


# --- pure shard algebra (no processes) ---------------------------------------------------
def test_split_columns_then_concat_restores_weight() -> None:
    w = helpers.seeded_tensor((6, 8), 0)

    shards = split_columns(w, 4)

    assert [s.shape for s in shards] == [(6, 2)] * 4
    assert torch.equal(torch.cat(shards, dim=1), w)


def test_split_rows_then_concat_restores_weight() -> None:
    w = helpers.seeded_tensor((8, 6), 0)

    shards = split_rows(w, 2)

    assert [s.shape for s in shards] == [(4, 6)] * 2
    assert torch.equal(torch.cat(shards, dim=0), w)


def test_shards_are_independent_copies() -> None:
    w = helpers.seeded_tensor((4, 4), 0)
    before = w.clone()

    shards = split_columns(w, 2)
    shards[0].zero_()

    assert torch.equal(w, before)


@pytest.mark.parametrize("splitter", [split_columns, split_rows, split_last_dim])
def test_split_rejects_indivisible_size(splitter) -> None:
    with pytest.raises(ValueError, match="divisible"):
        splitter(torch.zeros(5, 5), 2)


@pytest.mark.parametrize("splitter", [split_columns, split_rows, split_last_dim])
def test_split_rejects_nonpositive_parts(splitter) -> None:
    with pytest.raises(ValueError, match="parts"):
        splitter(torch.zeros(4, 4), 0)


@pytest.mark.parametrize("n", [1, 2, 4])
def test_column_partition_math_concatenation_reproduces_xw(n: int) -> None:
    """Y = X W: each device computes X @ W_i; concatenating columns gives Y."""
    x, w = helpers.seeded_tensor((5, 8), 1), helpers.seeded_tensor((8, 12), 2)

    partials = [x @ w_i for w_i in split_columns(w, n)]

    assert torch.allclose(torch.cat(partials, dim=1), x @ w, atol=TOL)


@pytest.mark.parametrize("n", [1, 2, 4])
def test_row_partition_math_summing_partials_reproduces_xw(n: int) -> None:
    """Row split needs X split on its last dim; partial products must be SUMMED."""
    x, w = helpers.seeded_tensor((5, 8), 1), helpers.seeded_tensor((8, 12), 2)

    partials = [x_i @ w_i for x_i, w_i in zip(split_last_dim(x, n), split_rows(w, n))]

    assert torch.allclose(sum(partials), x @ w, atol=TOL)


def test_row_partition_concatenating_partials_is_wrong() -> None:
    """Guards the key distinction: row-parallel partials are not column slices."""
    x, w = helpers.seeded_tensor((5, 8), 1), helpers.seeded_tensor((8, 12), 2)

    partials = [x_i @ w_i for x_i, w_i in zip(split_last_dim(x, 2), split_rows(w, 2))]

    assert torch.cat(partials, dim=1).shape != (x @ w).shape


# --- layers in a single process (world_size=1 degenerates to a plain linear) -------------
def test_layers_with_world_size_one_match_plain_linear() -> None:
    x, w, b = helpers._tp_inputs(3, 4, 6, 0)

    col = ColumnParallelLinear(w, b, rank=0, world_size=1)
    row = RowParallelLinear(w, b, rank=0, world_size=1, input_is_parallel=False)

    assert torch.allclose(col(x), x @ w + b, atol=TOL)
    assert torch.allclose(row(x), x @ w + b, atol=TOL)


def test_layer_rejects_rank_outside_world() -> None:
    w = torch.zeros(4, 4)
    with pytest.raises(ValueError, match="rank"):
        ColumnParallelLinear(w, None, rank=2, world_size=2)


def test_layer_rejects_indivisible_weight() -> None:
    with pytest.raises(ValueError, match="divisible"):
        RowParallelLinear(torch.zeros(5, 4), None, rank=0, world_size=2)


def test_layer_does_not_alias_caller_weight() -> None:
    w = torch.ones(4, 4)
    layer = ColumnParallelLinear(w, None, rank=0, world_size=2)

    layer.weight.zero_()

    assert torch.equal(w, torch.ones(4, 4))


# --- layers across real processes --------------------------------------------------------
@pytest.mark.parametrize("world", [2, 4])
@pytest.mark.parametrize("with_bias", [False, True])
def test_column_parallel_with_allgather_matches_reference(world: int, with_bias: bool) -> None:
    results = spawn_group(helpers.worker_column, world, True, with_bias)

    for r in results:
        assert r["max_err"] < TOL
        assert r["out_shape"] == (5, 12)  # full output on every rank
        assert r["weight_numel"] == 8 * 12 // world
        assert r["collective"] == "all_gather"


def test_column_parallel_without_gather_keeps_output_sharded_and_silent() -> None:
    results = spawn_group(helpers.worker_column, 2, False, True)

    for r in results:
        assert r["max_err"] < TOL
        assert r["out_shape"] == (5, 6)
        assert r["collective"] == "none"


@pytest.mark.parametrize("world", [2, 4])
@pytest.mark.parametrize("parallel_input", [False, True])
def test_row_parallel_allreduce_matches_reference(world: int, parallel_input: bool) -> None:
    results = spawn_group(helpers.worker_row, world, parallel_input, True)

    for r in results:
        assert r["max_err"] < TOL  # bias added exactly once, after the reduce
        assert r["weight_numel"] == 8 * 12 // world
        assert r["collective"] == "all_reduce"


def test_parallel_mlp_uses_exactly_one_allreduce() -> None:
    results = spawn_group(helpers.worker_mlp, 2)

    for r in results:
        assert r["max_err"] < TOL
        assert r["collectives"] == ["all_reduce"]


# --- refuse silent wrong answers --------------------------------------------------------
def test_sharded_layer_without_a_process_group_raises_instead_of_returning_garbage() -> None:
    w = torch.ones(4, 4)

    with pytest.raises(RuntimeError, match="process group"):
        RowParallelLinear(w, None, rank=0, world_size=2, input_is_parallel=False)(torch.ones(2, 4))
    with pytest.raises(RuntimeError, match="process group"):
        ColumnParallelLinear(w, None, rank=0, world_size=2)(torch.ones(2, 4))


def test_layer_built_for_a_different_world_size_than_the_group_raises() -> None:
    (message,) = set(spawn_group(helpers.worker_layer_without_group_check, 2, 4))

    assert "world_size=4" in message and "size 2" in message
