from __future__ import annotations

import math

import pytest

from bench_model import (
    bus_bandwidth_factor,
    fit_alpha_beta,
    fit_max_rel_error,
    iters_for,
    median_across_runs,
    merge_ranks,
    mlp_cost,
    predict_tp_speedup,
)


# --- fit_alpha_beta ----------------------------------------------------------------------
def test_fit_recovers_known_latency_and_per_byte_cost() -> None:
    alpha, beta = 20e-6, 2e-9
    sizes = [1_000, 10_000, 100_000, 1_000_000]

    got_alpha, got_beta = fit_alpha_beta(sizes, [alpha + beta * s for s in sizes])

    assert got_alpha == pytest.approx(alpha, rel=1e-6)
    assert got_beta == pytest.approx(beta, rel=1e-6)


def test_fit_weights_relative_error_so_large_messages_do_not_swamp_latency() -> None:
    alpha, beta = 20e-6, 2e-9
    sizes = [1_000, 10_000, 100_000, 1_000_000, 10_000_000, 100_000_000]
    times = [alpha + beta * s for s in sizes]
    times[-1] *= 1.5  # one slow huge-message outlier (cache/memory effects)

    got_alpha, _ = fit_alpha_beta(sizes, times)

    assert got_alpha > 0
    assert got_alpha == pytest.approx(alpha, rel=0.5)


def test_fit_max_rel_error_is_zero_for_exact_data_and_flags_misfit() -> None:
    sizes = [1_000, 100_000, 10_000_000]
    exact = [20e-6 + 2e-9 * s for s in sizes]

    assert fit_max_rel_error(sizes, exact, 20e-6, 2e-9) == pytest.approx(0.0, abs=1e-12)
    off = [exact[0], exact[1], exact[2] * 2]
    assert fit_max_rel_error(sizes, off, 20e-6, 2e-9) == pytest.approx(0.5)


def test_fit_needs_two_distinct_sizes() -> None:
    with pytest.raises(ValueError, match="two"):
        fit_alpha_beta([100], [1e-5])
    with pytest.raises(ValueError, match="distinct"):
        fit_alpha_beta([100, 100], [1e-5, 2e-5])


def test_fit_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="same length"):
        fit_alpha_beta([1, 2, 3], [1.0, 2.0])


# --- bus bandwidth -----------------------------------------------------------------------
@pytest.mark.parametrize(
    ("collective", "world", "expected"),
    [("all_reduce", 2, 1.0), ("all_reduce", 4, 1.5), ("all_gather", 2, 0.5),
     ("all_gather", 4, 0.75), ("broadcast", 4, 1.0)],
)
def test_bus_bandwidth_factor_follows_nccl_tests_definitions(collective, world, expected) -> None:
    assert bus_bandwidth_factor(collective, world) == pytest.approx(expected)


def test_bus_bandwidth_factor_rejects_unknown_collective() -> None:
    with pytest.raises(ValueError, match="collective"):
        bus_bandwidth_factor("reduce_scatter", 2)


# --- iteration budget --------------------------------------------------------------------
def test_iters_for_shrinks_for_huge_messages_but_never_below_floor() -> None:
    assert iters_for(1_024, 20) == 20
    assert iters_for(64 * 1024 * 1024, 20) < 20
    assert iters_for(64 * 1024 * 1024, 20) >= 3
    assert iters_for(64 * 1024 * 1024, 2) == 2  # never exceeds the requested base


# --- merge_ranks -------------------------------------------------------------------------
def test_merge_ranks_takes_slowest_rank_per_key() -> None:
    rank0 = [{"size": 1, "t": 1.0, "err": 0.1}, {"size": 2, "t": 5.0, "err": 0.0}]
    rank1 = [{"size": 1, "t": 3.0, "err": 0.0}, {"size": 2, "t": 4.0, "err": 0.2}]

    merged = merge_ranks([rank0, rank1], keys=("size",), max_fields=("t", "err"))

    assert merged == [{"size": 1, "t": 3.0, "err": 0.1}, {"size": 2, "t": 5.0, "err": 0.2}]


def test_merge_ranks_rejects_ranks_with_different_cases() -> None:
    with pytest.raises(ValueError, match="same cases"):
        merge_ranks([[{"size": 1, "t": 1.0}], [{"size": 2, "t": 1.0}]], ("size",), ("t",))


def test_merge_ranks_does_not_mutate_inputs() -> None:
    rank0 = [{"size": 1, "t": 1.0}]
    rank1 = [{"size": 1, "t": 2.0}]

    merge_ranks([rank0, rank1], ("size",), ("t",))

    assert rank0 == [{"size": 1, "t": 1.0}]


# --- performance model -------------------------------------------------------------------
def test_mlp_cost_counts_flops_weights_and_activation_bytes() -> None:
    flops, weight_bytes, comm_bytes = mlp_cost(tokens=8, d_model=16, d_ff=64, n_matrices=2, bytes_per_elem=4)

    assert flops == 2 * 8 * 16 * 64 * 2
    assert weight_bytes == 16 * 64 * 2 * 4
    assert comm_bytes == 8 * 16 * 4  # one [tokens, d_model] AllReduce


def _predict(world: int, **over):
    kw = dict(flops=1e9, weight_bytes=0.0, comm_bytes=1e6, device_flops=1e11, device_bw=math.inf,
              comm_alpha_s=1e-5, comm_beta_s_per_byte=1e-9, world=world)
    kw.update(over)
    return predict_tp_speedup(**kw)


def test_prediction_world_one_has_no_communication_and_speedup_one() -> None:
    p = _predict(1)

    assert p.t_comm == 0.0
    assert p.speedup == pytest.approx(1.0)


def test_prediction_compute_bound_with_cheap_comm_approaches_world_size() -> None:
    p = _predict(2, flops=1e12, comm_bytes=1e3)  # 10 s of compute, ~10 us of comm

    assert p.speedup == pytest.approx(2.0, rel=1e-3)
    assert p.comm_fraction < 1e-3


def test_prediction_small_work_with_costly_comm_loses_to_single_device() -> None:
    p = _predict(2, flops=1e6, comm_bytes=1e6)  # 10 us compute vs ~1 ms comm

    assert p.speedup < 1.0
    assert p.comm_fraction > 0.5


def test_prediction_memory_bound_decode_uses_weight_bandwidth() -> None:
    # 1e9 B weights at 1e11 B/s = 10 ms; flops term negligible. TP halves that read.
    p = _predict(2, flops=1.0, weight_bytes=1e9, device_bw=1e11, comm_bytes=0.0, comm_alpha_s=0.0)

    assert p.t_single == pytest.approx(1e9 / 1e11)
    assert p.t_tp_compute == pytest.approx(0.5e9 / 1e11)


def test_prediction_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError, match="world"):
        _predict(0)
    with pytest.raises(ValueError, match="device_flops"):
        _predict(2, device_flops=0.0)


# --- median_across_runs ------------------------------------------------------------------
def test_median_across_runs_takes_median_and_relative_spread_per_case() -> None:
    runs = [
        [{"case": "a", "t": 1.0, "err": 0.1}],
        [{"case": "a", "t": 3.0, "err": 0.0}],
        [{"case": "a", "t": 2.0, "err": 0.3}],
    ]

    (row,) = median_across_runs(runs, keys=("case",), fields=("t",), max_fields=("err",))

    assert row["t"] == 2.0
    assert row["t_spread"] == pytest.approx((3.0 - 1.0) / 2.0)
    assert row["err"] == 0.3


def test_median_across_runs_single_run_has_zero_spread() -> None:
    (row,) = median_across_runs([[{"case": 1, "t": 5.0}]], ("case",), ("t",))

    assert row["t"] == 5.0 and row["t_spread"] == 0.0


def test_median_across_runs_rejects_runs_with_different_cases() -> None:
    with pytest.raises(ValueError, match="same cases"):
        median_across_runs([[{"c": 1, "t": 1.0}], [{"c": 2, "t": 1.0}]], ("c",), ("t",))


def test_median_across_runs_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="at least one"):
        median_across_runs([], ("c",), ("t",))
