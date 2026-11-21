from __future__ import annotations

import pytest

from helpers import CM, cfg, req, run
from metrics import by_priority, percentile, summarize


def test_percentile_linear_interpolation() -> None:
    assert percentile([1, 2, 3, 4, 5], 50) == 3
    assert percentile([0, 10], 95) == pytest.approx(9.5)
    assert percentile([], 95) == 0.0


def test_summary_of_single_request_matches_hand_computation() -> None:
    r = req(1, p=100, o=3)
    s = summarize(run(cfg(), [r]))
    t1, t2, t3 = CM.iteration_time(100, 0, 0), CM.iteration_time(0, 1, 100), CM.iteration_time(0, 1, 101)
    total = t1 + t2 + t3
    assert s["makespan_ms"] == pytest.approx(total)
    assert s["throughput_tok_s"] == pytest.approx(3 / total * 1000)
    assert s["mean_latency_ms"] == pytest.approx(total)
    assert s["p95_latency_ms"] == pytest.approx(total)
    assert s["mean_ttft_ms"] == pytest.approx(t1)
    assert s["util_proxy"] == pytest.approx((100 + 1 + 1) * CM.compute_ms_per_token / total)
    assert s["busy_frac"] == pytest.approx(1.0)
    assert s["preemptions"] == 0


def test_idle_gaps_lower_utilisation_and_busy_fraction() -> None:
    s = summarize(run(cfg(), [req(1, 0.0, o=2), req(2, 5000.0, o=2)]))
    assert s["busy_frac"] < 0.05 and s["util_proxy"] < 0.05


def test_static_padding_lowers_proxy_versus_continuous() -> None:
    mk = lambda: [req(0, 0.0, p=32, o=4), req(1, 0.0, p=32, o=100)]
    cont = summarize(run(cfg(), mk()))
    static = summarize(run(cfg(kind="static"), mk()))
    assert static["mean_latency_ms"] >= cont["mean_latency_ms"] - 1e-9
    assert static["padded_slot_frac"] > 0 and cont["padded_slot_frac"] == 0


def test_by_priority_groups_latency() -> None:
    reqs = [req(0, 0.0, o=20, prio=0), req(1, 0.0, o=20, prio=1), req(2, 0.0, o=20, prio=1)]
    groups = by_priority(run(cfg(policy="priority", max_running_seqs=1), reqs))
    assert set(groups) == {0, 1}
    assert groups[0]["n"] == 1 and groups[1]["n"] == 2
    assert groups[0]["mean_latency_ms"] < groups[1]["mean_latency_ms"]


def test_summary_values_are_plain_floats_for_json() -> None:
    import json

    json.dumps(summarize(run(cfg(), [req(1)])))


def test_peak_waiting_reflects_queue_depth() -> None:
    reqs = [req(i, 0.0, o=10) for i in range(6)]
    s = summarize(run(cfg(max_running_seqs=2), reqs))
    assert s["peak_waiting"] == 4


def test_util_proxy_does_not_count_recomputed_work_as_useful() -> None:
    mk = lambda: [req(i, 0.0, p=64, o=200) for i in range(8)]
    res = run(cfg(num_kv_blocks=60), mk())
    s = summarize(res)
    assert s["recomputed_tokens"] > 0
    useful = sum((i.prefill_tokens + i.decode_tokens) for i in res.iterations) - s["recomputed_tokens"]
    assert s["util_proxy"] == pytest.approx(useful * CM.compute_ms_per_token / s["makespan_ms"])


def test_token_gap_percentile_sees_a_stall_that_the_per_request_mean_hides() -> None:
    decoder, long_req = req(0, 0.0, p=16, o=200), req(1, 100.0, p=2000, o=2)
    s = summarize(run(cfg(token_budget=4096), [decoder, long_req]))
    assert s["worst_itl_p95_ms"] >= s["p95_token_gap_ms"]
    assert s["max_token_gap_ms"] > 70  # the 2000-token prefill iteration
    assert s["p95_token_gap_ms"] < 20  # but a typical token is a ~7 ms decode step
