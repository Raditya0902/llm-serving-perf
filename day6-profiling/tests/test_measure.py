import pytest

from measure import (
    DRIFT_WARN_FRACTION,
    RequestMetrics,
    drift_fraction,
    interleaved,
    match_fraction,
    percentile,
    request_metrics,
    summarize_case,
)
from variants import DecodeRun


def run(batch: int = 1, prefill: float = 0.1, decode: float = 0.4, steps: int = 4, mem: int = 100) -> DecodeRun:
    tokens = tuple(tuple(range(steps + 1)) for _ in range(batch))
    return DecodeRun(tokens, prefill, decode, steps, (), mem)


def test_interleaved_rotates_the_start_of_each_round_and_discards_warmup():
    calls: list[str] = []
    cases = {"a": lambda: calls.append("a") or "A", "b": lambda: calls.append("b") or "B"}

    result = interleaved(cases, warmup=1, repeats=2, cooldown_s=0.0)

    assert result == {"a": ["A", "A"], "b": ["B", "B"]}
    assert calls[:2] == ["a", "b"]  # warmup
    assert calls[2:] == ["a", "b", "b", "a"]  # round 0 starts at a, round 1 starts at b


def test_interleaved_every_case_gets_every_position_over_a_full_cycle():
    order: list[str] = []
    cases = {name: (lambda n=name: order.append(n) or n) for name in "abc"}

    interleaved(cases, warmup=0, repeats=3)

    firsts = [order[i] for i in (0, 3, 6)]
    assert sorted(firsts) == ["a", "b", "c"]


def test_balanced_repeats_rounds_up_to_a_multiple_of_the_case_count():
    from measure import balanced_repeats

    assert balanced_repeats(5, 3) == 6
    assert balanced_repeats(6, 3) == 6
    assert balanced_repeats(5, 1) == 5
    with pytest.raises(ValueError):
        balanced_repeats(0, 3)


def test_interleaved_sleeps_between_rounds_when_cooldown_set():
    slept: list[float] = []

    interleaved({"a": lambda: 1}, warmup=0, repeats=3, cooldown_s=2.0, sleep=slept.append)

    assert slept == [2.0, 2.0]


def test_interleaved_validates_arguments():
    with pytest.raises(ValueError, match="repeats"):
        interleaved({"a": lambda: 1}, warmup=0, repeats=0)
    with pytest.raises(ValueError, match="warmup"):
        interleaved({"a": lambda: 1}, warmup=-1, repeats=1)


def test_request_metrics_for_sequential_requests_includes_queueing():
    # two B=1 requests served one after another: second waits for the first
    metrics = request_metrics([run(prefill=0.1, decode=0.4), run(prefill=0.1, decode=0.4)])

    assert metrics.wall_s == pytest.approx(1.0)
    assert metrics.ttfts_s == pytest.approx((0.1, 0.6))
    assert metrics.output_tokens == 10
    assert metrics.tpots_s == pytest.approx((0.1, 0.1))


def test_request_metrics_for_one_batch_shares_ttft_and_counts_all_rows():
    metrics = request_metrics([run(batch=4, prefill=0.2, decode=0.4, mem=500)])

    assert metrics.ttfts_s == pytest.approx((0.2,) * 4)
    assert metrics.output_tokens == 20
    assert metrics.peak_mem_bytes == 500


def test_request_metrics_rejects_empty_input():
    with pytest.raises(ValueError, match="batch"):
        request_metrics([])


def test_percentile_interpolates_linearly():
    assert percentile([1.0, 2.0, 3.0, 4.0, 5.0], 50) == pytest.approx(3.0)
    assert percentile([0.0, 10.0], 95) == pytest.approx(9.5)


def test_percentile_rejects_empty():
    with pytest.raises(ValueError, match="empty"):
        percentile([], 95)


def test_match_fraction_compares_rows_to_reference():
    ref = ((1, 2), (3, 4))
    assert match_fraction(ref, ref) == 1.0
    assert match_fraction(((1, 2), (3, 9)), ref) == 0.5


def test_drift_fraction_compares_first_and_second_half_medians():
    assert drift_fraction([1.0, 1.0, 1.1, 1.1]) == pytest.approx(0.1 / 1.05, rel=1e-3)
    assert drift_fraction([1.0, 1.0]) == 0.0  # too few repeats to judge
    assert DRIFT_WARN_FRACTION == 0.10


def test_summarize_case_reports_workload_level_metrics():
    reps = [[run(prefill=0.1, decode=0.4)] * 2 for _ in range(3)]  # 2 sequential requests x 3 reps

    row = summarize_case("1", reps, n_requests=2)

    assert row["n_reps"] == 3
    assert row["requests_per_s"] == pytest.approx(2.0)
    assert row["output_tok_s"] == pytest.approx(10.0)
    assert row["ttft_ms_p95"] == pytest.approx(percentile([100, 600] * 3, 95))
    assert row["tpot_ms_p95"] == pytest.approx(100.0)
    assert row["peak_mem_mb"] == pytest.approx(100 / 1024 / 1024)
    assert row["wall_spread_fraction"] == 0.0


def test_request_metrics_dataclass_is_frozen():
    metrics = RequestMetrics(1.0, (0.1,), (0.1,), 5, 1)
    with pytest.raises(AttributeError):
        metrics.wall_s = 2.0  # type: ignore[misc]


def _row(label: str, rps: float, lo: float, hi: float) -> dict:
    base = {k: 10.0 for k in ("output_tok_s", "ttft_ms_p95", "tpot_ms_p95", "peak_mem_mb", "wall_s")}
    return {"label": label, "requests_per_s": rps, "requests_per_s_min": lo,
            "requests_per_s_max": hi, **base}


def test_with_baseline_deltas_computes_percent_change_and_flags_noise():
    from measure import with_baseline_deltas

    rows = [_row("1", 1.0, 0.95, 1.05), _row("2", 1.5, 1.4, 1.6), _row("3", 1.02, 0.98, 1.1)]

    out = {r["label"]: r for r in with_baseline_deltas(rows, "1")}

    assert out["1"]["is_baseline"] is True
    assert out["2"]["requests_per_s_vs_baseline_pct"] == pytest.approx(50.0)
    assert out["2"]["rps_range_overlaps_baseline"] is False
    assert out["3"]["rps_range_overlaps_baseline"] is True  # inside the baseline's own spread
    assert "is_baseline" not in rows[0]  # inputs are not mutated


def test_with_baseline_deltas_rejects_missing_baseline():
    from measure import with_baseline_deltas

    with pytest.raises(ValueError, match="baseline"):
        with_baseline_deltas([_row("1", 1.0, 1.0, 1.0)], "9")
