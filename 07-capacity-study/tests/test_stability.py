"""Aggregation over repeats, coefficient of variation, and the A/A noise-floor rule."""
from __future__ import annotations

import math

import pytest

import stability as st


def _row(case_id="B-c16-baseline", rep=0, **over):
    base = {
        "case_id": case_id, "config_id": "abc", "workload": "B", "config": "baseline",
        "concurrency": 16, "rep": rep, "is_control": False, "source": "sim-day3",
        "simulated": True, "completed": 128, "output_tok_s": 100.0, "p95_e2e_ms": 1000.0,
        "mean_ttft_ms": 50.0, "peak_gpu_mem_gib": None, "preemptions": 0,
    }
    return {**base, **over}


def test_cv_is_sample_std_over_mean():
    assert st.cv([10.0, 12.0, 14.0]) == pytest.approx(2.0 / 12.0)


def test_cv_is_none_for_single_value_or_zero_mean():
    assert st.cv([5.0]) is None
    assert st.cv([0.0, 0.0]) is None


def test_aggregate_groups_by_case_and_counts_reps():
    rows = [_row(rep=0, output_tok_s=100), _row(rep=1, output_tok_s=110), _row(rep=2, output_tok_s=90),
            _row("A-c4-tuned", rep=0, workload="A", config="tuned", concurrency=4)]
    summary = {s["case_id"]: s for s in st.aggregate_cases(rows)}
    b = summary["B-c16-baseline"]
    assert b["n_reps"] == 3
    assert b["output_tok_s"]["mean"] == pytest.approx(100.0)
    assert b["output_tok_s"]["min"] == 90 and b["output_tok_s"]["max"] == 110
    assert b["output_tok_s"]["cv"] == pytest.approx(10.0 / 100.0)
    assert summary["A-c4-tuned"]["n_reps"] == 1
    assert summary["A-c4-tuned"]["output_tok_s"]["cv"] is None


def test_unmeasured_metric_stays_none_not_zero():
    summary = st.aggregate_cases([_row(rep=0), _row(rep=1)])[0]
    assert summary["peak_gpu_mem_gib"] is None


def test_aggregate_keeps_identity_fields_and_source():
    summary = st.aggregate_cases([_row(rep=0)])[0]
    assert (summary["workload"], summary["config"], summary["concurrency"]) == ("B", "baseline", 16)
    assert summary["source"] == "sim-day3" and summary["simulated"] is True


def test_aggregate_rejects_mixed_config_ids_in_one_case():
    with pytest.raises(ValueError, match="config_id"):
        st.aggregate_cases([_row(rep=0), _row(rep=1, config_id="zzz")])


def test_unstable_metrics_are_flagged_above_five_percent_cv():
    rows = [_row(rep=0, output_tok_s=100), _row(rep=1, output_tok_s=130), _row(rep=2, output_tok_s=70)]
    flagged = st.unstable_metrics(st.aggregate_cases(rows)[0])
    assert "output_tok_s" in flagged and "p95_e2e_ms" not in flagged


def test_low_sample_p95_is_marked():
    small = st.aggregate_cases([_row(completed=100)])[0]
    large = st.aggregate_cases([_row(completed=512)])[0]
    assert small["p95_low_sample"] and not large["p95_low_sample"]


def test_ranges_overlap_is_inclusive():
    assert st.ranges_overlap((1, 3), (3, 5))
    assert not st.ranges_overlap((1, 2), (3, 5))


def test_aa_floor_uses_larger_of_observed_delta_and_cv():
    original = st.aggregate_cases([_row(rep=i, output_tok_s=v) for i, v in enumerate((100, 102, 98))])[0]
    control = st.aggregate_cases(
        [_row("B-c16-baseline-aa", rep=i, is_control=True, output_tok_s=v) for i, v in enumerate((104, 106, 102))]
    )[0]
    floor = st.aa_floor_pct(original, control, "output_tok_s")
    assert floor == pytest.approx(4.0)  # observed A/A delta (4%) beats cv (2%)


def test_compare_calls_a_difference_real_only_if_it_clears_range_and_floor():
    base = st.aggregate_cases([_row(rep=i, output_tok_s=v) for i, v in enumerate((100, 101, 99))])[0]
    big = st.aggregate_cases([_row("x", rep=i, output_tok_s=v) for i, v in enumerate((120, 121, 119))])[0]
    near = st.aggregate_cases([_row("y", rep=i, output_tok_s=v) for i, v in enumerate((100.5, 101.5, 99.5))])[0]
    assert st.compare(base, big, "output_tok_s", floor_pct=2.0)["verdict"] == "real"
    assert st.compare(base, near, "output_tok_s", floor_pct=2.0)["verdict"] == "within noise"
    assert st.compare(base, big, "output_tok_s", floor_pct=2.0)["delta_pct"] == pytest.approx(20.0)


def test_compare_overlapping_ranges_are_never_real_even_if_delta_exceeds_floor():
    base = st.aggregate_cases([_row(rep=i, output_tok_s=v) for i, v in enumerate((90, 110))])[0]
    other = st.aggregate_cases([_row("x", rep=i, output_tok_s=v) for i, v in enumerate((100, 125))])[0]
    out = st.compare(base, other, "output_tok_s", floor_pct=1.0)
    assert out["ranges_overlap"] and out["verdict"] == "within noise"


def test_compare_single_rep_is_not_enough_to_call_anything_real():
    one = st.aggregate_cases([_row(rep=0, output_tok_s=100)])[0]
    two = st.aggregate_cases([_row("x", rep=0, output_tok_s=200)])[0]
    assert st.compare(one, two, "output_tok_s", floor_pct=1.0)["verdict"] == "insufficient reps"


def test_compare_from_a_zero_baseline_uses_ranges_not_a_percentage():
    zero = st.aggregate_cases([_row(rep=i, preemptions=0) for i in range(3)])[0]
    some = st.aggregate_cases([_row("x", rep=i, preemptions=v) for i, v in enumerate((4, 5, 6))])[0]
    out = st.compare(zero, some, "preemptions", floor_pct=1.0)
    assert out["delta_pct"] is None and out["verdict"] == "real"
    assert st.compare(zero, zero, "preemptions", floor_pct=1.0)["verdict"] == "within noise"


def test_compare_with_unmeasured_metric():
    a = st.aggregate_cases([_row(rep=0), _row(rep=1)])[0]
    b = st.aggregate_cases([_row("x", rep=0), _row("x", rep=1)])[0]
    assert st.compare(a, b, "peak_gpu_mem_gib", floor_pct=1.0)["verdict"] == "not measured"


def test_nan_values_are_rejected():
    with pytest.raises(ValueError, match="finite"):
        st.aggregate_cases([_row(output_tok_s=math.nan)])


def test_duplicate_case_and_rep_is_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        st.aggregate_cases([_row(rep=0), _row(rep=0)])


def test_mixed_sources_under_one_case_are_rejected():
    with pytest.raises(ValueError, match="source"):
        st.aggregate_cases([_row(rep=0), _row(rep=1, source="vllm-gpu", simulated=False)])


def test_metric_missing_in_some_reps_is_unmeasured_not_averaged_over_the_rest():
    rows = [_row(rep=0, kv_peak_frac=0.5), _row(rep=1, kv_peak_frac=None), _row(rep=2, kv_peak_frac=0.7)]
    assert st.aggregate_cases(rows)[0]["kv_peak_frac"] is None
