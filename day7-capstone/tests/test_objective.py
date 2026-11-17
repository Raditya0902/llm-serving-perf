"""The pre-registered optimization objective and the before/after comparison."""
from __future__ import annotations

import pytest

import objective as ob
import stability as st


def _rows(case_id, workload, config, conc, tok_s, p95, reps=3, spread=0.5, **extra):
    out = []
    for rep in range(reps):
        wobble = (rep - 1) * spread
        out.append({
            "case_id": case_id, "config_id": f"id-{case_id}", "workload": workload, "config": config,
            "concurrency": conc, "rep": rep, "is_control": case_id.endswith("-aa"),
            "source": "sim-day3", "simulated": True, "completed": 128,
            "output_tok_s": tok_s + wobble, "p95_e2e_ms": p95 + wobble, "mean_ttft_ms": 100.0 + wobble,
            "mean_tpot_ms": 20.0 + wobble / 10, "req_per_s": tok_s / 100, "preemptions": 0, **extra,
        })
    return out


def _summaries(spec):
    rows = [r for args in spec for r in _rows(*args)]
    return st.aggregate_cases(rows)


# (case_id, workload, config, conc, tok_s, p95_e2e)
GRID = [
    ("B-c4-baseline", "B", "baseline", 4, 100, 1000), ("B-c4-tuned", "B", "tuned", 4, 100, 1000),
    ("B-c16-baseline", "B", "baseline", 16, 300, 2500), ("B-c16-tuned", "B", "tuned", 16, 330, 2400),
    ("B-c64-baseline", "B", "baseline", 64, 500, 9000), ("B-c64-tuned", "B", "tuned", 64, 600, 8000),
    ("B-c16-baseline-aa", "B", "baseline", 16, 300.5, 2500),
]


def test_threshold_is_k_times_baseline_p95_at_lowest_concurrency():
    s = _summaries(GRID)
    assert ob.threshold_ms(s, "B", k=3.0) == pytest.approx(3000.0)
    assert ob.K_THRESHOLD == 3.0


def test_threshold_requires_the_baseline_floor_case():
    s = [x for x in _summaries(GRID) if x["case_id"] != "B-c4-baseline"]
    with pytest.raises(ValueError, match="B-c4-baseline"):
        ob.threshold_ms(s, "B", k=3.0)


def test_select_picks_max_throughput_among_feasible_and_excludes_control():
    decision = ob.select(_summaries(GRID), "B", k=3.0)
    assert decision.threshold_ms == pytest.approx(3000.0)
    assert set(decision.feasible) == {"B-c4-baseline", "B-c4-tuned", "B-c16-baseline", "B-c16-tuned"}
    assert decision.best_case == "B-c16-tuned"  # c64 cells blow the latency limit


def test_tuned_gain_that_clears_noise_is_adopted():
    decision = ob.select(_summaries(GRID), "B", k=3.0)
    assert decision.verdict == "real" and decision.recommendation.startswith("adopt tuned")
    assert decision.delta_pct == pytest.approx(10.0, abs=0.5)


def test_gain_inside_noise_keeps_baseline():
    grid = [g if g[0] != "B-c16-tuned" else ("B-c16-tuned", "B", "tuned", 16, 300.4, 2400) for g in GRID]
    decision = ob.select(_summaries(grid), "B", k=3.0)
    assert decision.best_case == "B-c16-tuned"
    assert decision.verdict == "within noise" and decision.recommendation.startswith("keep baseline")


def test_baseline_winner_means_no_change():
    grid = [g if g[0] != "B-c16-tuned" else ("B-c16-tuned", "B", "tuned", 16, 250, 2400) for g in GRID]
    decision = ob.select(_summaries(grid), "B", k=3.0)
    assert decision.best_case == "B-c16-baseline" and decision.recommendation.startswith("keep baseline")


def test_nothing_feasible_returns_no_winner():
    grid = [("B-c4-baseline", "B", "baseline", 4, 100, 1000), ("B-c16-baseline", "B", "baseline", 16, 300, 5000)]
    decision = ob.select(_summaries(grid), "B", k=0.9)  # limit below the floor itself
    assert decision.best_case is None and decision.recommendation.startswith("no feasible")


def test_select_ignores_other_workloads():
    mixed = GRID + [("A-c4-baseline", "A", "baseline", 4, 9999, 10), ("A-c4-tuned", "A", "tuned", 4, 9999, 10)]
    assert ob.select(_summaries(mixed), "B", k=3.0).best_case == "B-c16-tuned"


def test_before_after_labels_direction_per_metric():
    s = {x["case_id"]: x for x in _summaries(GRID)}
    table = {r["metric"]: r for r in ob.before_after(s["B-c64-baseline"], s["B-c64-tuned"], floor_pct=1.0)}
    assert table["output_tok_s"]["direction"] == "improved"
    assert table["p95_e2e_ms"]["direction"] == "improved"  # lower latency is better
    assert table["output_tok_s"]["delta_pct"] == pytest.approx(20.0, abs=0.5)


def test_before_after_marks_regression_and_unchanged():
    base = _summaries([("x-base", "B", "baseline", 16, 100, 1000)])[0]
    worse = _summaries([("x-new", "B", "tuned", 16, 100, 1300)])[0]
    rows = {r["metric"]: r for r in ob.before_after(base, worse, floor_pct=1.0)}
    assert rows["p95_e2e_ms"]["direction"] == "regressed"
    assert rows["output_tok_s"]["direction"] == "unchanged"


def test_before_after_skips_unmeasured_metrics_instead_of_inventing_them():
    base = _summaries([("x-base", "B", "baseline", 16, 100, 1000)])[0]
    rows = {r["metric"]: r for r in ob.before_after(base, base, floor_pct=1.0)}
    assert rows["peak_gpu_mem_gib"]["direction"] == "not measured"


def test_noise_floor_uses_the_aa_control_when_present():
    s = _summaries(GRID)
    original = next(x for x in s if x["case_id"] == "B-c16-baseline")
    control = next(x for x in s if x["is_control"])
    assert ob.noise_floor_pct(s, "output_tok_s") == pytest.approx(
        st.aa_floor_pct(original, control, "output_tok_s"))


def test_noise_floor_falls_back_to_the_largest_cv_without_a_control():
    s = [x for x in _summaries(GRID) if not x["is_control"]]
    widest = max(x["output_tok_s"]["cv"] for x in s) * 100
    assert ob.noise_floor_pct(s, "output_tok_s") == pytest.approx(widest)


def test_pair_floor_is_widened_by_a_noisy_case():
    quiet = _summaries(GRID)
    noisy = quiet + st.aggregate_cases(_rows("N-c16-tuned", "B", "tuned", 16, 300, 2500, spread=20.0))
    a = next(x for x in quiet if x["case_id"] == "B-c16-baseline")
    b = next(x for x in noisy if x["case_id"] == "N-c16-tuned")
    assert ob.pair_floor_pct(noisy, a, b, "output_tok_s") >= b["output_tok_s"]["cv"] * 100
    assert ob.pair_floor_pct(noisy, a, b, "output_tok_s") > ob.noise_floor_pct(noisy, "output_tok_s")


# ---------------------------------------------------------------- review: compare against a *feasible* baseline
LATENCY_GRID = [
    ("B-c4-baseline", "B", "baseline", 4, 100, 1000), ("B-c4-tuned", "B", "tuned", 4, 100, 1000),
    ("B-c16-baseline", "B", "baseline", 16, 300, 2500), ("B-c16-tuned", "B", "tuned", 16, 310, 2500),
    ("B-c64-baseline", "B", "baseline", 64, 590, 5000),  # misses L = 3000
    ("B-c64-tuned", "B", "tuned", 64, 600, 2900),         # meets it
]


def test_tuned_that_alone_meets_the_limit_is_compared_with_the_best_feasible_baseline():
    d = ob.select(_summaries(LATENCY_GRID), "B", k=3.0)
    assert d.best_case == "B-c64-tuned"
    assert d.baseline_case == "B-c16-baseline"          # the best baseline that actually meets L
    assert d.same_concurrency_baseline_feasible is False
    assert d.verdict == "real" and d.recommendation.startswith("adopt tuned")
    assert "only" in d.recommendation or "feasible" in d.recommendation


def test_a_real_gain_whose_latency_is_within_noise_of_the_limit_is_not_adopted():
    rows = [r for g in LATENCY_GRID if g[0] != "B-c64-tuned" for r in _rows(*g)]
    rows += _rows("B-c64-tuned", "B", "tuned", 64, 600, 2995, spread=40.0)  # 0.17% under L, CV ~1.3%
    d = ob.select(st.aggregate_cases(rows), "B", k=3.0)
    assert d.best_case == "B-c64-tuned" and d.verdict == "real"
    assert d.recommendation.startswith("keep baseline") and "limit" in d.recommendation


def test_decision_reports_whether_the_same_concurrency_baseline_is_feasible():
    assert ob.select(_summaries(GRID), "B", k=3.0).same_concurrency_baseline_feasible is True
