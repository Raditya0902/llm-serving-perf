import json

import pytest

from conftest import SYNTHETIC_NOTE
from make_performance_report import (
    MISSING,
    control_note,
    stability_note,
    build_report,
    comparison_table,
    difference,
    find_row,
    main,
)


def row(label, rps, tok, ttft, tpot, mem, base=False, match=1.0, spread=0.05):
    return {
        "label": label, "requests_per_s": rps, "output_tok_s": tok, "ttft_ms_p95": ttft,
        "tpot_ms_p95": tpot, "peak_mem_mb": mem, "tokens_match": match, "is_baseline": base,
        "requests_per_s_vs_baseline_pct": 0.0, "requests_per_s_min": rps * (1 - spread),
        "requests_per_s_max": rps * (1 + spread), "synthetic_note": SYNTHETIC_NOTE,
    }


@pytest.fixture()
def raw(tmp_path):
    rows = [row("day1", 1.0, 100.0, 50.0, 10.0, 1000.0, base=True),
            row("deferred_sync", 1.1, 110.0, 45.0, 9.0, 1000.0)]
    (tmp_path / "exp_o1.json").write_text(json.dumps({"meta": {}, "rows": rows}))
    return tmp_path


def test_difference_reports_absolute_and_percent_change():
    assert difference({"x": 10.0}, {"x": 12.0}, "x") == "+2 (+20.0%)"
    assert difference({"x": 10.0}, {"x": 5.0}, "x") == "-5 (-50.0%)"


def test_difference_is_not_measured_when_a_side_lacks_the_metric():
    assert difference({"x": None}, {"x": 1.0}, "x") == MISSING
    assert difference({}, {"x": 1.0}, "x") == MISSING


def test_difference_handles_zero_baseline_without_dividing():
    assert "n/a" in difference({"x": 0.0}, {"x": 2.0}, "x")


def test_comparison_table_has_the_five_required_metric_rows():
    table = comparison_table(row("a", 1, 2, 3, 4, 5), row("b", 2, 4, 6, 8, 10), "a", "b")

    for name in ("Requests/sec", "Output tokens/sec", "P95 TTFT", "P95 TPOT", "Peak memory"):
        assert name in table
    assert "| Requests/sec | 1.000 | 2.000 | +1 (+100.0%) |" in table


def test_comparison_table_marks_unmeasured_memory():
    mac = row("a", 1, 2, 3, 4, None)
    assert f"| Peak memory (MB) | {MISSING} | {MISSING} | {MISSING} |" in comparison_table(mac, mac, "a", "b")


def test_find_row_raises_for_unknown_label(raw):
    with pytest.raises(KeyError, match="nope"):
        find_row(raw, "O1:nope")
    with pytest.raises(FileNotFoundError):
        find_row(raw, "E9:x")


def test_build_report_flags_noise_and_includes_experiment_table(raw):
    report = build_report(raw, "O1:day1", "O1:deferred_sync")

    assert "NOT clearly above run-to-run variation" in report
    assert "| deferred_sync |" in report
    assert "(baseline)" in report


def test_main_refuses_to_overwrite_the_hand_written_report(raw, tmp_path):
    with pytest.raises(SystemExit, match="hand-written"):
        main(["--raw", str(raw), "--baseline", "O1:day1", "--modified", "O1:deferred_sync",
              "--out", str(tmp_path / "performance-report.md")])


def test_main_writes_generated_file(raw, tmp_path):
    out = tmp_path / "out" / "tables.md"

    assert main(["--raw", str(raw), "--baseline", "O1:day1", "--modified", "O1:deferred_sync",
                 "--out", str(out)]) == 0
    assert out.read_text().startswith("# Generated tables")


def test_control_note_reports_the_ab_noise_floor():
    control = {**row("day1_control", 1.0, 100.0, 50.0, 10.0, 1000.0), "is_control": True,
               "requests_per_s_vs_baseline_pct": 3.2, "tpot_ms_p95_vs_baseline_pct": -1.5}

    note = control_note([row("day1", 1.0, 100.0, 50.0, 10.0, 1000.0, base=True), control])

    assert "+3.2%" in note and "-1.5%" in note and "noise floor" in note
    assert control_note([row("day1", 1.0, 100.0, 50.0, 10.0, 1000.0, base=True)]) == ""


def test_stability_note_flags_rows_with_large_repeat_spread():
    steady = {**row("a", 1.0, 1.0, 1.0, 1.0, 1.0), "wall_spread_fraction": 0.05}
    shaky = {**row("b", 1.0, 1.0, 1.0, 1.0, 1.0), "wall_spread_fraction": 0.43}

    assert stability_note([steady]) == ""
    note = stability_note([steady, shaky])
    assert "b (43%)" in note and "a (" not in note


def test_report_recomputes_range_overlap_from_the_two_chosen_rows(tmp_path):
    rows = [row("a", 1.0, 1, 1, 1, 1, base=True, spread=0.01), row("b", 1.5, 1, 1, 1, 1, spread=0.01)]
    (tmp_path / "exp_x1.json").write_text(json.dumps({"meta": {}, "rows": rows}))

    report = build_report(tmp_path, "X1:a", "X1:b")

    assert "do not overlap" in report and "NOT clearly" not in report


def test_report_rejects_a_non_baseline_row_as_the_baseline(raw):
    with pytest.raises(ValueError, match="not the experiment's baseline"):
        build_report(raw, "O1:deferred_sync", "O1:day1")


def test_report_rejects_cross_experiment_pairs(raw, tmp_path):
    (raw / "exp_o2.json").write_text(json.dumps({"meta": {}, "rows": [row("sdpa", 1, 1, 1, 1, 1, base=True)]}))

    with pytest.raises(ValueError, match="same experiment"):
        build_report(raw, "O1:day1", "O2:sdpa")


def test_report_warns_when_the_modified_outputs_diverged(tmp_path):
    rows = [row("a", 1.0, 1, 1, 1, 1, base=True), row("b", 1.2, 1, 1, 1, 1, match=0.5)]
    (tmp_path / "exp_x1.json").write_text(json.dumps({"meta": {}, "rows": rows}))

    assert "outputs diverged" in build_report(tmp_path, "X1:a", "X1:b")
