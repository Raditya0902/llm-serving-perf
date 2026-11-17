"""make_report.py renders tables only and never overwrites a hand-written file."""
from __future__ import annotations

import pytest

import analyze as an
import make_report as mr
import matrix as mx
import multi_gpu
import results_io as rio
import run_matrix_sim as rms
import stability as st


@pytest.fixture(scope="module")
def result_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("sim")
    rows = rms.run_matrix(mx.build_runs(reps=2, key_reps=2))
    meta = rio.run_meta(simulated=True, note="test")
    rio.write_results("matrix", rows, meta, out)
    rio.write_results("matrix_summary", st.aggregate_cases(rows), meta, out)
    abl = rms.run_matrix(rms.ablation_runs(1))
    rio.write_results("ablation_summary", st.aggregate_cases(abl), meta, out)
    assert an.main(["--dir", str(out)]) == 0
    return out


def test_fmt_renders_none_as_not_measured():
    assert mr.fmt(None) == "not measured"
    assert mr.fmt(3.14159, 2) == "3.14"
    assert mr.fmt(12345.6, 0) == "12,346"


def test_render_is_labelled_simulated_and_has_every_section(result_dir):
    text = mr.render(result_dir)
    assert text.startswith(mr.GENERATED_MARKER)
    assert "SIMULATED" in text.splitlines()[2] or "SIMULATED" in text[:600]
    for heading in ("## Matrix", "## Stability", "## Objective", "## Before / after", "## Bottleneck",
                    "## Ablation", "## Multi-GPU"):
        assert heading in text


def test_matrix_table_has_twelve_case_rows_and_marks_unmeasured_memory(result_dir):
    text = mr.render(result_dir)
    for case in mx.build_matrix():
        assert f"| {case.case_id} |" in text
    assert "not measured" in text  # peak GPU memory is not measured in a simulation


def test_decision_section_quotes_threshold_and_recommendation(result_dir):
    text = mr.render(result_dir)
    assert "k = 3.0" in text and "keep baseline" in text


def test_measured_results_are_not_labelled_simulated(tmp_path):
    rows = rms.run_matrix(mx.build_runs(reps=2, key_reps=2, include_control=False))
    for r in rows:
        r["simulated"] = False
    meta = rio.run_meta(simulated=False, note="gpu")
    rio.write_results("matrix_summary", st.aggregate_cases(rows), meta, tmp_path)
    assert an.main(["--dir", str(tmp_path)]) == 0
    assert "SIMULATED" not in mr.render(tmp_path).split("## Matrix")[0]


def test_write_creates_the_generated_file(result_dir, tmp_path):
    out = tmp_path / "tables.generated.md"
    assert mr.main(["--dir", str(result_dir), "--out", str(out)]) == 0
    assert out.read_text().startswith(mr.GENERATED_MARKER)


def test_write_overwrites_its_own_previous_output(result_dir, tmp_path):
    out = tmp_path / "t.md"
    assert mr.main(["--dir", str(result_dir), "--out", str(out)]) == 0
    assert mr.main(["--dir", str(result_dir), "--out", str(out)]) == 0


def test_refuses_to_overwrite_a_hand_written_file(result_dir, tmp_path, capsys):
    out = tmp_path / "REPORT.md"
    out.write_text("# my hand-written report\n")
    assert mr.main(["--dir", str(result_dir), "--out", str(out)]) == 2
    assert out.read_text() == "# my hand-written report\n"
    assert "hand-written" in capsys.readouterr().err


def test_missing_inputs_fail_clearly(tmp_path, capsys):
    assert mr.main(["--dir", str(tmp_path), "--out", str(tmp_path / "x.md")]) == 2
    assert "analysis.json" in capsys.readouterr().err


def test_multi_gpu_table_uses_the_assumed_links(result_dir):
    text = mr.render(result_dir)
    for link in multi_gpu.LINKS:
        assert link in text
    assert "ASSUMED" in text


def test_stale_analysis_is_refused(result_dir, tmp_path, capsys):
    import shutil
    work = tmp_path / "copy"
    shutil.copytree(result_dir, work)
    summary = rio.read_results(work / "matrix_summary.json")
    rio.write_results("matrix_summary", summary["rows"][:-1], summary["meta"], work)  # regenerated input
    assert mr.main(["--dir", str(work), "--out", str(work / "t.md")]) == 2
    assert "stale" in capsys.readouterr().err


def test_missing_simulated_flag_is_not_called_measured(result_dir, tmp_path):
    import shutil
    work = tmp_path / "nometa"
    shutil.copytree(result_dir, work)
    data = rio.read_results(work / "analysis.json")
    data["meta"].pop("simulated")
    rio.write_results("analysis", data["rows"], data["meta"], work, write_csv=False)
    text = mr.render(work)
    assert "UNKNOWN" in text.split("## Matrix")[0] and "Measured with vLLM" not in text


def test_table_cells_are_escaped():
    lines = mr._table(["a"], [["x | y\nz"]])
    assert "x \\| y z" in lines[2]


def test_flags_use_the_stability_constants(result_dir):
    import stability
    text = mr.render(result_dir)
    assert f"{stability.CV_FLAG:.0%}" in text and str(stability.P95_MIN_REQUESTS) in text
