from __future__ import annotations

import pytest

from helpers import make_result, write_result
from pathlib import Path

from make_report import DEFAULT_OUT, main, render_report
from parse_results import aggregate, load_dir, write_summary
from plot_results import load_summary


@pytest.fixture
def rows(tmp_path):
    raw = tmp_path / "raw"
    write_result(raw, make_result(output_throughput=250.0, p99_ttft_ms=80.0))
    write_result(raw, make_result("max_num_seqs-4", "max_num_seqs", "4", max_num_seqs=4,
                                  output_throughput=90.0, failed=2))
    write_result(raw, make_result("request_rate-inf", "request_rate", "inf",
                                  request_rate="inf", output_throughput=900.0))
    json_path, _ = write_summary(aggregate(load_dir(raw)), tmp_path)
    return load_summary(json_path)


def test_report_has_one_table_per_knob_with_baseline_marked(rows):
    text = render_report(rows, env_text=None)

    assert "## max_num_seqs" in text and "## request_rate" in text
    assert "## input_len" not in text  # no data -> no empty section
    assert "256 (baseline)" in text
    assert "inf" in text


def test_report_never_interprets_the_data(rows):
    text = render_report(rows, env_text=None)

    assert "Observations" in text
    assert "not interpreted" in text


def test_failed_requests_are_called_out(rows):
    text = render_report(rows, env_text=None)

    assert "failed" in text.lower()
    assert "max_num_seqs-4" in text.split("Warnings")[1]


def test_missing_preemption_data_renders_as_dash(rows):
    text = render_report(rows, env_text=None)

    assert "| – |" in text


def test_environment_block_is_included_verbatim(rows):
    text = render_report(rows, env_text="gpu: NVIDIA L4\nvllm: 0.34.0")

    assert "gpu: NVIDIA L4" in text
    assert "```" in text


def test_missing_environment_is_stated_not_invented(rows):
    assert "environment not recorded" in render_report(rows, env_text=None)


def test_single_repetition_is_flagged_as_having_no_variance(rows):
    assert "single repetition" in render_report(rows, env_text=None)


def test_empty_rows_are_rejected():
    with pytest.raises(ValueError, match="no rows"):
        render_report([], env_text=None)


def test_main_writes_report_and_reads_env_file(rows, tmp_path):
    raw = tmp_path / "raw"
    summary = tmp_path / "sweep_summary.json"
    env = tmp_path / "env.txt"
    env.write_text("gpu: test-gpu")
    out = tmp_path / "RESULTS.md"
    assert summary.exists() and raw.exists()

    code = main(["--summary", str(summary), "--env", str(env), "--out", str(out)])

    assert code == 0
    assert "gpu: test-gpu" in out.read_text()


def test_main_errors_on_missing_summary(tmp_path, capsys):
    code = main(["--summary", str(tmp_path / "nope.json"), "--out", str(tmp_path / "R.md")])

    assert code == 1
    assert "summary" in capsys.readouterr().err


# --- review fixes -------------------------------------------------------------------------
def test_non_finite_metrics_render_instead_of_crashing(rows):
    broken = [dict(r) for r in rows]
    broken[0]["output_throughput"] = float("nan")
    broken[1]["p99_ttft_ms"] = float("inf")

    text = render_report(broken, env_text=None)

    assert "nan" in text and "inf" in text


def test_default_output_does_not_clobber_the_top_level_stub():
    assert DEFAULT_OUT != Path("RESULTS.md")


def test_main_refuses_to_overwrite_a_hand_edited_report(tmp_path, capsys):
    summary = _write_summary(tmp_path)
    out = tmp_path / "RESULTS.md"
    out.write_text("# my hand-written conclusions\n")

    code = main(["--summary", str(summary), "--out", str(out)])

    assert code == 1
    assert out.read_text() == "# my hand-written conclusions\n"
    assert "--force" in capsys.readouterr().err


def test_main_overwrites_when_forced_or_when_report_is_untouched(tmp_path):
    summary = _write_summary(tmp_path)
    out = tmp_path / "RESULTS.md"
    out.write_text("# my notes\n")

    assert main(["--summary", str(summary), "--out", str(out), "--force"]) == 0
    assert "vLLM knob sweep: results" in out.read_text()
    assert main(["--summary", str(summary), "--out", str(out)]) == 0  # still the placeholder


def _write_summary(tmp_path):
    raw = tmp_path / "raw"
    write_result(raw, make_result())
    return write_summary(aggregate(load_dir(raw)), tmp_path)[0]
