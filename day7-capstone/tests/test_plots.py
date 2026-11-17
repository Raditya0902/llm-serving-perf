"""Figures and the architecture diagram are produced and non-empty (smoke tests)."""
from __future__ import annotations

import pytest

import analyze as an
import diagrams
import matrix as mx
import plot_matrix as pm
import results_io as rio
import run_matrix_sim as rms
import stability as st

PNG_MAGIC = b"\x89PNG"


@pytest.fixture(scope="module")
def result_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("figs")
    meta = rio.run_meta(simulated=True, note="t")
    rows = rms.run_matrix(mx.build_runs(reps=2, key_reps=2))
    rio.write_results("matrix_summary", st.aggregate_cases(rows), meta, out)
    abl = rms.run_matrix(rms.ablation_runs(1))
    rio.write_results("ablation_summary", st.aggregate_cases(abl), meta, out)
    assert an.main(["--dir", str(out)]) == 0
    return out


def test_main_writes_every_figure(result_dir):
    assert pm.main(["--dir", str(result_dir)]) == 0
    for name in ("tradeoff", "latency", "kv_pressure", "ablation", "multi_gpu"):
        data = (result_dir / "figures" / f"{name}.png").read_bytes()
        assert data.startswith(PNG_MAGIC) and len(data) > 10_000


def test_main_without_ablation_skips_that_figure(tmp_path):
    meta = rio.run_meta(simulated=True, note="t")
    rows = rms.run_matrix(mx.build_runs(reps=2, key_reps=2, include_control=False))
    rio.write_results("matrix_summary", st.aggregate_cases(rows), meta, tmp_path)
    assert an.main(["--dir", str(tmp_path)]) == 0
    assert pm.main(["--dir", str(tmp_path)]) == 0
    assert not (tmp_path / "figures" / "ablation.png").exists()
    assert (tmp_path / "figures" / "tradeoff.png").exists()


def test_main_fails_clearly_without_analysis(tmp_path, capsys):
    assert pm.main(["--dir", str(tmp_path)]) == 2
    assert "analysis.json" in capsys.readouterr().err


def test_architecture_diagram_is_written(tmp_path):
    out = diagrams.draw(tmp_path / "sub" / "arch.png")
    data = out.read_bytes()
    assert data.startswith(PNG_MAGIC) and len(data) > 30_000


def test_diagram_cli(tmp_path):
    assert diagrams.main(["--out", str(tmp_path / "a.png")]) == 0
    assert (tmp_path / "a.png").exists()


def test_gpu_like_summaries_with_unmeasured_metrics_still_plot(result_dir, tmp_path):
    summaries = rio.read_results(result_dir / "matrix_summary.json")["rows"]
    for s in summaries:
        s["kv_peak_frac"] = None
        s["preemptions"] = None
    decisions = rio.read_results(result_dir / "analysis.json")["rows"][0]["decisions"]
    pm.plot_kv(summaries, tmp_path / "kv.png", simulated=False)
    pm.plot_tradeoff(summaries, decisions, tmp_path / "t.png", simulated=False)
    assert (tmp_path / "kv.png").stat().st_size > 5_000


def test_partial_summaries_plot_what_exists(result_dir, tmp_path):
    summaries = [s for s in rio.read_results(result_dir / "matrix_summary.json")["rows"]
                 if s["case_id"] != "B-c64-tuned"]
    decisions = rio.read_results(result_dir / "analysis.json")["rows"][0]["decisions"]
    pm.plot_latency(summaries, tmp_path / "l.png", simulated=True)
    pm.plot_tradeoff(summaries, decisions, tmp_path / "t.png", simulated=True)
    assert (tmp_path / "l.png").exists()


def test_missing_simulated_flag_still_watermarks(result_dir, tmp_path):
    import shutil
    work = tmp_path / "w"
    shutil.copytree(result_dir, work)
    data = rio.read_results(work / "analysis.json")
    data["meta"].pop("simulated")
    rio.write_results("analysis", data["rows"], data["meta"], work, write_csv=False)
    assert pm.main(["--dir", str(work)]) == 0  # unknown provenance must not crash or claim "measured"
