"""analyze.py: summaries -> decisions, before/after pairs, bottleneck labels, stability flags."""
from __future__ import annotations

import pytest

import analyze as an
import matrix as mx
import results_io as rio
import run_matrix_sim as rms
import stability as st

pytestmark = pytest.mark.filterwarnings("ignore")


@pytest.fixture(scope="module")
def summaries():
    rows = rms.run_matrix(mx.build_runs(reps=2, key_reps=2))
    return st.aggregate_cases(rows)


def test_analysis_has_a_decision_per_workload(summaries):
    result = an.analyze(summaries)
    assert [d["workload"] for d in result["decisions"]] == ["A", "B"]
    assert all(d["threshold_ms"] > 0 for d in result["decisions"])


def test_pairs_cover_every_workload_and_concurrency(summaries):
    pairs = an.analyze(summaries)["pairs"]
    assert {(p["workload"], p["concurrency"]) for p in pairs} == {
        (w, c) for w in "AB" for c in mx.CONCURRENCY_LEVELS}
    assert all(r["metric"] for p in pairs for r in p["rows"])


def test_every_case_gets_a_bottleneck_label_with_a_reason(summaries):
    labels = an.analyze(summaries)["bottlenecks"]
    assert len(labels) == 12 and all(b["label"] and b["reason"] for b in labels)


def test_kv_pressure_case_is_labelled_kv_bound(summaries):
    labels = {b["case_id"]: b["label"] for b in an.analyze(summaries)["bottlenecks"]}
    assert labels["B-c64-baseline"].startswith("kv-capacity")
    assert not labels["A-c4-baseline"].startswith("kv-capacity")


def test_noise_floors_are_reported_per_metric(summaries):
    floors = an.analyze(summaries)["noise_floor_pct"]
    assert set(floors) >= {"output_tok_s", "p95_e2e_ms"} and all(v >= 0 for v in floors.values())


def test_analysis_records_the_preregistered_threshold_factor(summaries):
    assert an.analyze(summaries)["k_threshold"] == 3.0


def test_cli_writes_analysis_and_multi_gpu_files(tmp_path):
    rows = rms.run_matrix(mx.build_runs(reps=2, key_reps=2, include_control=False))
    rio.write_results("matrix", rows, rio.run_meta(simulated=True, note="t"), tmp_path)
    rio.write_results("matrix_summary", st.aggregate_cases(rows),
                      rio.run_meta(simulated=True, note="t"), tmp_path)
    assert an.main(["--dir", str(tmp_path)]) == 0
    data = rio.read_results(tmp_path / "analysis.json")
    assert data["meta"]["simulated"] is True and data["rows"][0]["decisions"]
    assert (tmp_path / "multi_gpu.json").exists() and (tmp_path / "multi_gpu.csv").exists()


def test_cli_fails_clearly_without_a_summary(tmp_path, capsys):
    assert an.main(["--dir", str(tmp_path)]) == 2
    assert "matrix_summary.json" in capsys.readouterr().err


def test_partial_matrix_is_analysed_without_crashing(summaries):
    partial = [s for s in summaries if s["case_id"] != "B-c64-tuned"]
    result = an.analyze(partial)
    assert ("B", 64) not in {(p["workload"], p["concurrency"]) for p in result["pairs"]}
    assert len(result["pairs"]) == 5


def test_single_workload_matrix_gets_one_decision(summaries):
    only_a = [s for s in summaries if s["workload"] == "A"]
    assert [d["workload"] for d in an.analyze(only_a)["decisions"]] == ["A"]


def test_analysis_records_the_digest_of_its_input(tmp_path):
    rows = rms.run_matrix(mx.build_runs(reps=2, key_reps=2, include_control=False))
    meta = rio.run_meta(simulated=True, note="t")
    rio.write_results("matrix_summary", st.aggregate_cases(rows), meta, tmp_path)
    assert an.main(["--dir", str(tmp_path)]) == 0
    recorded = rio.read_results(tmp_path / "analysis.json")["meta"]["matrix_summary_sha256"]
    assert recorded == rio.file_sha256(tmp_path / "matrix_summary.json")
