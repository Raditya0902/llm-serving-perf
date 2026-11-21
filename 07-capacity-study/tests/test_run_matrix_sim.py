"""Results I/O and the simulated-matrix runner (CLI included)."""
from __future__ import annotations

import csv
import json

import pytest

import matrix as mx
import results_io as rio
import run_matrix_sim as rms


def test_write_and_read_roundtrip_keeps_none_as_null(tmp_path):
    rows = [{"case_id": "x", "peak_gpu_mem_gib": None, "output_tok_s": 1.5}]
    rio.write_results("m", rows, {"simulated": True}, tmp_path)
    data = rio.read_results(tmp_path / "m.json")
    assert data["meta"] == {"simulated": True} and data["rows"][0]["peak_gpu_mem_gib"] is None
    with (tmp_path / "m.csv").open() as fh:
        csv_rows = list(csv.DictReader(fh))
    assert csv_rows[0]["peak_gpu_mem_gib"] == rio.CSV_NOT_MEASURED


def test_write_results_unions_columns_across_rows(tmp_path):
    rio.write_results("m", [{"a": 1}, {"a": 2, "b": 3}], {}, tmp_path)
    header = (tmp_path / "m.csv").read_text().splitlines()[0]
    assert header == "a,b"


def test_read_results_rejects_files_without_meta_and_rows(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"rows": []}))
    with pytest.raises(ValueError, match="meta"):
        rio.read_results(bad)


def test_run_meta_has_provenance_fields():
    meta = rio.run_meta(simulated=True, note="n")
    assert meta["simulated"] is True and meta["note"] == "n"
    assert {"git_commit", "git_dirty", "platform", "python"} <= set(meta)


def test_run_matrix_executes_every_run_once():
    runs = mx.build_runs(reps=1, key_reps=1, include_control=False)
    rows = rms.run_matrix(runs)
    assert len(rows) == 12
    assert {r["case_id"] for r in rows} == {c.case_id for c in mx.build_matrix()}


def test_control_rows_are_flagged_and_share_config_id_with_the_original():
    rows = rms.run_matrix(mx.build_runs(reps=1, key_reps=1))
    control = next(r for r in rows if r["is_control"])
    original = next(r for r in rows if r["case_id"] == "B-c16-baseline")
    assert control["config_id"] == original["config_id"] and control["case_id"].endswith("-aa")


def test_ablation_variants_change_one_knob_at_a_time():
    variants = {v.name: v for v in rms.ablation_configs()}
    base = mx.CONFIGS["baseline"]
    assert variants["util-only"].gpu_memory_utilization == mx.CONFIGS["tuned"].gpu_memory_utilization
    assert variants["util-only"].max_num_batched_tokens == base.max_num_batched_tokens
    assert variants["tokens-only"].max_num_batched_tokens == mx.CONFIGS["tuned"].max_num_batched_tokens
    assert variants["tokens-only"].gpu_memory_utilization == base.gpu_memory_utilization
    assert {"baseline", "tuned"} <= set(variants)


def test_admission_cap_variants_only_change_max_num_seqs():
    caps = [v for v in rms.ablation_configs() if v.name.startswith("cap")]
    assert caps and all(v.max_num_seqs < mx.CONFIGS["baseline"].max_num_seqs for v in caps)
    base = mx.CONFIGS["baseline"]
    assert all((v.gpu_memory_utilization, v.max_num_batched_tokens) ==
               (base.gpu_memory_utilization, base.max_num_batched_tokens) for v in caps)


def test_cli_smoke_writes_matrix_summary_and_labels_simulated(tmp_path):
    assert rms.main(["--smoke", "--out", str(tmp_path)]) == 0
    raw = rio.read_results(tmp_path / "matrix.json")
    summary = rio.read_results(tmp_path / "matrix_summary.json")
    assert raw["meta"]["simulated"] is True and raw["meta"]["smoke"] is True
    assert len(raw["rows"]) == 12 and len(summary["rows"]) == 12
    assert (tmp_path / "matrix.csv").exists() and (tmp_path / "matrix_summary.csv").exists()


def test_cli_ablation_writes_its_own_files(tmp_path):
    assert rms.main(["--smoke", "--ablation", "--out", str(tmp_path)]) == 0
    ablation = rio.read_results(tmp_path / "ablation.json")
    assert {r["config"] for r in ablation["rows"]} >= {"baseline", "tuned", "util-only", "tokens-only"}


def test_cli_rejects_nonsense_reps(tmp_path):
    with pytest.raises(SystemExit):
        rms.main(["--reps", "0", "--out", str(tmp_path)])


# ---------------------------------------------------------------- results_io hardening
@pytest.mark.parametrize("cell", ["=1+1", "+cmd", "-2+3", "@SUM(A1)", "\tx", "\rx"])
def test_csv_cells_that_look_like_formulas_are_neutralised(tmp_path, cell):
    rio.write_results("m", [{"case_id": cell, "n": -5, "x": 1.5}], {}, tmp_path)
    with (tmp_path / "m.csv").open(newline="") as fh:
        row = next(csv.DictReader(fh))
    assert row["case_id"] == "'" + cell
    assert row["n"] == "-5"  # real numbers are never altered


def test_empty_rows_remove_a_stale_csv(tmp_path):
    rio.write_results("m", [{"a": 1}], {}, tmp_path)
    assert (tmp_path / "m.csv").exists()
    rio.write_results("m", [], {}, tmp_path)
    assert not (tmp_path / "m.csv").exists() and (tmp_path / "m.json").exists()


def test_failed_write_leaves_the_previous_file_intact(tmp_path):
    rio.write_results("m", [{"a": 1}], {"v": 1}, tmp_path)
    with pytest.raises(ValueError):
        rio.write_results("m", [{"a": float("nan")}], {"v": 2}, tmp_path)
    assert rio.read_results(tmp_path / "m.json")["meta"] == {"v": 1}
    assert not list(tmp_path.glob("*.tmp"))


def test_read_results_rejects_oversized_and_symlinked_files(tmp_path, monkeypatch):
    rio.write_results("m", [{"a": 1}], {}, tmp_path)
    monkeypatch.setattr(rio, "MAX_RESULT_BYTES", 5)
    with pytest.raises(ValueError, match="too large"):
        rio.read_results(tmp_path / "m.json")
    monkeypatch.undo()
    link = tmp_path / "link.json"
    link.symlink_to(tmp_path / "m.json")
    with pytest.raises(ValueError, match="regular file"):
        rio.read_results(link)


def test_analysis_freshness_detects_a_changed_summary(tmp_path):
    rio.write_results("matrix_summary", [{"a": 1}], {}, tmp_path)
    digest = rio.file_sha256(tmp_path / "matrix_summary.json")
    rio.assert_fresh(tmp_path / "matrix_summary.json", digest)
    rio.write_results("matrix_summary", [{"a": 2}], {}, tmp_path)
    with pytest.raises(ValueError, match="stale"):
        rio.assert_fresh(tmp_path / "matrix_summary.json", digest)
