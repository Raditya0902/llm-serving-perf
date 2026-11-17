from __future__ import annotations

import csv
import json
import sys

import pytest
import torch

import run_experiments as rx
from memory_calculator import PRESETS, DeploymentSpec, kv_cache_bytes

BYTES_PER_MB = rx.BYTES_PER_MB


def test_exp_a_smoke_rows_match_theory():
    rows = rx.run_exp_a("cpu", rx.SMOKE)
    assert len(rows) == len(rx.SMOKE.a_contexts) * len(rx.SMOKE.a_seqs)
    for r in rows:
        spec = DeploymentSpec(PRESETS[rx.EXP_A_MODEL], r["context_len"], r["concurrent_seqs"], "fp16")
        assert r["theoretical_kv_mb"] == pytest.approx(kv_cache_bytes(spec) / BYTES_PER_MB, abs=1e-3)
        assert not r["skipped"]
        assert r["num_tensors"] == 2 * PRESETS[rx.EXP_A_MODEL].num_layers
        assert r["measured_kv_mb"] is not None


def test_exp_a_skips_points_over_budget():
    cfg = rx.ExperimentConfig(**{**rx.SMOKE.__dict__, "a_max_kv_bytes": 1})
    rows = rx.run_exp_a("cpu", cfg)
    assert all(r["skipped"] for r in rows)
    assert all("measured_kv_mb" not in r for r in rows)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS for exact counters")
def test_exp_a_mps_measured_close_to_theory():
    cfg = rx.ExperimentConfig(**{**rx.SMOKE.__dict__, "a_contexts": (256,), "a_seqs": (2,)})
    (row,) = rx.run_exp_a("mps", cfg)
    assert row["measured_over_theory"] == pytest.approx(1.0, abs=0.02)


def test_exp_b_smoke_has_all_stages_in_order():
    rows = rx.run_exp_b("cpu", rx.SMOKE)
    stages = [r["stage"] for r in rows]
    assert stages[0] == "0 baseline" and stages[-1] == "10 cleanup"
    assert len(stages) == 11
    assert all(r["reserved_is_meaningful"] is False for r in rows)
    peaks = [r["peak_allocated_mb"] for r in rows[:7]]
    assert peaks == sorted(peaks)  # peak never decreases between resets


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS")
def test_exp_b_mps_allocated_tracks_live_tensors_and_peak_survives_free():
    rows = {r["stage"]: r for r in rx.run_exp_b("mps", rx.SMOKE)}
    size = rx.SMOKE.b_tensor_mb
    assert rows["1 alloc A"]["allocated_mb"] - rows["0 baseline"]["allocated_mb"] >= size * 0.99
    assert rows["3 free A"]["allocated_mb"] < rows["2 alloc B"]["allocated_mb"]
    assert rows["5 free all"]["allocated_mb"] < rows["5 free all"]["peak_allocated_mb"]


def test_exp_c_smoke_shapes_and_paged_beats_contiguous():
    rows = rx.run_exp_c(rx.SMOKE)
    assert [r["allocator"] for r in rows] == ["contiguous-max", "paged-bs8", "paged-bs32"]
    contig, *paged = rows
    assert all(r["pool_tokens"] == rx.SMOKE.c_pool_tokens for r in rows)
    assert all(p["unused_reserved_fraction"] < contig["unused_reserved_fraction"] for p in paged)
    assert all(p["completed"] == contig["completed"] == rx.SMOKE.c_workload.num_requests for p in paged)


def test_exp_c_larger_blocks_fragment_more():
    rows = {r["allocator"]: r for r in rx.run_exp_c(rx.SMOKE)}
    assert rows["paged-bs8"]["mean_unused_reserved_slots"] <= rows["paged-bs32"]["mean_unused_reserved_slots"]


def test_exp_c_is_deterministic_and_seed_changes_it():
    assert rx.run_exp_c(rx.SMOKE) == rx.run_exp_c(rx.SMOKE)
    assert rx.run_exp_c(rx.SMOKE, seed=1) != rx.run_exp_c(rx.SMOKE, seed=2)


def test_write_results_json_and_csv(tmp_path):
    rows = [{"a": 1, "b": 2.5}, {"a": 3, "b": 4.5, "c": "x"}]
    rx.write_results("demo", rows, {"k": "v"}, tmp_path)
    data = json.loads((tmp_path / "demo.json").read_text())
    assert data["meta"] == {"k": "v"} and data["rows"] == rows
    with (tmp_path / "demo.csv").open() as fh:
        assert [r["a"] for r in csv.DictReader(fh)] == ["1", "3"]


def test_write_results_empty_rows_writes_only_json(tmp_path):
    rx.write_results("empty", [], {}, tmp_path)
    assert (tmp_path / "empty.json").exists() and not (tmp_path / "empty.csv").exists()


def test_run_metadata_has_day1_kv_bytes():
    meta = rx.run_metadata("cpu", rx.SMOKE, True)
    assert meta["kv_bytes_per_token"] == 12_288 and meta["smoke"] is True


def test_cli_smoke_all_writes_files(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["run_experiments.py", "--smoke", "--device", "cpu", "--out", str(tmp_path)])
    rx._main()
    names = {p.name for p in tmp_path.iterdir()}
    for exp in "abc":
        assert f"exp_{exp}_smoke.json" in names and f"exp_{exp}_smoke.csv" in names
    assert "paged-bs8" in capsys.readouterr().out
