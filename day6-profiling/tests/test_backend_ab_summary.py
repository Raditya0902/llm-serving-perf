"""Summary of backend A/B runs laid out as <root>/<model>/s<seqs>/<BACKEND>_<in>x<out>_r<rep>.json (synthetic bench JSONs)."""
from __future__ import annotations

import json

import pytest

from gpu import backend_ab_summary as bs


def _write(root, model, seqs, backend_tag, workload, rep, tok_s, itl=10.0, failed=0):
    d = root / model / f"s{seqs}"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{backend_tag}_{workload}_r{rep}.json").write_text(
        json.dumps({"output_throughput": tok_s, "median_itl_ms": itl, "p99_itl_ms": itl * 5, "failed": failed, "completed": 1000}))


def test_parse_name_handles_plain_and_fa_version_backend_tags():
    assert bs.parse_name("FLASH_ATTN_512x128_r1.json") == ("FLASH_ATTN", "512x128", 1)
    assert bs.parse_name("FLASH_ATTN-fa3_3584x128_r0.json") == ("FLASH_ATTN-fa3", "3584x128", 0)
    assert bs.parse_name("TRITON_ATTN_128x128_r12.json") == ("TRITON_ATTN", "128x128", 12)
    with pytest.raises(ValueError):
        bs.parse_name("bench_FLASH_ATTN.log")


def test_gather_reads_model_and_seqs_from_the_directory_layout(tmp_path):
    _write(tmp_path, "qwen2.5-3b", 256, "FLASH_ATTN", "512x128", 0, 5000)
    _write(tmp_path, "qwen2.5-3b", 1024, "FLASH_ATTN", "512x128", 0, 4800)
    cells, rejected = bs.gather(tmp_path)
    assert set(cells) == {("qwen2.5-3b", 256, "512x128", "FLASH_ATTN"), ("qwen2.5-3b", 1024, "512x128", "FLASH_ATTN")} and not rejected


def test_runs_with_failed_requests_are_rejected_and_named(tmp_path):
    _write(tmp_path, "m", 256, "FLASH_ATTN", "512x128", 0, 5000)
    _write(tmp_path, "m", 256, "FLASH_ATTN", "512x128", 1, 1.0, failed=4)
    cells, rejected = bs.gather(tmp_path)
    assert len(cells[("m", 256, "512x128", "FLASH_ATTN")]) == 1 and rejected == ["m/s256/FLASH_ATTN_512x128_r1.json"]


def test_summary_reports_gain_over_the_base_backend_and_the_seqs_loss(tmp_path):
    for rep, (fa, tr, fa_hi, tr_hi) in enumerate([(9000, 10000, 8200, 10050), (9200, 10200, 8400, 10150)]):
        _write(tmp_path, "m", 256, "FLASH_ATTN", "512x128", rep, fa)
        _write(tmp_path, "m", 256, "TRITON_ATTN", "512x128", rep, tr)
        _write(tmp_path, "m", 1024, "FLASH_ATTN", "512x128", rep, fa_hi)
        _write(tmp_path, "m", 1024, "TRITON_ATTN", "512x128", rep, tr_hi)
    out = bs.summarize(tmp_path)
    gain = {(g["seqs"], g["backend"]): g["gain_pct_vs_base"] for g in out["gains"]}
    assert gain[(256, "TRITON_ATTN")] == pytest.approx((10100 / 9100 - 1) * 100)
    assert gain[(1024, "TRITON_ATTN")] == pytest.approx((10100 / 8300 - 1) * 100)
    loss = {l["backend"]: l["loss_pct"] for l in out["seqs_loss"]}
    assert loss["FLASH_ATTN"] == pytest.approx((8300 / 9100 - 1) * 100)
    assert loss["TRITON_ATTN"] == pytest.approx(0.0, abs=0.01)


def test_cells_carry_mean_spread_and_median_itl(tmp_path):
    for rep, v in enumerate([100.0, 104.0, 102.0]):
        _write(tmp_path, "m", 256, "FLASH_ATTN", "128x128", rep, v, itl=7.0 + rep)
    row = bs.summarize(tmp_path)["cells"][0]
    assert row["n"] == 3 and row["mean_tok_s"] == pytest.approx(102.0)
    assert row["spread_pct"] == pytest.approx(4.0) and row["median_itl_ms"] == pytest.approx(8.0)


def test_cli_prints_json(tmp_path, capsys):
    _write(tmp_path, "m", 256, "FLASH_ATTN", "512x128", 0, 5000)
    assert bs.main([str(tmp_path)]) == 0
    assert json.loads(capsys.readouterr().out)["cells"][0]["workload"] == "512x128"


def test_gain_can_be_measured_against_another_base_backend(tmp_path):
    _write(tmp_path, "m", 256, "TRITON_ATTN", "512x128", 0, 10000)
    _write(tmp_path, "m", 256, "FLASH_ATTN-fa3", "512x128", 0, 10500)
    out = bs.summarize(tmp_path, base_backend="TRITON_ATTN")
    assert out["gains"][0]["backend"] == "FLASH_ATTN-fa3" and out["gains"][0]["gain_pct_vs_base"] == pytest.approx(5.0)


def test_cli_takes_a_base_flag(tmp_path, capsys):
    _write(tmp_path, "m", 256, "TRITON_ATTN", "512x128", 0, 10000)
    _write(tmp_path, "m", 256, "FLASH_ATTN-fa2", "512x128", 0, 9000)
    assert bs.main([str(tmp_path), "--base", "TRITON_ATTN"]) == 0
    assert json.loads(capsys.readouterr().out)["gains"][0]["gain_pct_vs_base"] == pytest.approx(-10.0)
