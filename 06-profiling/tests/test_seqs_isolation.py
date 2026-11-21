"""Summary of the max_num_seqs isolation grid (synthetic bench JSONs)."""
from __future__ import annotations

import json

import pytest

from gpu import seqs_isolation as si


def _write(dirpath, seqs, workload, rep, tok_s, p50=10.0, p99=100.0):
    path = dirpath / f"seqs{seqs}_{workload}_r{rep}.json"
    path.write_text(json.dumps({"output_throughput": tok_s, "median_itl_ms": p50, "p99_itl_ms": p99, "failed": 0,
                                "completed": 1000}))


def test_parse_name_splits_seqs_workload_and_repeat():
    assert si.parse_name("seqs1024_512x128_r2.json") == (1024, "512x128", 2)
    with pytest.raises(ValueError):
        si.parse_name("bench_seqs1024.log")


def test_summarize_reports_loss_against_the_256_arm_of_the_same_workload(tmp_path):
    for rep, (a, b) in enumerate([(9400, 8500), (9300, 8450), (9500, 8600)]):
        _write(tmp_path, 256, "512x128", rep, a)
        _write(tmp_path, 1024, "512x128", rep, b)
        _write(tmp_path, 256, "512x512", rep, 6000)
        _write(tmp_path, 1024, "512x512", rep, 5900)
    rows = {(r["seqs"], r["workload"]): r for r in si.summarize(tmp_path)}
    base = rows[(1024, "512x128")]
    assert base["mean_tok_s"] == pytest.approx(8516.67, abs=0.01) and base["n"] == 3
    assert base["loss_pct_vs_256"] == pytest.approx((8516.67 / 9400 - 1) * 100, abs=0.01)
    assert rows[(1024, "512x512")]["loss_pct_vs_256"] == pytest.approx((5900 / 6000 - 1) * 100, abs=0.01)
    assert rows[(256, "512x128")]["loss_pct_vs_256"] == pytest.approx(0.0)


def test_summarize_skips_runs_with_failed_requests_and_says_so(tmp_path):
    _write(tmp_path, 256, "512x128", 0, 9400)
    bad = tmp_path / "seqs1024_512x128_r0.json"
    bad.write_text(json.dumps({"output_throughput": 1.0, "failed": 3, "completed": 997, "median_itl_ms": 1, "p99_itl_ms": 1}))
    rows = si.summarize(tmp_path)
    assert [(r["seqs"], r["n"]) for r in rows] == [(256, 1)]
    assert si.rejected(tmp_path) == ["seqs1024_512x128_r0.json"]


def test_verdict_follows_the_preregistered_rule():
    # H1: loss at 512 out and at 128 in is at least 5 points smaller than at baseline, busy fractions within 5 points
    losses = {"512x128": -11.0, "128x128": -4.0, "512x512": -3.0, "128x512": -1.0}
    assert si.verdict(losses, busy_256=0.93, busy_1024=0.91)["supported"] == "H1"
    # H2: loss persists across workloads (within 3 points) -> per-step overhead
    flat = {"512x128": -11.0, "128x128": -10.0, "512x512": -12.0, "128x512": -11.5}
    assert si.verdict(flat, busy_256=0.93, busy_1024=0.93)["supported"] == "H2"
    # H2 also when the 1024 arm has at least 10 points more idle time
    assert si.verdict(losses, busy_256=0.95, busy_1024=0.80)["supported"] == "H2"
    assert si.verdict({"512x128": -11.0, "128x128": -7.0, "512x512": -9.0, "128x512": -5.0}, busy_256=0.9, busy_1024=0.88)["supported"] == "mixed"


def test_verdict_without_busy_fractions_says_what_is_missing():
    assert si.verdict({"512x128": -11.0, "512x512": -2.0, "128x512": -1.0}, busy_256=None, busy_1024=None)["supported"] == "undetermined"
