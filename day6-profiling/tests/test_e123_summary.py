"""E1-E3 hypotheses evaluated on vLLM bench results (synthetic JSONs)."""
from __future__ import annotations

import json

import pytest

from gpu import e123_summary as e


def _w(d, name, **keys):
    d.mkdir(parents=True, exist_ok=True)
    (d / name).write_text(json.dumps({"failed": 0, "completed": 10} | keys))


def _model(root, name, tok, tpot, ttft, tpot_by_out):
    d = root / name
    for c in (1, 2, 4, 8):
        _w(d, f"e1_c{c}.json", output_throughput=tok[c], median_tpot_ms=tpot[c])
    for n, v in ttft.items():
        _w(d, f"e2_in{n}.json", median_ttft_ms=v)
    for n, v in tpot_by_out.items():
        _w(d, f"e3_out{n}.json", median_tpot_ms=v)
    return d


def test_all_three_hypotheses_confirmed_on_a_weight_bound_model(tmp_path):
    _model(tmp_path, "m", {1: 40, 2: 80, 4: 158, 8: 310}, {1: 25, 2: 25, 4: 25.3, 8: 25.8},
           {128: 20, 512: 60, 1024: 120, 2048: 260}, {32: 25.0, 128: 25.2, 256: 25.4})
    out = e.summarize(tmp_path)["m"]
    assert out["E1"]["confirmed"] and out["E2"]["confirmed"] and out["E3"]["confirmed"]
    assert out["E1"]["throughput_ratio"] == pytest.approx(7.75) and out["E2"]["ratio"] == pytest.approx(260 / 60)


def test_e2_is_refuted_when_a_per_step_floor_hides_the_prefill_growth(tmp_path):
    _model(tmp_path, "m", {1: 40, 2: 80, 4: 158, 8: 310}, {1: 25, 2: 25, 4: 25, 8: 25},
           {128: 9, 512: 10, 1024: 12, 2048: 17}, {32: 25, 128: 25, 256: 25})
    out = e.summarize(tmp_path)["m"]
    assert out["E2"]["confirmed"] is False and out["E2"]["ratio"] == pytest.approx(1.7)


def test_e1_needs_both_the_step_time_and_the_throughput_condition(tmp_path):
    _model(tmp_path, "m", {1: 40, 2: 70, 4: 100, 8: 150}, {1: 25, 2: 28, 4: 40, 8: 53},
           {512: 60, 2048: 260}, {32: 25, 256: 25})
    out = e.summarize(tmp_path)["m"]["E1"]
    assert out["confirmed"] is False and out["step_time_ratio"] > 1.5 and out["throughput_ratio"] < 5


def test_e3_is_refuted_beyond_five_percent(tmp_path):
    _model(tmp_path, "m", {1: 40, 2: 80, 4: 158, 8: 310}, {1: 25, 2: 25, 4: 25, 8: 25}, {512: 60, 2048: 260}, {32: 25.0, 256: 27.0})
    assert e.summarize(tmp_path)["m"]["E3"]["confirmed"] is False


def test_missing_points_are_reported_instead_of_guessed(tmp_path):
    _w(tmp_path / "m", "e1_c1.json", output_throughput=40, median_tpot_ms=25)
    out = e.summarize(tmp_path)["m"]
    assert out["E1"]["confirmed"] is None and "e1_c8" in out["E1"]["missing"]


def test_failed_runs_count_as_missing(tmp_path):
    _model(tmp_path, "m", {1: 40, 2: 80, 4: 158, 8: 310}, {1: 25, 2: 25, 4: 25, 8: 25}, {512: 60, 2048: 260}, {32: 25, 256: 25})
    _w(tmp_path / "m", "e2_in2048.json", median_ttft_ms=260, failed=3)
    out = e.summarize(tmp_path)["m"]["E2"]
    assert out["confirmed"] is None and "e2_in2048" in out["missing"]
