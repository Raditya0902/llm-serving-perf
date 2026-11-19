"""Per-step GPU cost analysis from vLLM torch-profiler traces (synthetic Chrome-trace events, no GPU)."""
from __future__ import annotations

import gzip
import json

import pytest

from gpu import step_costs as sc


def _ev(name, dur_us, ts=0, cat="gpu_user_annotation", ph="X"):
    return {"name": name, "dur": dur_us, "ts": ts, "cat": cat, "ph": ph}


def _step_name(prefill, decodes):
    return f"execute_context_{1 if prefill else 0}({prefill})_generation_{decodes}({decodes})"


def test_parse_steps_reads_prefill_tokens_decodes_and_duration_in_ms():
    steps = sc.parse_steps([_ev(_step_name(4096, 100), 80_000, ts=5), _ev("other", 1), _ev(_step_name(0, 256), 7_300, ts=9)])
    assert [(s.prefill_tokens, s.decodes, round(s.ms, 1)) for s in steps] == [(4096, 100, 80.0), (0, 256, 7.3)]


def test_parse_steps_prefers_gpu_annotations_over_cpu_ones_and_ignores_non_complete_events():
    events = [_ev(_step_name(0, 8), 9_000, cat="user_annotation"), _ev(_step_name(0, 8), 5_000),
              _ev(_step_name(0, 8), 5_000, ph="B")]
    steps = sc.parse_steps(events)
    assert len(steps) == 1 and steps[0].ms == pytest.approx(5.0)


def test_parse_steps_falls_back_to_cpu_annotations_when_no_gpu_ones_exist():
    steps = sc.parse_steps([_ev(_step_name(0, 8), 9_000, cat="user_annotation")])
    assert len(steps) == 1 and steps[0].ms == pytest.approx(9.0)


def test_summary_separates_pure_decode_from_mixed_steps_and_recovers_the_fit():
    events = []
    for i in range(12):  # exact model: ms = 5 + 9 * prefill/1000 + 0.1 * decodes
        prefill, decodes = 1000 * (1 + i % 6), 50 * (1 + i % 4)
        events.append(_ev(_step_name(prefill, decodes), (5 + 9 * prefill / 1000 + 0.1 * decodes) * 1000, ts=i))
    events += [_ev(_step_name(0, 256), 7_300, ts=100 + i) for i in range(5)]
    out = sc.summarize(sc.parse_steps(events))
    assert out["pure"]["n"] == 5 and out["pure"]["mean_ms"] == pytest.approx(7.3)
    assert out["mixed"]["n"] == 12
    fit = out["mixed"]["fit"]
    assert fit["intercept_ms"] == pytest.approx(5.0, abs=1e-6)
    assert fit["ms_per_1000_prefill_tokens"] == pytest.approx(9.0, abs=1e-6)
    assert fit["ms_per_decode"] == pytest.approx(0.1, abs=1e-9)


def test_fit_is_omitted_when_there_are_too_few_mixed_steps():
    out = sc.summarize(sc.parse_steps([_ev(_step_name(2000, 10), 30_000), _ev(_step_name(0, 10), 7_000)]))
    assert out["mixed"]["fit"] is None and out["mixed"]["n"] == 1


def test_summary_of_nothing_is_empty_not_an_error():
    out = sc.summarize([])
    assert out["steps"] == 0 and out["pure"]["n"] == 0 and out["mixed"]["n"] == 0


def test_rule_prediction_uses_the_day6_rule_of_thumb():
    assert sc.rule_of_thumb_ms(prefill_tokens=8192, decodes=256) == pytest.approx(9 * 8.192 + 0.1 * 256)


def test_load_events_reads_gzip_traces(tmp_path):
    path = tmp_path / "rank0.1.pt.trace.json.gz"
    with gzip.open(path, "wt") as fh:
        json.dump({"traceEvents": [_ev(_step_name(0, 4), 1000)]}, fh)
    assert len(sc.load_events(tmp_path)) == 1


def test_load_events_names_the_missing_trace(tmp_path):
    with pytest.raises(FileNotFoundError, match="rank0"):
        sc.load_events(tmp_path)
