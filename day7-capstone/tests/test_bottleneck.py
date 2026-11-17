"""Rule-based bottleneck classification (supports closed-book Q9 and the report's 'what remains')."""
from __future__ import annotations

import bottleneck as bn

WEIGHT_READ_MS = 10.0


def _summary(**means):
    base = {"kv_peak_frac": 0.2, "preemptions": 0.0, "mean_batch": 10.0, "mean_tpot_ms": 11.0}
    base.update(means)
    return {k: {"mean": v} for k, v in base.items()}


def test_full_kv_pool_with_preemptions_is_kv_capacity_bound():
    out = bn.classify(_summary(kv_peak_frac=1.0, preemptions=40), weight_read_ms=WEIGHT_READ_MS, max_num_seqs=256)
    assert out["label"] == bn.KV_CAPACITY and "preempt" in out["reason"]


def test_full_pool_without_preemption_is_not_called_kv_bound():
    out = bn.classify(_summary(kv_peak_frac=1.0, preemptions=0), weight_read_ms=WEIGHT_READ_MS, max_num_seqs=256)
    assert out["label"] != bn.KV_CAPACITY


def test_batch_pinned_at_the_sequence_cap_is_scheduler_bound():
    out = bn.classify(_summary(mean_batch=60.0), weight_read_ms=WEIGHT_READ_MS, max_num_seqs=64)
    assert out["label"] == bn.SCHEDULER_CAP


def test_decode_step_near_the_weight_read_floor_is_bandwidth_bound():
    out = bn.classify(_summary(mean_tpot_ms=12.0), weight_read_ms=WEIGHT_READ_MS, max_num_seqs=256)
    assert out["label"] == bn.BANDWIDTH_DECODE


def test_decode_step_far_above_the_floor_is_compute_or_interference_bound():
    out = bn.classify(_summary(mean_tpot_ms=30.0), weight_read_ms=WEIGHT_READ_MS, max_num_seqs=256)
    assert out["label"] == bn.COMPUTE_OR_INTERFERENCE


def test_kv_rule_wins_over_the_others():
    out = bn.classify(_summary(kv_peak_frac=1.0, preemptions=3, mean_batch=250.0),
                      weight_read_ms=WEIGHT_READ_MS, max_num_seqs=256)
    assert out["label"] == bn.KV_CAPACITY


def test_missing_metrics_give_unknown_not_a_guess():
    out = bn.classify({"kv_peak_frac": None, "preemptions": None, "mean_batch": None, "mean_tpot_ms": None},
                      weight_read_ms=WEIGHT_READ_MS, max_num_seqs=256)
    assert out["label"] == bn.UNKNOWN


def test_reason_quotes_the_numbers_it_used():
    out = bn.classify(_summary(mean_tpot_ms=12.0), weight_read_ms=WEIGHT_READ_MS, max_num_seqs=256)
    assert "12.0" in out["reason"] and "10.0" in out["reason"]


def test_preemptions_with_a_sampled_gauge_below_full_still_mean_kv_bound():
    out = bn.classify(_summary(kv_peak_frac=0.93, preemptions=12), weight_read_ms=WEIGHT_READ_MS, max_num_seqs=256)
    assert out["label"] == bn.KV_CAPACITY


def test_preemptions_without_a_kv_gauge_still_mean_kv_bound_with_a_caveat():
    s = _summary(preemptions=7)
    s["kv_peak_frac"] = None
    out = bn.classify(s, weight_read_ms=WEIGHT_READ_MS, max_num_seqs=256)
    assert out["label"] == bn.KV_CAPACITY and "gauge" in out["caveat"]


def test_unmeasured_kv_pressure_is_called_out_when_another_label_wins():
    s = _summary(mean_tpot_ms=12.0)
    s["kv_peak_frac"] = s["preemptions"] = None
    out = bn.classify(s, weight_read_ms=WEIGHT_READ_MS, max_num_seqs=256)
    assert out["label"] == bn.BANDWIDTH_DECODE and "KV" in out["caveat"]


def test_unmeasured_batch_size_is_called_out():
    s = _summary(mean_tpot_ms=12.0)
    s["mean_batch"] = None
    assert "batch" in bn.classify(s, weight_read_ms=WEIGHT_READ_MS, max_num_seqs=256)["caveat"]


def test_no_caveat_when_everything_was_measured():
    assert bn.classify(_summary(), weight_read_ms=WEIGHT_READ_MS, max_num_seqs=256)["caveat"] is None
