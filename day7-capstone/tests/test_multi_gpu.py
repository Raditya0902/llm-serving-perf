"""Analytic tensor-parallel model: decode step time and KV capacity per TP degree."""
from __future__ import annotations

import pytest

import kv_sizing
import multi_gpu as mg


def test_tp1_is_the_identity():
    row = mg.tp_decode_step("llama3-8b", "a100-40gb", "nvlink", tp=1, batch_tokens=64)
    assert row["speedup"] == pytest.approx(1.0) and row["t_comm_ms"] == 0.0 and row["comm_fraction"] == 0.0


def test_two_allreduces_per_layer_and_their_bytes():
    row = mg.tp_decode_step("llama3-8b", "a100-40gb", "nvlink", tp=2, batch_tokens=64)
    assert row["n_allreduce"] == 2 * 32
    assert row["comm_bytes"] == 2 * 32 * 64 * 4096 * 2  # per layer: 2 x tokens x hidden x fp16


def test_decode_is_weight_read_bound_so_tp2_nearly_halves_compute_time():
    row = mg.tp_decode_step("llama3-8b", "a100-40gb", "nvlink", tp=2, batch_tokens=16)
    weight_ms = 16.06e9 / kv_sizing.HARDWARE["a100-40gb"].mem_bw_bytes_s * 1e3
    assert row["t_single_ms"] == pytest.approx(weight_ms, rel=0.05)
    assert 1.4 < row["speedup"] < 2.0


def test_slower_links_gain_less_and_a_network_link_can_lose():
    nv, pcie, net = (mg.tp_decode_step("llama3-8b", "a100-40gb", link, tp=2, batch_tokens=64)
                     for link in ("nvlink", "pcie", "network-100gbe"))
    assert nv["speedup"] > pcie["speedup"] > net["speedup"]
    assert net["speedup"] < 1.0


def test_comm_fraction_grows_with_tp_degree():
    fractions = [mg.tp_decode_step("llama3-8b", "a100-40gb", "pcie", tp=t, batch_tokens=16)["comm_fraction"]
                 for t in (2, 4, 8)]
    assert fractions == sorted(fractions)


def test_tp_unknown_link_and_bad_degree_are_rejected():
    with pytest.raises(ValueError, match="link"):
        mg.tp_decode_step("llama3-8b", "a100-40gb", "carrier-pigeon", tp=2, batch_tokens=1)
    with pytest.raises(ValueError, match="tp"):
        mg.tp_decode_step("llama3-8b", "a100-40gb", "nvlink", tp=0, batch_tokens=1)


def test_kv_capacity_more_than_doubles_when_weights_are_sharded():
    one = mg.kv_capacity("llama3-8b", "a100-40gb", 0.85, tp=1)
    two = mg.kv_capacity("llama3-8b", "a100-40gb", 0.85, tp=2)
    assert one["gain_vs_tp1"] == 1.0
    assert two["pool_tokens_total"] > 2 * one["pool_tokens_total"]
    assert two["gain_vs_tp1"] > 2.0


def test_tp1_capacity_matches_the_single_gpu_pool():
    one = mg.kv_capacity("llama3-8b", "a100-40gb", 0.85, tp=1)
    expected = kv_sizing.kv_pool_blocks(
        kv_sizing.model_spec("llama3-8b"), kv_sizing.HARDWARE["a100-40gb"], 0.85) * kv_sizing.BLOCK_SIZE
    assert one["pool_tokens_total"] == expected


def test_kv_head_count_not_divisible_by_tp_is_flagged_and_gets_no_sharding_bonus():
    ok = mg.kv_capacity("llama3-8b", "a100-40gb", 0.85, tp=2)
    bad = mg.kv_capacity("llama3-8b", "a100-40gb", 0.85, tp=3)
    assert ok["kv_sharded"] and not bad["kv_sharded"]


def test_workload_b_at_c64_fits_on_two_gpus_but_not_one():
    need = 64 * (2560 + 384 / 2)
    assert mg.kv_capacity("llama3-8b", "a100-40gb", 0.85, tp=1)["pool_tokens_total"] < need
    assert mg.kv_capacity("llama3-8b", "a100-40gb", 0.85, tp=2)["pool_tokens_total"] > need


def test_sweep_covers_every_combination_and_labels_assumptions():
    rows = mg.sweep("llama3-8b", "a100-40gb", util=0.85, tps=(1, 2), batches=(1, 64))
    assert len(rows) == len(mg.LINKS) * 2 * 2
    assert all(r["assumed"] is True for r in rows)
    assert {"link", "tp", "batch_tokens", "speedup", "pool_tokens_total", "gain_vs_tp1"} <= set(rows[0])


def test_weights_that_do_not_fit_give_zero_pool_and_no_gain_not_infinity():
    one = mg.kv_capacity("llama3-70b", "a100-40gb", 0.85, tp=1)
    assert one["pool_tokens_total"] == 0 and one["gain_vs_tp1"] is None
    eight = mg.kv_capacity("llama3-70b", "a100-40gb", 0.85, tp=8)
    assert eight["pool_tokens_total"] > 0 and eight["gain_vs_tp1"] is None


def test_sweep_is_json_safe_for_a_model_that_does_not_fit_one_gpu():
    import json
    rows = mg.sweep("llama3-70b", "a100-40gb", util=0.85, tps=(1, 8), batches=(1,))
    json.dumps(rows, allow_nan=False)


def test_tp_above_kv_head_count_replicates_heads_instead_of_losing_capacity():
    four = mg.kv_capacity("qwen2.5-7b", "a100-40gb", 0.85, tp=4)   # 4 kv heads: one per GPU
    eight = mg.kv_capacity("qwen2.5-7b", "a100-40gb", 0.85, tp=8)  # 2 GPUs share each head
    assert four["kv_sharded"] and eight["kv_replicated"]
    assert eight["pool_tokens_total"] > four["pool_tokens_total"]


def test_indivisible_tp_is_unsharded():
    out = mg.kv_capacity("llama3-8b", "a100-40gb", 0.85, tp=3)
    assert not out["kv_sharded"] and not out["kv_replicated"]
