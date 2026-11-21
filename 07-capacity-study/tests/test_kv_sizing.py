"""KV-pool sizing, hardware-derived cost model, and the analytic capacity check."""
from __future__ import annotations

import pytest

import kv_sizing as ks

GIB = 1024**3


def test_kv_bytes_per_token_llama3_8b_is_128_kib():
    # 2 (K,V) x 32 layers x 8 kv heads x 128 head_dim x 2 bytes
    assert ks.kv_bytes_per_token_for(ks.model_spec("llama3-8b")) == 2 * 32 * 8 * 128 * 2 == 131072


def test_qwen_7b_preset_uses_gqa_four_kv_heads():
    assert ks.kv_bytes_per_token_for(ks.model_spec("qwen2.5-7b")) == 2 * 28 * 4 * 128 * 2


def test_unknown_model_raises_with_valid_names():
    with pytest.raises(ValueError, match="llama3-8b"):
        ks.model_spec("gpt-nope")


def test_pool_blocks_matches_hand_formula():
    model, hw = ks.model_spec("llama3-8b"), ks.HARDWARE["a100-40gb"]
    expected_bytes = hw.gpu_bytes * 0.85 - 16.06e9 - ks.ACTIVATION_RESERVE_GIB * GIB
    block_bytes = ks.BLOCK_SIZE * 131072
    assert ks.kv_pool_blocks(model, hw, 0.85) == int(expected_bytes // block_bytes)


def test_pool_grows_with_utilization():
    model, hw = ks.model_spec("llama3-8b"), ks.HARDWARE["a100-40gb"]
    assert ks.kv_pool_blocks(model, hw, 0.92) > ks.kv_pool_blocks(model, hw, 0.85)


def test_pool_is_zero_when_weights_do_not_fit():
    model, hw = ks.model_spec("llama3-70b"), ks.HARDWARE["a10g-24gb"]
    assert ks.kv_pool_blocks(model, hw, 0.9) == 0


@pytest.mark.parametrize("util", [0.0, -0.1, 1.01])
def test_pool_rejects_bad_utilization(util):
    with pytest.raises(ValueError, match="gpu_memory_utilization"):
        ks.kv_pool_blocks(ks.model_spec("llama3-8b"), ks.HARDWARE["a100-40gb"], util)


def test_cost_model_is_derived_from_bytes_and_bandwidth():
    model, hw = ks.model_spec("llama3-8b"), ks.HARDWARE["a100-40gb"]
    cost = ks.cost_model_for(model, hw)
    assert cost.weight_read_ms == pytest.approx(16.06e9 / hw.mem_bw_bytes_s * 1e3)
    assert cost.kv_read_ms_per_token == pytest.approx(131072 / hw.mem_bw_bytes_s * 1e3)
    assert cost.compute_ms_per_token == pytest.approx(
        2 * model.params / (hw.peak_flops * hw.assumed_mfu) * 1e3
    )


def test_slower_gpu_gives_slower_decode_step():
    model = ks.model_spec("llama3-8b")
    fast = ks.cost_model_for(model, ks.HARDWARE["a100-40gb"])
    slow = ks.cost_model_for(model, ks.HARDWARE["a10g-24gb"])
    assert slow.iteration_time(0, 1, 100) > fast.iteration_time(0, 1, 100)


def test_capacity_pressure_flags_demand_above_pool():
    model, hw = ks.model_spec("llama3-8b"), ks.HARDWARE["a100-40gb"]
    pool_tokens = ks.kv_pool_blocks(model, hw, 0.85) * ks.BLOCK_SIZE
    low = ks.capacity_pressure(model, hw, 0.85, concurrency=4, mean_input=2560, mean_output=384)
    high = ks.capacity_pressure(model, hw, 0.85, concurrency=64, mean_input=2560, mean_output=384)
    assert low["pool_tokens"] == pool_tokens
    assert not low["pressured"] and high["pressured"]
    assert high["demand_ratio"] > 1.0 > low["demand_ratio"]


def test_chat_workload_is_never_pressured_on_a100():
    model, hw = ks.model_spec("llama3-8b"), ks.HARDWARE["a100-40gb"]
    row = ks.capacity_pressure(model, hw, 0.85, concurrency=64, mean_input=256, mean_output=128)
    assert not row["pressured"]
