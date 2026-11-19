"""Batch construction for the attention microbenchmark (pure Python; the kernel itself needs a GPU)."""
from __future__ import annotations

import json

import pytest

from gpu import attn_microbench as ab


def test_pure_decode_batch_has_one_query_token_per_sequence():
    spec = ab.pure_decode(n=4, ctx=100)
    assert spec.q_lens == (1, 1, 1, 1) and spec.kv_lens == (101,) * 4 and spec.label == "pure_decode"


def test_mixed_batch_puts_decodes_first_then_one_prefill_chunk_like_vllms_reordered_batch():
    spec = ab.mixed(n_decodes=3, ctx=50, prefill_tokens=7000, prefill_ctx=0)
    assert spec.q_lens == (1, 1, 1, 7000) and spec.kv_lens == (51, 51, 51, 7000)


def test_prefill_only_batch_and_chunk_with_existing_context():
    assert ab.prefill_only(prefill_tokens=512, prefill_ctx=1024).kv_lens == (1536,)


def test_cu_seqlens_is_the_exclusive_prefix_sum():
    assert ab.cu_seqlens((1, 1, 7000)) == [0, 1, 2, 7002]


def test_block_table_shape_pads_to_the_longest_sequence():
    assert ab.block_table_shape((17, 33, 1), block_size=16) == (3, 3)  # 2, 3 and 1 blocks, 3 rows


def test_describe_counts_tokens_and_the_kv_bytes_a_decode_must_read_per_layer():
    geometry = ab.GEOMETRIES["qwen2.5-0.5b"]
    info = ab.describe(ab.pure_decode(n=256, ctx=3583), geometry, block_size=16)
    assert info["q_tokens"] == 256 and info["max_seqlen_q"] == 1
    assert info["kv_bytes_read_per_layer"] == 256 * 3584 * 2 * 2 * 64 * 2  # seqs x ctx x (K,V) x kv heads x d x fp16
    assert info["gqa_group"] == 7
    assert info["bandwidth_floor_ms"] == pytest.approx(info["kv_bytes_read_per_layer"] / 0.696e12 * 1000)


def test_geometries_cover_the_two_models_and_a_no_gqa_control():
    assert ab.GEOMETRIES["llama3.1-8b"].gqa_group == 4
    assert ab.GEOMETRIES["qwen2.5-0.5b-mha"].gqa_group == 1 and ab.GEOMETRIES["qwen2.5-0.5b-mha"].q_heads == 14


def test_dry_run_prints_the_three_batches_without_touching_torch(capsys):
    code = ab.main(["--dry-run", "--model", "qwen2.5-0.5b", "--ctx", "3584", "--decodes", "256", "--prefill", "7600"])
    assert code == 0
    out = json.loads(capsys.readouterr().out)
    assert [b["label"] for b in out["batches"]] == ["pure_decode", "mixed", "prefill_only"]
    assert out["batches"][1]["q_tokens"] == 256 + 7600


def test_main_rejects_nonpositive_sizes():
    with pytest.raises(SystemExit):
        ab.main(["--dry-run", "--decodes", "0"])


def test_geometries_include_the_other_qwen_sizes_for_the_end_to_end_runs():
    assert ab.GEOMETRIES["qwen2.5-1.5b"] == ab.Geometry(12, 2, 128) and ab.GEOMETRIES["qwen2.5-1.5b"].gqa_group == 6
    assert ab.GEOMETRIES["qwen2.5-3b"].gqa_group == 8 and ab.GEOMETRIES["qwen2.5-3b"].head_dim == 128


def test_fa_version_flag_defaults_to_two_and_accepts_three():
    assert ab.parse_args([]).fa_version == 2
    assert ab.parse_args(["--fa-version", "3"]).fa_version == 3
    with pytest.raises(SystemExit):
        ab.parse_args(["--fa-version", "1"])
