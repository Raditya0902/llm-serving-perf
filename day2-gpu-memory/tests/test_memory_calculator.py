from __future__ import annotations

import pytest

from memory_calculator import (
    PRESETS,
    DeploymentSpec,
    ModelSpec,
    gqa_savings,
    kv_bytes_per_token,
    kv_cache_bytes,
    main,
    max_concurrent_seqs,
    memory_growth,
    total_bytes,
    weight_bytes,
)


def _spec(preset: str = "llama3-8b", ctx: int = 1024, seqs: int = 1, kv_dtype: str = "fp16"):
    return DeploymentSpec(PRESETS[preset], context_len=ctx, concurrent_seqs=seqs, kv_dtype=kv_dtype)


def test_qwen_kv_bytes_per_token_matches_day1():
    assert kv_bytes_per_token(_spec("qwen2.5-0.5b")) == 12_288


def test_llama3_8b_kv_bytes_per_token_is_128_kib():
    assert kv_bytes_per_token(_spec("llama3-8b")) == 128 * 1024


def test_weight_bytes_scales_with_precision():
    def _bytes(dtype: str) -> int:
        return weight_bytes(DeploymentSpec(ModelSpec(8_000_000_000, dtype, 32, 8, 128), 1, 1))

    fp16 = _bytes("fp16")
    int4 = _bytes("int4")
    assert fp16 == 16_000_000_000
    assert int4 == 4_000_000_000


def test_kv_cache_is_linear_in_context_and_sequences():
    base = kv_cache_bytes(_spec(ctx=1000, seqs=2))
    assert kv_cache_bytes(_spec(ctx=2000, seqs=2)) == 2 * base
    assert kv_cache_bytes(_spec(ctx=1000, seqs=4)) == 2 * base


def test_kv_dtype_halves_cache():
    assert kv_cache_bytes(_spec(kv_dtype="fp8")) * 2 == kv_cache_bytes(_spec(kv_dtype="fp16"))


def test_total_is_weights_plus_kv():
    s = _spec(ctx=4096, seqs=8)
    assert total_bytes(s) == weight_bytes(s) + kv_cache_bytes(s)


def test_memory_growth_rows_are_monotonic():
    rows = memory_growth(_spec(seqs=4), [128, 1024, 8192])
    assert [r["context_len"] for r in rows] == [128, 1024, 8192]
    kv = [r["kv_bytes"] for r in rows]
    assert kv == sorted(kv) and kv[0] < kv[-1]
    assert all(r["total_bytes"] == weight_bytes(_spec()) + r["kv_bytes"] for r in rows)


def test_max_concurrent_seqs_fits_budget():
    s = _spec(ctx=8192)
    gpu = 80 * 1024**3
    n = max_concurrent_seqs(s, gpu, overhead_frac=0.1)
    per_seq = kv_cache_bytes(_spec(ctx=8192, seqs=1))
    budget = gpu * 0.9 - weight_bytes(s)
    assert n * per_seq <= budget < (n + 1) * per_seq


def test_max_concurrent_seqs_zero_when_weights_do_not_fit():
    assert max_concurrent_seqs(_spec(), 8 * 1024**3, overhead_frac=0.0) == 0


def test_gqa_savings_is_query_over_kv_heads():
    # Llama-3-8B: 32 query heads, 8 KV heads -> 4x smaller than full MHA.
    assert gqa_savings(_spec(), num_query_heads=32) == 4.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"context_len": 0},
        {"concurrent_seqs": 0},
        {"kv_dtype": "fp3"},
    ],
)
def test_invalid_deployment_raises(kwargs):
    base = {"model": PRESETS["llama3-8b"], "context_len": 1, "concurrent_seqs": 1, "kv_dtype": "fp16"}
    with pytest.raises(ValueError):
        DeploymentSpec(**{**base, **kwargs})


def test_invalid_model_raises():
    with pytest.raises(ValueError):
        ModelSpec(params=0, weight_dtype="fp16", num_layers=1, num_kv_heads=1, head_dim=1)
    with pytest.raises(ValueError):
        ModelSpec(params=1, weight_dtype="nope", num_layers=1, num_kv_heads=1, head_dim=1)


def test_cli_prints_summary(capsys):
    main(["--preset", "llama3-8b", "--context", "8192", "--seqs", "16", "--gpu-gb", "80"])
    out = capsys.readouterr().out
    assert "Weights" in out and "KV cache" in out and "Total" in out
    assert "max concurrent" in out.lower()


def test_cli_custom_model(capsys):
    main(["--params", "1e9", "--weight-dtype", "fp16", "--layers", "10", "--kv-heads", "2",
          "--head-dim", "64", "--context", "512", "--seqs", "2", "--kv-dtype", "fp16"])
    assert "KV cache" in capsys.readouterr().out


def test_cli_preset_honours_weight_dtype_override(capsys):
    main(["--preset", "llama3-8b", "--weight-dtype", "int4"])
    out = capsys.readouterr().out
    assert "(int4)" in out
    assert "Weights  : 3.74 GiB" in out


def test_cli_preset_defaults_to_preset_dtype(capsys):
    main(["--preset", "llama3-8b"])
    assert "(fp16)" in capsys.readouterr().out


def test_cli_zero_override_is_rejected_not_ignored():
    with pytest.raises(SystemExit):
        main(["--preset", "llama3-8b", "--layers", "0"])


def test_cli_missing_custom_fields_exits():
    with pytest.raises(SystemExit):
        main(["--params", "1e9"])


def test_cli_zero_gpu_reports_no_fit(capsys):
    main(["--preset", "qwen2.5-0.5b", "--gpu-gb", "0"])
    assert "DOES NOT FIT" in capsys.readouterr().out
