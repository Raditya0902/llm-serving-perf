import json

import pytest
import torch

from conftest import TINY_VOCAB
from run_experiments import (
    ExperimentConfig,
    kv_bytes_per_token,
    run_exp_a,
    run_exp_b,
    run_exp_c,
    write_results,
)

CFG = ExperimentConfig(
    warmup=0,
    repeats=2,
    a_prompt_lens=(4, 8),
    a_new_tokens=3,
    b_prompt_len=6,
    b_new_tokens=(2, 4),
    c_prompt_len=6,
    c_new_tokens=(3, 5),
)


def make_ids(n: int) -> torch.Tensor:
    return torch.arange(n).remainder(TINY_VOCAB).unsqueeze(0)


def test_kv_bytes_per_token_matches_formula(tiny_model):
    # 2 layers * 2 kv heads * head_dim 16 * (K+V) 2 * 4 bytes (fp32)
    assert kv_bytes_per_token(tiny_model) == 2 * 2 * 2 * 16 * 4


def test_exp_a_has_one_row_per_prompt_length(tiny_model):
    rows = run_exp_a(tiny_model, make_ids, (), CFG)

    assert [r["prompt_len"] for r in rows] == [4, 8]
    assert all(r["new_tokens"] == 3 and r["mode"] == "cached" for r in rows)
    assert all(r["prefill_ms"] > 0 for r in rows)


def test_exp_b_has_one_row_per_output_length(tiny_model):
    rows = run_exp_b(tiny_model, make_ids, (), CFG)

    assert [r["new_tokens"] for r in rows] == [2, 4]
    assert all(r["prompt_len"] == 6 for r in rows)


def test_exp_c_pairs_modes_and_tokens_match(tiny_model):
    rows = run_exp_c(tiny_model, make_ids, (), CFG)

    assert [(r["mode"], r["new_tokens"]) for r in rows] == [
        ("cached", 3),
        ("uncached", 3),
        ("cached", 5),
        ("uncached", 5),
    ]
    assert all(r["tokens_match_other_mode"] for r in rows)
    cached = rows[0]
    assert cached["analytic_kv_mb"] == pytest.approx(
        kv_bytes_per_token(tiny_model) * (6 + 3 - 1) / (1024 * 1024)
    )
    assert rows[1]["analytic_kv_mb"] == 0.0
    assert {"prefill_drift_fraction", "drift_warning", "peak_over_baseline_mb"} <= rows[0].keys()
    assert len(rows[0]["step_times_ms"]) == 2


def test_exp_c_flags_token_mismatch(tiny_model, monkeypatch):
    import run_experiments as mod
    from generate import generate_cached as real_cached

    def broken_cached(*args, **kwargs):
        result = real_cached(*args, **kwargs)
        return type(result)(
            token_ids=tuple(t + 1 for t in result.token_ids),
            prefill_s=result.prefill_s,
            step_times_s=result.step_times_s,
            peak_mem_bytes=result.peak_mem_bytes,
            stopped_on_eos=result.stopped_on_eos,
        )

    monkeypatch.setitem(mod.GENERATORS, "cached", broken_cached)

    rows = run_exp_c(tiny_model, make_ids, (), CFG)

    assert not any(r["tokens_match_other_mode"] for r in rows)


def test_collect_runs_interleaves_modes(tiny_model, monkeypatch):
    import run_experiments as mod

    order: list[str] = []
    for mode, real in list(mod.GENERATORS.items()):
        def traced(*args, _mode=mode, _real=real, **kwargs):
            order.append(_mode)
            return _real(*args, **kwargs)
        monkeypatch.setitem(mod.GENERATORS, mode, traced)
    cfg = ExperimentConfig(
        warmup=1, repeats=2, uncached_warmup=1,
        a_prompt_lens=(), a_new_tokens=1, b_prompt_len=1, b_new_tokens=(),
        c_prompt_len=1, c_new_tokens=(),
    )

    mod.collect_runs(tiny_model, make_ids(6), 2, (), ("cached", "uncached"), cfg)

    assert order == ["cached", "uncached", "cached", "uncached", "cached", "uncached"]


def test_write_results_creates_json_and_csv(tmp_path):
    rows = [{"mode": "cached", "total_s": 1.5}]

    write_results("exp_x", rows, {"device": "cpu"}, tmp_path)

    payload = json.loads((tmp_path / "exp_x.json").read_text())
    assert payload["rows"] == rows
    assert (tmp_path / "exp_x.csv").read_text().splitlines()[0] == "mode,total_s"
