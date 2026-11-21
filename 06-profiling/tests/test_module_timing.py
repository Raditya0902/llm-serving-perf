import pytest
import torch

from module_timing import (
    COARSE_LABELS,
    attribution,
    coarse_selector,
    linear_selector,
    replay_linears,
    sync_bracket_overhead_us,
    time_modules,
)

CPU = torch.device("cpu")
EOS = (1, 2)


def test_coarse_selector_labels_attention_mlp_norms_and_head(tiny_model):
    labels = {name: coarse_selector(name, m) for name, m in tiny_model.named_modules()}

    assert labels["model.layers.0.self_attn"] == "attn"
    assert labels["model.layers.1.mlp"] == "mlp"
    assert labels["model.layers.0.input_layernorm"] == "norm"
    assert labels["model.norm"] == "norm"
    assert labels["lm_head"] == "lm_head"
    assert labels["model.embed_tokens"] == "embed"
    # nested modules are never labelled, so ranges cannot double count
    assert labels["model.layers.0.self_attn.q_proj"] is None
    assert set(filter(None, labels.values())) <= set(COARSE_LABELS)


def test_linear_selector_groups_linear_leaves_by_role(tiny_model):
    labels = {n: linear_selector(n, m) for n, m in tiny_model.named_modules() if linear_selector(n, m)}
    layers = len(tiny_model.model.layers)

    assert sum(v == "proj" for v in labels.values()) == 4 * layers
    assert sum(v == "mlp_linear" for v in labels.values()) == 3 * layers
    assert labels["lm_head"] == "head_linear"
    assert len(labels) == 7 * layers + 1


def test_time_modules_decode_reports_per_step_ms_and_removes_hooks(tiny_model, prompt_ids):
    before = sum(len(m._forward_hooks) for m in tiny_model.modules())

    report = time_modules(tiny_model, prompt_ids, EOS, coarse_selector, phase="decode", n_steps=4)

    assert report.steps == 4
    assert report.ms_per_step["attn"] > 0
    assert report.calls_per_step["attn"] == len(tiny_model.model.layers)
    assert sum(len(m._forward_hooks) for m in tiny_model.modules()) == before


def test_time_modules_prefill_counts_a_single_step(tiny_model, prompt_ids):
    report = time_modules(tiny_model, prompt_ids, EOS, coarse_selector, phase="prefill", n_steps=1)

    assert report.steps == 1
    assert report.calls_per_step["mlp"] == len(tiny_model.model.layers)


@pytest.mark.parametrize("phase", ["prefill", "decode"])
def test_replay_linears_times_every_linear_group_and_removes_hooks(tiny_model, prompt_ids, phase):
    before = sum(len(m._forward_pre_hooks) for m in tiny_model.modules())
    layers = len(tiny_model.model.layers)

    report = replay_linears(tiny_model, prompt_ids, EOS, phase, repeats=3)

    assert report.calls_per_step == {"head_linear": 1, "mlp_linear": 3 * layers, "proj": 4 * layers}
    assert all(v > 0 for v in report.ms_per_step.values())
    assert all(0 < f <= 1.0 for f in report.enqueue_fraction.values())
    assert sum(len(m._forward_pre_hooks) for m in tiny_model.modules()) == before


def test_replay_linears_rejects_unknown_phase(tiny_model, prompt_ids):
    with pytest.raises(ValueError, match="phase"):
        replay_linears(tiny_model, prompt_ids, EOS, "sideways")


def test_time_modules_rejects_unknown_phase(tiny_model, prompt_ids):
    with pytest.raises(ValueError, match="phase"):
        time_modules(tiny_model, prompt_ids, EOS, coarse_selector, phase="sideways", n_steps=1)


def test_sync_bracket_overhead_is_small_and_positive_on_cpu():
    assert 0 < sync_bracket_overhead_us(CPU, repeats=50) < 1000


def test_attribution_splits_attention_core_from_projections():
    coarse = {"attn": 4.0, "mlp": 5.0, "norm": 1.0, "lm_head": 2.0, "embed": 0.0}
    matmul = {"proj": 1.0, "mlp_linear": 4.5, "head_linear": 2.0}

    result = attribution(coarse, matmul, step_ms=14.0)

    assert result["matmul_ms"] == pytest.approx(7.5)
    assert result["attention_core_ms"] == pytest.approx(3.0)  # attn 4.0 minus its 1.0 of projections
    assert result["other_ms"] == pytest.approx(14.0 - 7.5 - 3.0)
    assert result["matmul_share"] == pytest.approx(7.5 / 14)
    assert result["attention_core_share"] == pytest.approx(3 / 14)


def test_attribution_subtracts_bracketing_overhead_from_attention_core():
    result = attribution({"attn": 4.0}, {"proj": 1.0}, step_ms=10.0, attn_overhead_ms=2.0)

    assert result["attention_core_ms"] == pytest.approx(1.0)


def test_attribution_clamps_negative_core_time_to_zero():
    result = attribution({"attn": 1.0}, {"proj": 3.0, "mlp_linear": 0.0, "head_linear": 0.0}, step_ms=10.0)

    assert result["attention_core_ms"] == 0.0


def test_decode_window_excludes_prefill_exactly(tiny_model, prompt_ids):
    """n_steps=3 with 2 layers: a leaked prefill would give 8 attn calls, not a multiple of 3."""
    layers = len(tiny_model.model.layers)

    report = time_modules(tiny_model, prompt_ids, EOS, coarse_selector, phase="decode", n_steps=3)

    assert report.steps == 3
    assert report.calls_per_step["attn"] == layers


def test_hooks_are_removed_when_generation_raises(tiny_model, prompt_ids, monkeypatch):
    import module_timing

    def boom(*_a, **_k):
        raise RuntimeError("generation failed")

    monkeypatch.setattr(module_timing, "generate_cached", boom)
    before = sum(len(m._forward_hooks) + len(m._forward_pre_hooks) for m in tiny_model.modules())

    with pytest.raises(RuntimeError, match="generation failed"):
        time_modules(tiny_model, prompt_ids, EOS, coarse_selector, phase="decode", n_steps=3)
    with pytest.raises(RuntimeError, match="generation failed"):
        replay_linears(tiny_model, prompt_ids, EOS, "decode", repeats=1)

    after = sum(len(m._forward_hooks) + len(m._forward_pre_hooks) for m in tiny_model.modules())
    assert after == before


def test_hooks_are_removed_when_a_selector_raises_midway(tiny_model, prompt_ids):
    seen = {"n": 0}

    def flaky(name, module):
        seen["n"] += 1
        if seen["n"] > 5:
            raise RuntimeError("selector failed")
        return coarse_selector(name, module)

    before = sum(len(m._forward_hooks) + len(m._forward_pre_hooks) for m in tiny_model.modules())

    with pytest.raises(RuntimeError, match="selector failed"):
        time_modules(tiny_model, prompt_ids, EOS, flaky, phase="decode", n_steps=3)

    assert sum(len(m._forward_hooks) + len(m._forward_pre_hooks) for m in tiny_model.modules()) == before


def test_attribution_flags_clamped_core_and_negative_other():
    clamped = attribution({"attn": 1.0}, {"proj": 3.0, "mlp_linear": 0.0, "head_linear": 0.0}, step_ms=10.0)
    assert clamped["attention_core_clamped"] is True

    over = attribution({"attn": 4.0}, {"proj": 1.0, "mlp_linear": 9.0, "head_linear": 3.0}, step_ms=10.0)
    assert over["other_negative"] is True
    assert over["attention_core_clamped"] is False


def test_attribution_rejects_non_positive_step_time():
    with pytest.raises(ValueError, match="step_ms"):
        attribution({}, {}, step_ms=0.0)
