"""Comparison of two attention backends' outputs (synthetic run files; the runs themselves need a GPU)."""
from __future__ import annotations

import json

import pytest

from gpu import quality_compare as qc


def _prompt(pid, tokens, logprobs, prompt_lp=(None, -1.0, -2.0), top=None):
    return {"id": pid, "n_prompt": len(prompt_lp), "prompt_logprobs": list(prompt_lp), "tokens": list(tokens),
            "logprobs": list(logprobs), "top": top if top is not None else [{str(t): lp} for t, lp in zip(tokens, logprobs)]}


def _run(*prompts):
    return {"config": {}, "prompts": list(prompts)}


def test_identical_runs_match_exactly_with_zero_logprob_difference():
    a = _run(_prompt(0, [5, 6, 7], [-0.1, -0.2, -0.3]), _prompt(1, [8, 9], [-1.0, -1.1]))
    out = qc.compare(a, a)
    assert out["exact_match_rate"] == 1.0 and out["mean_abs_dlogprob"] == 0 and out["mean_abs_dprompt"] == 0
    assert out["divergences"] == 0 and out["near_tie_share"] is None and out["mean_first_divergence"] is None


def test_first_divergence_and_logprob_difference_over_the_common_prefix_only():
    a = _run(_prompt(0, [5, 6, 7], [-0.1, -0.2, -0.3], top=[{"5": -0.1}, {"6": -0.2, "9": -0.25}, {"7": -0.3}]))
    b = _run(_prompt(0, [5, 9, 1], [-0.1, -0.25, -4.0]))
    out = qc.compare(a, b)
    assert out["exact_match_rate"] == 0.0 and out["mean_first_divergence"] == 1
    assert out["mean_abs_dlogprob"] == pytest.approx(0.0)  # position 0 only: positions at and after the divergence are not comparable
    assert out["divergences"] == 1


def test_near_tie_is_the_gap_between_run_as_choice_and_the_token_run_b_chose():
    a = _run(_prompt(0, [6], [-0.2], top=[{"6": -0.2, "9": -0.3}]), _prompt(1, [6], [-0.2], top=[{"6": -0.2, "9": -3.0}]),
             _prompt(2, [6], [-0.2], top=[{"6": -0.2}]))
    b = _run(_prompt(0, [9], [-0.3]), _prompt(1, [9], [-3.0]), _prompt(2, [4], [-5.0]))
    out = qc.compare(a, b)
    assert out["divergences"] == 3
    assert out["near_tie_share"] == pytest.approx(1 / 3)  # gap 0.1 yes; gap 2.8 no; token outside the stored top-k no


def test_prompt_logprob_difference_skips_the_missing_first_position():
    a = _run(_prompt(0, [1], [-1.0], prompt_lp=(None, -1.0, -2.0)))
    b = _run(_prompt(0, [1], [-1.0], prompt_lp=(None, -1.5, -2.0)))
    assert qc.compare(a, b)["mean_abs_dprompt"] == pytest.approx(0.25)


def test_runs_with_different_prompt_sets_are_rejected():
    with pytest.raises(ValueError, match="same prompts"):
        qc.compare(_run(_prompt(0, [1], [-1.0])), _run(_prompt(1, [1], [-1.0])))


def test_error_against_the_reference_is_the_mean_absolute_prompt_logprob_error():
    ref = {"prompts": [{"id": 0, "prompt_logprobs": [None, -1.0, -2.0]}]}
    run = _run(_prompt(0, [1], [-1.0], prompt_lp=(None, -1.2, -2.4)), _prompt(7, [1], [-1.0]))
    assert qc.error_vs_reference(run, ref) == pytest.approx(0.3)  # prompts missing from the reference are ignored


def _cmp(dlp=0.01, dprompt=0.01, exact=0.9, near=1.0):
    return {"mean_abs_dlogprob": dlp, "mean_abs_dprompt": dprompt, "exact_match_rate": exact, "near_tie_share": near}


def test_verdict_equivalent_when_triton_is_within_the_noise_floor_of_fa2_against_itself():
    out = qc.verdict(triton_vs_fa2=_cmp(0.015, 0.012, 0.88), control=_cmp(0.01, 0.01, 0.9), err_fa2=0.02, err_triton=0.025)
    assert out["equivalent"] and all(out["criteria"].values())


@pytest.mark.parametrize("override,failed", [({"dlp": 0.05}, "logprob_within_2x"), ({"dprompt": 0.05}, "prompt_logprob_within_2x"),
                                              ({"exact": 0.7}, "exact_match_within_10_points"), ({"near": 0.5}, "divergences_are_near_ties")])
def test_verdict_names_the_criterion_that_fails(override, failed):
    out = qc.verdict(triton_vs_fa2=_cmp(**override), control=_cmp(0.01, 0.01, 0.9), err_fa2=0.02, err_triton=0.025)
    assert not out["equivalent"] and out["criteria"][failed] is False


def test_verdict_fails_when_triton_is_further_from_the_fp32_reference_than_fa2():
    out = qc.verdict(triton_vs_fa2=_cmp(), control=_cmp(), err_fa2=0.02, err_triton=0.05)
    assert out["criteria"]["reference_error_within_1_5x"] is False and not out["equivalent"]


def test_verdict_with_a_zero_noise_floor_uses_a_small_absolute_tolerance():
    ok = qc.verdict(triton_vs_fa2=_cmp(5e-5, 5e-5, 1.0), control=_cmp(0.0, 0.0, 1.0), err_fa2=0.0, err_triton=1e-5)
    assert ok["equivalent"]


def test_load_and_cli_report(tmp_path, capsys):
    base = _run(_prompt(0, [5, 6], [-0.1, -0.2]))
    for name in ("fa2", "triton", "fa2_bt8192"):
        (tmp_path / f"{name}.json").write_text(json.dumps(base))
    ref = {"prompts": [{"id": 0, "prompt_logprobs": [None, -1.0, -2.0]}]}
    (tmp_path / "reference.json").write_text(json.dumps(ref))
    assert qc.main([str(tmp_path)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["verdict"]["equivalent"] is True
