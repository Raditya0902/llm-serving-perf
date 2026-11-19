"""Compare the outputs of two attention backends (performance-report 9.9: is TRITON_ATTN a drop-in for FlashAttention 2?).

    python -m gpu.quality_compare <dir with fa2.json triton.json fa2_bt8192.json [reference.json]>

Run files come from gpu/quality_run.py. The control is the same kernel with a different chunking (fa2_bt8192): any difference
between two correct fp16 implementations is at least that large, so Triton is judged against that noise floor, not against zero.
Decision rule (PREREGISTRATION_ROUND4.md, Q1) - equivalent if all hold:
  1 mean |delta chosen-token log-prob| and mean |delta prompt log-prob| each at most 2x the control's
  2 exact-match rate within 10 points of the control's
  3 mean prompt-log-prob error against the fp32 reference at most 1.5x FlashAttention 2's
  4 at least 80% of the divergences are near-ties (gap below 0.5 nats in the FlashAttention 2 run)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

NOISE_FLOOR_EPS = 1e-4  # nats; used when the control is (almost) exactly zero
LOGPROB_FACTOR, REFERENCE_FACTOR = 2.0, 1.5
EXACT_MATCH_MARGIN, NEAR_TIE_GAP, NEAR_TIE_SHARE = 0.10, 0.5, 0.8


def _by_id(run: dict) -> dict[int, dict]:
    return {p["id"]: p for p in run["prompts"]}


def _first_divergence(a: list[int], b: list[int]) -> int | None:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def _gap(run_a_prompt: dict, position: int, other_token: int) -> float | None:
    top = run_a_prompt["top"][position] if position < len(run_a_prompt["top"]) else {}
    if str(other_token) not in top:
        return None
    return run_a_prompt["logprobs"][position] - top[str(other_token)]


def _mean_abs_prompt_diff(pa: list, pb: list) -> list[float]:
    return [abs(x - y) for x, y in zip(pa, pb) if x is not None and y is not None]


def compare(run_a: dict, run_b: dict) -> dict:
    a, b = _by_id(run_a), _by_id(run_b)
    if a.keys() != b.keys():
        raise ValueError("runs must cover the same prompts")
    exact, firsts, dlp, dprompt, gaps = 0, [], [], [], []
    for pid, pa in a.items():
        pb = b[pid]
        div = _first_divergence(pa["tokens"], pb["tokens"])
        common = len(pa["tokens"]) if div is None else div
        dlp += [abs(x - y) for x, y in zip(pa["logprobs"][:common], pb["logprobs"][:common])]
        dprompt += _mean_abs_prompt_diff(pa["prompt_logprobs"], pb["prompt_logprobs"])
        if div is None:
            exact += 1
            continue
        firsts.append(div)
        if div < min(len(pa["tokens"]), len(pb["tokens"])):
            gaps.append(_gap(pa, div, pb["tokens"][div]))
    near = [g is not None and g < NEAR_TIE_GAP for g in gaps]
    return {"prompts": len(a), "exact_match_rate": exact / len(a), "divergences": len(firsts),
            "mean_first_divergence": float(np.mean(firsts)) if firsts else None,
            "mean_abs_dlogprob": float(np.mean(dlp)) if dlp else 0.0, "mean_abs_dprompt": float(np.mean(dprompt)) if dprompt else 0.0,
            "near_tie_share": float(np.mean(near)) if near else None}


def error_vs_reference(run: dict, reference: dict) -> float:
    ref = _by_id(reference)
    errors = []
    for pid, p in _by_id(run).items():
        if pid in ref:
            errors += _mean_abs_prompt_diff(p["prompt_logprobs"], ref[pid]["prompt_logprobs"])
    return float(np.mean(errors)) if errors else float("nan")


def verdict(triton_vs_fa2: dict, control: dict, err_fa2: float | None, err_triton: float | None) -> dict:
    t, c = triton_vs_fa2, control
    criteria = {
        "logprob_within_2x": t["mean_abs_dlogprob"] <= LOGPROB_FACTOR * max(c["mean_abs_dlogprob"], NOISE_FLOOR_EPS),
        "prompt_logprob_within_2x": t["mean_abs_dprompt"] <= LOGPROB_FACTOR * max(c["mean_abs_dprompt"], NOISE_FLOOR_EPS),
        "exact_match_within_10_points": t["exact_match_rate"] >= c["exact_match_rate"] - EXACT_MATCH_MARGIN,
        "divergences_are_near_ties": t["near_tie_share"] is None or t["near_tie_share"] >= NEAR_TIE_SHARE,
    }
    if err_fa2 is not None and err_triton is not None:
        criteria["reference_error_within_1_5x"] = err_triton <= REFERENCE_FACTOR * max(err_fa2, NOISE_FLOOR_EPS)
    return {"criteria": criteria, "equivalent": all(criteria.values())}


def _load(directory: Path, name: str) -> dict | None:
    path = directory / f"{name}.json"
    return json.loads(path.read_text()) if path.is_file() else None


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        raise SystemExit("usage: python -m gpu.quality_compare <dir>")
    directory = Path(argv[0])
    fa2, triton, control = (_load(directory, n) for n in ("fa2", "triton", "fa2_bt8192"))
    if fa2 is None or triton is None or control is None:
        raise SystemExit("need fa2.json, triton.json and fa2_bt8192.json")
    reference = _load(directory, "reference")
    t_vs, c_vs = compare(fa2, triton), compare(fa2, control)
    err_fa2 = error_vs_reference(fa2, reference) if reference else None
    err_triton = error_vs_reference(triton, reference) if reference else None
    report = {"triton_vs_fa2": t_vs, "control_fa2_vs_fa2_bt8192": c_vs, "error_vs_fp32_reference": {"fa2": err_fa2, "triton": err_triton},
              "verdict": verdict(t_vs, c_vs, err_fa2, err_triton)}
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
