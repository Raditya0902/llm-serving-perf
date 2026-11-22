"""Measurement stability: aggregate repeats, coefficient of variation, and the A/A noise floor.

Rules (carried over from `06-profiling`'s harness pitfalls):

* A difference between two cases is called ``real`` only if (a) the two cases' min..max
  ranges over their repeats do not overlap AND (b) it exceeds the noise floor.
* The noise floor is the larger of the observed A/A delta (a repeat of the same case with
  fresh prompts) and the case's own coefficient of variation.
* Fewer than two repeats can never establish a difference.
* A metric a source could not measure stays ``None`` ("not measured"); it is never zero.
* P95 over a few hundred requests is a tail estimate from ~10 samples; summaries carry
  ``p95_low_sample`` so reports can say so.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

METRICS: tuple[str, ...] = (
    "mean_ttft_ms", "p95_ttft_ms", "mean_tpot_ms", "p95_tpot_ms", "mean_e2e_ms", "p95_e2e_ms",
    "req_per_s", "output_tok_s", "peak_gpu_mem_gib", "modeled_kv_pool_gib", "kv_peak_frac",
    "preemptions", "recomputed_tokens", "peak_waiting", "mean_batch",
)
IDENTITY_FIELDS: tuple[str, ...] = (
    "case_id", "config_id", "workload", "config", "concurrency", "is_control", "source", "simulated",
)
CV_FLAG = 0.05  # CV above this marks a metric unstable
MIN_REPS_FOR_COMPARISON = 2
P95_MIN_REQUESTS = 200  # below this a per-run P95 rests on fewer than ~10 tail samples
PCT = 100.0

VERDICT_REAL = "real"
VERDICT_NOISE = "within noise"
VERDICT_FEW_REPS = "insufficient reps"
VERDICT_UNMEASURED = "not measured"


def cv(values: Sequence[float]) -> float | None:
    """Sample standard deviation over the mean; None if undefined."""
    if len(values) < 2:
        return None
    mean = float(np.mean(values))
    if mean == 0:
        return None
    return float(np.std(values, ddof=1)) / abs(mean)


def _stats(values: list[float]) -> dict[str, float | None]:
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if len(values) > 1 else None,
        "min": float(min(values)), "max": float(max(values)), "cv": cv(values), "n": len(values),
    }


def _metric_values(rows: Sequence[Mapping[str, Any]], metric: str) -> list[float]:
    """Values over all repeats, or [] if ANY repeat did not measure it (never average a subset)."""
    values = [row.get(metric) for row in rows]
    if any(v is None for v in values):
        return []
    if any(not math.isfinite(v) for v in values):
        raise ValueError(f"metric {metric!r} has a non-finite value; results must be finite")
    return [float(v) for v in values]


def _check_case_rows(case_id: str, reps: Sequence[Mapping[str, Any]]) -> None:
    if len({r["config_id"] for r in reps}) > 1:
        raise ValueError(f"case {case_id}: repeats have different config_id (mixed experiments)")
    if len({(r.get("source"), r.get("simulated")) for r in reps}) > 1:
        raise ValueError(f"case {case_id}: repeats mix sources (simulated and measured rows)")
    ids = [r.get("rep") for r in reps]
    if len(set(ids)) != len(ids):
        raise ValueError(f"case {case_id}: duplicate repeat numbers {sorted(i for i in ids if ids.count(i) > 1)}")


def aggregate_cases(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """One summary per `case_id`, in first-seen order."""
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["case_id"], []).append(row)

    summaries = []
    for case_id, reps in grouped.items():
        _check_case_rows(case_id, reps)
        summary: dict[str, Any] = {k: reps[0].get(k) for k in IDENTITY_FIELDS}
        summary["n_reps"] = len(reps)
        summary["completed"] = min(int(r.get("completed", 0)) for r in reps)
        summary["p95_low_sample"] = summary["completed"] < P95_MIN_REQUESTS
        for metric in METRICS:
            values = _metric_values(reps, metric)
            summary[metric] = _stats(values) if values else None
        summaries.append(summary)
    return summaries


def unstable_metrics(summary: Mapping[str, Any]) -> list[str]:
    return [
        m for m in METRICS
        if summary.get(m) and summary[m]["cv"] is not None and summary[m]["cv"] > CV_FLAG
    ]


def ranges_overlap(a: tuple[float, float], b: tuple[float, float]) -> bool:
    return a[0] <= b[1] and b[0] <= a[1]


def aa_floor_pct(original: Mapping[str, Any], control: Mapping[str, Any], metric: str) -> float:
    """Noise floor (%) for `metric`: the larger of the A/A delta and the original's CV."""
    o, c = original[metric], control[metric]
    if o is None or c is None:
        raise ValueError(f"metric {metric!r} was not measured; no noise floor")
    observed = abs(c["mean"] - o["mean"]) / abs(o["mean"]) * PCT if o["mean"] else 0.0
    own_cv = (o["cv"] or 0.0) * PCT
    return max(observed, own_cv)


def compare(
    base: Mapping[str, Any], other: Mapping[str, Any], metric: str, *, floor_pct: float,
) -> dict[str, Any]:
    """Is `other` really different from `base` on `metric`? See the module rules."""
    b, o = base[metric], other[metric]
    if b is None or o is None:
        return {"metric": metric, "verdict": VERDICT_UNMEASURED, "delta_pct": None,
                "ranges_overlap": None, "floor_pct": floor_pct}
    # A percentage change from a zero baseline is undefined; then only the ranges decide.
    delta = (o["mean"] - b["mean"]) / abs(b["mean"]) * PCT if b["mean"] else None
    overlap = ranges_overlap((b["min"], b["max"]), (o["min"], o["max"]))
    if min(b["n"], o["n"]) < MIN_REPS_FOR_COMPARISON:
        verdict = VERDICT_FEW_REPS
    elif not overlap and (delta is None or abs(delta) > floor_pct):
        verdict = VERDICT_REAL
    else:
        verdict = VERDICT_NOISE
    return {"metric": metric, "verdict": verdict, "delta_pct": delta,
            "ranges_overlap": overlap, "floor_pct": floor_pct}
