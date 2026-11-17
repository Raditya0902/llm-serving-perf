"""The pre-registered optimization objective, and the before/after comparison.

Objective (fixed before any run, per workload):

    maximize  output tokens / s
    subject to  P95 end-to-end latency <= L,   L = K_THRESHOLD x (baseline config's P95 e2e at
                                                                  the lowest concurrency)

The lowest-concurrency baseline P95 is the hardware's unloaded latency floor for that
workload, so ``L`` is derived from the baseline hardware and workload instead of assuming an
arbitrary number is achievable. The winner is the best feasible (config, concurrency) cell.

A ``tuned`` win is adopted only if it clears the noise rules in ``stability.compare``;
otherwise the recommendation is to keep the (simpler) baseline.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import stability

K_THRESHOLD = 3.0
BASELINE = "baseline"
TUNED = "tuned"
THROUGHPUT_METRIC = "output_tok_s"
LATENCY_METRIC = "p95_e2e_ms"

HIGHER_IS_BETTER = frozenset({"req_per_s", "output_tok_s"})
LOWER_IS_BETTER = frozenset({
    "mean_ttft_ms", "p95_ttft_ms", "mean_tpot_ms", "p95_tpot_ms", "mean_e2e_ms", "p95_e2e_ms",
    "preemptions", "recomputed_tokens", "peak_waiting", "peak_gpu_mem_gib",
})
COMPARED_METRICS: tuple[str, ...] = (
    "mean_ttft_ms", "p95_ttft_ms", "mean_tpot_ms", "p95_tpot_ms", "p95_e2e_ms", "req_per_s",
    "output_tok_s", "peak_gpu_mem_gib", "kv_peak_frac", "preemptions", "peak_waiting",
)
IMPROVED, REGRESSED, UNCHANGED, CHANGED, NOT_MEASURED = (
    "improved", "regressed", "unchanged", "changed", "not measured",
)


@dataclass(frozen=True)
class Decision:
    workload: str
    k: float
    threshold_ms: float
    feasible: tuple[str, ...]
    best_case: str | None
    baseline_case: str | None  # baseline cell at the winner's concurrency
    delta_pct: float | None
    verdict: str | None
    recommendation: str
    same_concurrency_baseline_feasible: bool | None = None  # None: that baseline cell is absent


def _case_id(workload: str, concurrency: int, config: str) -> str:
    return f"{workload}-c{concurrency}-{config}"


def _cells(summaries: Sequence[Mapping[str, Any]], workload: str) -> list[Mapping[str, Any]]:
    return [s for s in summaries if s["workload"] == workload and not s["is_control"]]


def threshold_ms(summaries: Sequence[Mapping[str, Any]], workload: str, k: float = K_THRESHOLD) -> float:
    cells = _cells(summaries, workload)
    if not cells:
        raise ValueError(f"no cases for workload {workload!r}")
    lowest = min(c["concurrency"] for c in cells)
    floor_id = _case_id(workload, lowest, BASELINE)
    floor = next((c for c in cells if c["case_id"] == floor_id), None)
    if floor is None or floor[LATENCY_METRIC] is None:
        raise ValueError(f"the latency floor case {floor_id} is missing; cannot derive the threshold")
    return k * floor[LATENCY_METRIC]["mean"]


def noise_floor_pct(summaries: Sequence[Mapping[str, Any]], metric: str) -> float:
    """% below which a difference is noise: the A/A delta if a control exists, else the max CV."""
    by_id = {s["case_id"]: s for s in summaries}
    for control in (s for s in summaries if s["is_control"]):
        original = by_id.get(control["case_id"].removesuffix("-aa"))
        if original and original[metric] and control[metric]:
            return stability.aa_floor_pct(original, control, metric)
    cvs = [s[metric]["cv"] for s in summaries if s[metric] and s[metric]["cv"] is not None]
    return max(cvs, default=0.0) * stability.PCT


def pair_floor_pct(
    summaries: Sequence[Mapping[str, Any]], base: Mapping[str, Any], other: Mapping[str, Any], metric: str,
) -> float:
    """Noise floor for one comparison: the matrix-wide floor, widened by both cases' own CV."""
    own = [s[metric]["cv"] for s in (base, other) if s[metric] and s[metric]["cv"] is not None]
    return max([noise_floor_pct(summaries, metric), *(c * stability.PCT for c in own)])


def select(summaries: Sequence[Mapping[str, Any]], workload: str, k: float = K_THRESHOLD) -> Decision:
    """Best feasible cell, and whether tuned really beats the best *feasible* baseline cell.

    Comparing against the baseline at the winner's concurrency would be wrong when that cell misses
    the latency limit: keeping the baseline would then violate the objective. A tuned win must also
    clear the limit by more than the noise on P95, not merely sit under it."""
    limit = threshold_ms(summaries, workload, k)
    cells = _cells(summaries, workload)
    feasible = [c for c in cells if c[LATENCY_METRIC] and c[LATENCY_METRIC]["mean"] <= limit]
    ids = tuple(c["case_id"] for c in feasible)
    if not feasible:
        return Decision(workload, k, limit, ids, None, None, None, None,
                        "no feasible configuration: every cell exceeds the latency limit")

    best = max(feasible, key=lambda c: c[THROUGHPUT_METRIC]["mean"])
    same = next((c for c in cells if c["case_id"] == _case_id(workload, best["concurrency"], BASELINE)), None)
    same_feasible = None if same is None else same["case_id"] in ids

    def decision(base_id, delta, verdict, text) -> Decision:
        return Decision(workload, k, limit, ids, best["case_id"], base_id, delta, verdict, text, same_feasible)

    if best["config"] == BASELINE:
        return decision(best["case_id"], None, None,
                        f"keep baseline: {best['case_id']} is already the best feasible cell")
    baselines = [c for c in feasible if c["config"] == BASELINE]
    if not baselines:
        return decision(None, None, None, f"adopt tuned: no baseline cell meets the limit; "
                                          f"{best['case_id']} is the only feasible configuration")
    ref = max(baselines, key=lambda c: c[THROUGHPUT_METRIC]["mean"])

    floor = pair_floor_pct(summaries, ref, best, THROUGHPUT_METRIC)
    result = stability.compare(ref, best, THROUGHPUT_METRIC, floor_pct=floor)
    delta, verdict = result["delta_pct"], result["verdict"]
    if verdict != stability.VERDICT_REAL:
        return decision(ref["case_id"], delta, verdict,
                        f"keep baseline: tuned's {delta:+.1f}% {THROUGHPUT_METRIC} is {verdict} (noise floor {floor:.1f}%)")
    lat_floor = pair_floor_pct(summaries, ref, best, LATENCY_METRIC)
    margin = (limit - best[LATENCY_METRIC]["mean"]) / limit * stability.PCT
    if margin <= lat_floor:
        return decision(ref["case_id"], delta, verdict,
                        f"keep baseline: tuned's {delta:+.1f}% gain is real, but its P95 is within noise of the "
                        f"latency limit (margin {margin:.1f}% <= noise {lat_floor:.1f}%)")
    note = "" if same_feasible in (True, None) else "; the baseline is not feasible there"
    return decision(ref["case_id"], delta, verdict,
                    f"adopt tuned: +{delta:.1f}% {THROUGHPUT_METRIC} at concurrency {best['concurrency']}{note}")


def _direction(metric: str, base_mean: float, other_mean: float, verdict: str) -> str:
    if verdict == stability.VERDICT_UNMEASURED:
        return NOT_MEASURED
    if verdict != stability.VERDICT_REAL:
        return UNCHANGED
    went_up = other_mean > base_mean
    if metric in HIGHER_IS_BETTER:
        return IMPROVED if went_up else REGRESSED
    if metric in LOWER_IS_BETTER:
        return REGRESSED if went_up else IMPROVED
    return CHANGED


def before_after(
    before: Mapping[str, Any], after: Mapping[str, Any], *, floor_pct: float | Callable[[str], float],
    metrics: Sequence[str] = COMPARED_METRICS,
) -> list[dict[str, Any]]:
    """Per-metric comparison. `floor_pct` is one number, or a function metric -> floor."""
    rows = []
    for metric in metrics:
        floor = floor_pct(metric) if callable(floor_pct) else floor_pct
        cmp = stability.compare(before, after, metric, floor_pct=floor)
        b, a = before[metric], after[metric]
        rows.append({
            "metric": metric,
            "before": None if b is None else b["mean"],
            "after": None if a is None else a["mean"],
            "delta_pct": cmp["delta_pct"],
            "verdict": cmp["verdict"],
            "direction": _direction(metric, *(x["mean"] if x else 0.0 for x in (b, a)), cmp["verdict"]),
        })
    return rows
