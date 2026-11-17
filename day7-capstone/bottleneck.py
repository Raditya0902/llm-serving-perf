"""Which resource limits a case? A transparent rule list, not a model.

Checked in this order; the first rule that fires wins:

1. KV capacity   : the scheduler preempted AND the KV pool was (near) full. Preemption only happens
                   when the pool is exhausted, so a pool gauge sampled every ~0.5 s that reads a
                   little under 100% (``KV_NEAR_FULL_FRAC``) still counts; with no gauge at all,
                   preemptions alone decide and the caveat says so.
2. Scheduler cap : the running batch sits at ``max_num_seqs``.
3. HBM bandwidth : the decode step is within ``DECODE_FLOOR_FACTOR`` of one full weight read
                   (decode streams every weight once per step, so that read is its floor; the
                   floor assumes peak bandwidth, which real GPUs do not reach, so 1.5x is generous).
4. Compute / interference : the decode step is well above the weight-read floor, i.e. the step
                   is stretched by prefill chunks sharing it, a large batch's FLOPs, or KV reads.

Thresholds are named constants so the report can quote them. When an input a rule needs was not
measured the result says so in ``caveat``; with no TPOT the label is ``unknown``. A rule that could
not be evaluated never silently counts as "did not fire".
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

KV_CAPACITY = "kv-capacity-bound"
SCHEDULER_CAP = "scheduler-bound (batch at max_num_seqs)"
BANDWIDTH_DECODE = "bandwidth-bound decode (weight reads)"
COMPUTE_OR_INTERFERENCE = "compute / prefill-interference bound"
UNKNOWN = "unknown (metric not measured)"

KV_NEAR_FULL_FRAC = 0.90
BATCH_AT_CAP_FRAC = 0.9
DECODE_FLOOR_FACTOR = 1.5


def _mean(summary: Mapping[str, Any], metric: str) -> float | None:
    stats = summary.get(metric)
    return None if stats is None else stats["mean"]


def _caveat(kv: float | None, pre: float | None, batch: float | None) -> str | None:
    notes = []
    if kv is None or pre is None:
        notes.append("KV pressure not measured (no KV gauge or preemption count), so the KV rule could not be fully checked")
    if batch is None:
        notes.append("batch size not measured, so the scheduler-cap rule was not checked")
    return "; ".join(notes) or None


def classify(summary: Mapping[str, Any], *, weight_read_ms: float, max_num_seqs: int) -> dict[str, str | None]:
    kv, pre = _mean(summary, "kv_peak_frac"), _mean(summary, "preemptions")
    batch, tpot = _mean(summary, "mean_batch"), _mean(summary, "mean_tpot_ms")
    caveat = _caveat(kv, pre, batch)

    if pre is not None and pre > 0 and (kv is None or kv >= KV_NEAR_FULL_FRAC):
        gauge = "KV gauge not exposed" if kv is None else f"KV pool {kv:.0%} full (>= {KV_NEAR_FULL_FRAC:.0%})"
        note = "KV gauge not exposed, so fullness was inferred from preemptions" if kv is None else caveat
        return {"label": KV_CAPACITY, "reason": f"{gauge} with {pre:.0f} preemptions", "caveat": note}
    if batch is not None and batch >= BATCH_AT_CAP_FRAC * max_num_seqs:
        return {"label": SCHEDULER_CAP, "caveat": caveat,
                "reason": f"mean batch {batch:.1f} >= {BATCH_AT_CAP_FRAC:.0%} of max_num_seqs {max_num_seqs}"}
    if tpot is None:
        return {"label": UNKNOWN, "reason": "TPOT was not measured", "caveat": caveat}
    floor = DECODE_FLOOR_FACTOR * weight_read_ms
    if tpot <= floor:
        return {"label": BANDWIDTH_DECODE, "caveat": caveat,
                "reason": f"TPOT {tpot:.1f} ms <= {DECODE_FLOOR_FACTOR}x weight-read floor {weight_read_ms:.1f} ms"}
    return {"label": COMPUTE_OR_INTERFERENCE, "caveat": caveat,
            "reason": f"TPOT {tpot:.1f} ms > {DECODE_FLOOR_FACTOR}x weight-read floor {weight_read_ms:.1f} ms"}
