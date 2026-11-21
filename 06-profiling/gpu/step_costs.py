"""Per-step GPU time vs prefill tokens and running decodes, from vLLM torch-profiler Chrome traces.

    python -m gpu.step_costs <trace dir> [<trace dir> ...]      # from 06-profiling/
    python gpu/step_costs.py <trace dir>

Reads the `execute_context_<n>(<prefill tokens>)_generation_<m>(<decodes>)` annotations of rank 0 (GPU-side when the trace has
them), splits steps into pure decode (no prefill tokens) and mixed, and fits `ms = a + b * prefill_tokens/1000 + c * decodes`
to the mixed steps. `rule_of_thumb_ms` is the A40 / 0.5B rule from performance-report 9.3 (mixed step ms about 9 x prefill
tokens/1000 + 0.1 x decodes), kept here so a run at another context length can be compared with it.
"""
from __future__ import annotations

import glob
import gzip
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

STEP = re.compile(r"execute_context_(\d+)\((\d+)\)_generation_(\d+)\((\d+)\)")
GPU_CAT, CPU_CAT = "gpu_user_annotation", "user_annotation"
MIN_MIXED_FOR_FIT = 4
RULE_MS_PER_1000_PREFILL, RULE_MS_PER_DECODE = 9.0, 0.1
# The decode slope of the fit is only meaningful when the number of running decodes actually varies across the mixed steps.
# With a saturated batch (e.g. 249-255 decodes in every step) `decodes` is a constant and the slope cannot be separated from the
# intercept: only `intercept + slope * decodes` is determined. Require a spread of at least this fraction of the mean.
MIN_DECODE_SPREAD = 0.2


def decodes_identifiable(decodes: list[int]) -> bool:
    if len(decodes) < MIN_MIXED_FOR_FIT:
        return False
    mean = float(np.mean(decodes))
    return mean > 0 and (max(decodes) - min(decodes)) / mean >= MIN_DECODE_SPREAD


@dataclass(frozen=True)
class Step:
    ts: float
    ms: float
    prefill_tokens: int
    decodes: int


def rule_of_thumb_ms(*, prefill_tokens: int, decodes: int) -> float:
    return RULE_MS_PER_1000_PREFILL * prefill_tokens / 1000 + RULE_MS_PER_DECODE * decodes


def load_events(trace_dir: Path | str) -> list[dict]:
    matches = sorted(glob.glob(str(Path(trace_dir) / "rank0.*.pt.trace.json*")))
    if not matches:
        raise FileNotFoundError(f"no rank0.*.pt.trace.json(.gz) under {trace_dir}")
    path = matches[0]
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as fh:
        return json.load(fh)["traceEvents"]


def parse_steps(events: list[dict]) -> list[Step]:
    found = [(e, STEP.match(e.get("name", ""))) for e in events if e.get("ph") == "X" and e.get("cat") in (GPU_CAT, CPU_CAT)]
    found = [(e, m) for e, m in found if m]
    use = GPU_CAT if any(e["cat"] == GPU_CAT for e, _ in found) else CPU_CAT
    steps = [Step(e["ts"], e["dur"] / 1000, int(m.group(2)), int(m.group(4))) for e, m in found if e["cat"] == use]
    return sorted(steps, key=lambda s: s.ts)


def busy_fraction(steps: list[Step]) -> float | None:
    """Share of the wall span (first step start to last step end) covered by step GPU time, capped at 1.

    About 1.0 means the GPU never waited between steps (device-bound); clearly below 1 means gaps (CPU, scheduler or
    output processing between steps). None with fewer than two steps, where no gap can exist.
    """
    if len(steps) < 2:
        return None
    ordered = sorted(steps, key=lambda s: s.ts)
    span_us = max(s.ts + s.ms * 1000 for s in ordered) - ordered[0].ts
    if span_us <= 0:
        return None
    return float(min(1.0, sum(s.ms for s in ordered) * 1000 / span_us))


def _stats(steps: list[Step]) -> dict:
    if not steps:
        return {"n": 0}
    ms = np.array([s.ms for s in steps])
    return {"n": len(steps), "mean_ms": float(ms.mean()), "median_ms": float(np.median(ms)),
            "p90_ms": float(np.percentile(ms, 90)), "mean_decodes": float(np.mean([s.decodes for s in steps])),
            "mean_prefill_tokens": float(np.mean([s.prefill_tokens for s in steps]))}


def summarize(steps: list[Step]) -> dict:
    pure = [s for s in steps if s.prefill_tokens == 0]
    mixed = [s for s in steps if s.prefill_tokens > 0]
    out = {"steps": len(steps), "pure": _stats(pure), "mixed": _stats(mixed), "busy_fraction": busy_fraction(steps),
           "mixed_share_of_steps": len(mixed) / len(steps) if steps else None}
    fit = None
    if len(mixed) >= MIN_MIXED_FOR_FIT:
        design = np.c_[np.ones(len(mixed)), [s.prefill_tokens for s in mixed], [s.decodes for s in mixed]]
        coef = np.linalg.lstsq(design, [s.ms for s in mixed], rcond=None)[0]
        fit = {"intercept_ms": float(coef[0]), "ms_per_1000_prefill_tokens": float(coef[1] * 1000),
               "ms_per_decode": float(coef[2])}
        predicted = [rule_of_thumb_ms(prefill_tokens=s.prefill_tokens, decodes=s.decodes) for s in mixed]
        fit["mean_rule_of_thumb_ms"] = float(np.mean(predicted))
    identifiable = decodes_identifiable([s.decodes for s in mixed])
    if fit is not None and not identifiable:
        fit["note"] = ("ms_per_decode is not identified: the number of decodes barely varies across these mixed steps, so only "
                       "intercept + ms_per_decode x mean decodes is determined")
    out["mixed"]["decode_identifiable"] = identifiable
    out["mixed"]["fit"] = fit
    return out


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__, file=sys.stderr)
        return 2
    print(json.dumps({d: summarize(parse_steps(load_events(d))) for d in argv}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
