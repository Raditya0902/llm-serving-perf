"""Which kernels carry the per-step cost? Per-kernel attribution of vLLM torch-profiler steps.

    python -m gpu.kernel_costs <trace dir> [<trace dir> ...]      # from day6-profiling/

`step_costs.py` fits a whole step's GPU time against prefill tokens and decodes. This module splits the same steps by
kernel family (attention, gemm, norm/activation, kv-cache write, sampling, comm, other) so the fit can be made per family:
the family whose per-decode slope in *mixed* steps carries most of the total slope is where the extra cost of a decode riding
in a mixed step comes from. Each kernel is assigned to the step annotation that contains its start time.

Coverage (kernel ms / step ms) is reported per step: a low value means the parse missed kernels (or the step is mostly idle),
and an attribution built on it must not be trusted. This is the lesson of the invalid graph-mode nsys capture in Day 5.
"""
from __future__ import annotations

import bisect
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from gpu.step_costs import MIN_MIXED_FOR_FIT, Step, decodes_identifiable, load_events, parse_steps

FAMILIES = ("attention", "gemm", "norm_act", "kv_write", "sampling", "comm", "other")
KERNEL_CAT = "kernel"
MIN_COVERAGE = 0.8
DOMINANCE_THRESHOLD = 0.7
TOP_KERNELS = 5

# Order matters: the first family with a matching marker wins.
_MARKERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("kv_write", ("reshape_and_cache", "concat_and_cache", "cache_kernel", "copy_blocks")),  # before attention: "..._cache_flash_..."
    ("attention", ("flash", "fmha", "attention", "mha_", "paged_attn", "splitkv")),
    ("comm", ("nccl", "cross_device_reduce", "allreduce", "all_reduce", "custom_ar")),
    ("sampling", ("topk", "topp", "sampl", "argmax", "multinomial", "radixsort", "radix_sort", "softmax")),
    ("gemm", ("gemm", "gemv", "cutlass", "cublas", "xmma", "nvjet", "splitk", "wmma", "sm80_", "sm86_", "ampere_")),
    ("norm_act", ("rms_norm", "layer_norm", "rotary", "rope", "silu", "gelu", "act_and_mul", "elementwise",
                  "triton_poi", "triton_red", "triton_per", "fused_add", "vectorized")),
)


@dataclass(frozen=True)
class Kernel:
    ts: float
    ms: float
    name: str


@dataclass
class StepKernels:
    step: Step
    family_ms: dict[str, float] = field(default_factory=dict)
    name_ms: dict[str, float] = field(default_factory=dict)

    @property
    def kernel_ms(self) -> float:
        return sum(self.family_ms.values())

    @property
    def coverage(self) -> float:
        return self.kernel_ms / self.step.ms if self.step.ms > 0 else 0.0


def classify_kernel(name: str) -> str:
    lowered = name.lower().replace(" ", "")
    for family, markers in _MARKERS:
        if any(marker in lowered for marker in markers):
            return family
    return "other"


def parse_kernels(events: list[dict]) -> list[Kernel]:
    kernels = [Kernel(e["ts"], e["dur"] / 1000, e.get("name", "")) for e in events
               if e.get("ph") == "X" and e.get("cat") == KERNEL_CAT]
    return sorted(kernels, key=lambda k: k.ts)


def attribute(steps: list[Step], kernels: list[Kernel]) -> list[StepKernels]:
    ordered = sorted(steps, key=lambda s: s.ts)
    starts = [s.ts for s in ordered]
    result = [StepKernels(step) for step in ordered]
    for kernel in kernels:
        i = bisect.bisect_right(starts, kernel.ts) - 1
        if i < 0 or kernel.ts > ordered[i].ts + ordered[i].ms * 1000:
            continue
        family = classify_kernel(kernel.name)
        row = result[i]
        row.family_ms[family] = row.family_ms.get(family, 0.0) + kernel.ms
        row.name_ms[kernel.name] = row.name_ms.get(kernel.name, 0.0) + kernel.ms
    return result


def unclassified_top(rows: list[StepKernels], n: int = 10) -> list[tuple[str, float]]:
    totals: dict[str, float] = defaultdict(float)
    for row in rows:
        for name, ms in row.name_ms.items():
            if classify_kernel(name) == "other":
                totals[name] += ms
    return sorted(totals.items(), key=lambda item: -item[1])[:n]


def _group(rows: list[StepKernels]) -> dict:
    if not rows:
        return {"n": 0, "mean_ms_by_family": {}, "top_kernels": []}
    by_family = {f: float(np.mean([r.family_ms.get(f, 0.0) for r in rows])) for f in FAMILIES}
    names: dict[str, float] = defaultdict(float)
    for r in rows:
        for name, ms in r.name_ms.items():
            names[name] += ms / len(rows)
    top = sorted(names.items(), key=lambda item: -item[1])[:TOP_KERNELS]
    return {"n": len(rows), "mean_ms_by_family": by_family, "top_kernels": [[n, round(ms, 4)] for n, ms in top]}


def _fits(mixed: list[StepKernels]) -> dict | None:
    if len(mixed) < MIN_MIXED_FOR_FIT:
        return None
    design = np.c_[np.ones(len(mixed)), [r.step.prefill_tokens for r in mixed], [r.step.decodes for r in mixed]]
    fits = {}
    for family in FAMILIES:
        coef = np.linalg.lstsq(design, [r.family_ms.get(family, 0.0) for r in mixed], rcond=None)[0]
        fits[family] = {"intercept_ms": float(coef[0]), "ms_per_1000_prefill_tokens": float(coef[1] * 1000),
                        "ms_per_decode": float(coef[2])}
    return fits


def _slope_shares(fits: dict | None) -> dict:
    if not fits:
        return {}
    total = sum(max(f["ms_per_decode"], 0.0) for f in fits.values())
    if total <= 0:
        return {}
    return {family: max(f["ms_per_decode"], 0.0) / total for family, f in fits.items()}


def summarize(rows: list[StepKernels]) -> dict:
    pure = [r for r in rows if r.step.prefill_tokens == 0]
    mixed = [r for r in rows if r.step.prefill_tokens > 0]
    fits = _fits(mixed)
    identifiable = decodes_identifiable([r.step.decodes for r in mixed])
    coverage = [r.coverage for r in rows]
    warnings = []
    if coverage and min(coverage) < MIN_COVERAGE:
        low = sum(1 for c in coverage if c < MIN_COVERAGE)
        warnings.append(f"coverage below {MIN_COVERAGE:.0%} in {low} of {len(rows)} steps; kernel ms / step ms min "
                        f"{min(coverage):.0%}, so the attribution misses time")
    if fits is not None and not identifiable:
        warnings.append("per-decode slope not identified: the number of decodes barely varies across mixed steps, so the "
                        "per-decode fits (and decode_slope_share) only reflect a constant; use a controlled sweep over decodes")
    return {
        "steps": len(rows),
        "coverage": {"mean": float(np.mean(coverage)) if coverage else None, "min": float(min(coverage)) if coverage else None},
        "pure": _group(pure),
        "mixed": {**_group(mixed), "fits": fits, "decode_identifiable": identifiable,
                  "decode_slope_share": _slope_shares(fits) if identifiable else {}},
        "warnings": warnings,
    }


def dominant_decode_family(summary: dict, threshold: float = DOMINANCE_THRESHOLD) -> str | None:
    shares = summary.get("mixed", {}).get("decode_slope_share", {})
    winners = [family for family, share in shares.items() if share >= threshold]
    return winners[0] if winners else None


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__, file=sys.stderr)
        return 2
    out = {}
    for directory in argv:
        events = load_events(directory)
        rows = attribute(parse_steps(events), parse_kernels(events))
        summary = summarize(rows)
        summary["dominant_decode_family"] = dominant_decode_family(summary)
        summary["unclassified_top"] = [[n, round(ms, 3)] for n, ms in unclassified_top(rows)]
        out[Path(directory).name] = summary
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
