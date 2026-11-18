"""Summarize `run_followup_gpu.py` arms next to the matrix's B@64 baseline/tuned runs (read straight from raw JSON).

    python summarize_followup.py --matrix-raw results/raw/a40-capped --followup-raw results/raw/followup
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path

CASES = ("B-c64-baseline", "B-c64-tuned", "B-c64-budget2048", "B-c64-pool092",
         "B-c64-cap16", "B-c64-cap24", "B-c64-cap32",
         "B-c16-tp1", "B-c16-tp2", "B-c64-tp1", "B-c64-tp2", "B-c64-tp2nc")
# Each row is compared with its own baseline: the single-GPU matrix baseline, or the TP=1 run at the same concurrency.
BASELINE_FOR = {"B-c16-tp2": "B-c16-tp1", "B-c64-tp2": "B-c64-tp1", "B-c64-tp2nc": "B-c64-tp1"}
METRICS = ("output_throughput", "mean_ttft_ms", "mean_tpot_ms", "p95_e2el_ms")


def load_runs(raw: Path, case_id: str) -> list[dict]:
    runs = []
    for path in sorted(raw.glob(f"{case_id}__r*.json")):
        bench = json.loads(path.read_text())
        sidecar = json.loads((raw / "sidecar" / path.name).read_text())["poll"]
        runs.append({**{m: float(bench[m]) for m in METRICS}, "failed": bench["failed"],
                     "kv_peak": sidecar["kv_peak_frac"], "preempt": sidecar["preemptions"],
                     "peak_wait": sidecar["waiting_peak"]})
    return runs


def mean_cv(values: Sequence[float]) -> tuple[float, float]:
    mean = statistics.fmean(values)
    cv = statistics.stdev(values) / mean * 100 if len(values) > 1 and mean else float("nan")
    return mean, cv


def summarize(raws: Sequence[Path]) -> dict[str, dict[str, float]]:
    table: dict[str, dict[str, float]] = {}
    for case_id in CASES:
        runs = [r for raw in raws for r in load_runs(raw, case_id)]
        if not runs:
            continue
        row: dict[str, float] = {"n": len(runs), "failed": sum(r["failed"] for r in runs)}
        for key in (*METRICS, "kv_peak", "preempt", "peak_wait"):
            row[key], row[f"{key}_cv"] = mean_cv([r[key] for r in runs])
        table[case_id] = row
    return table


def render(table: Mapping[str, Mapping[str, float]]) -> str:
    lines = ["| case | n | out tok/s (CV) | vs base | mean TTFT ms | vs base | mean TPOT ms | vs base | P95 e2e ms | KV peak | preempt | peak wait |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]

    def delta(row: Mapping[str, float], base: Mapping[str, float] | None, key: str) -> str:
        return "-" if base is None or row is base else f"{(row[key] / base[key] - 1) * 100:+.1f}%"

    for case_id, r in table.items():
        base = table.get(BASELINE_FOR.get(case_id, "B-c64-baseline"))
        lines.append(
            f"| {case_id} | {r['n']:.0f} | {r['output_throughput']:.0f} ({r['output_throughput_cv']:.1f}%) | "
            f"{delta(r, base, 'output_throughput')} | {r['mean_ttft_ms']:.0f} | {delta(r, base, 'mean_ttft_ms')} | "
            f"{r['mean_tpot_ms']:.1f} | {delta(r, base, 'mean_tpot_ms')} | {r['p95_e2el_ms']:.0f} | "
            f"{r['kv_peak'] * 100:.0f}% | {r['preempt']:.1f} | {r['peak_wait']:.0f} |")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--matrix-raw", type=Path, required=True)
    p.add_argument("--followup-raw", type=Path, required=True)
    args = p.parse_args(argv)
    print(render(summarize([args.matrix_raw, args.followup_raw])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
