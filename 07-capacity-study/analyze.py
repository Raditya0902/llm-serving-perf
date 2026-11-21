"""Summaries -> the objective's decision, before/after pairs, bottleneck labels, stability flags.

    .venv/bin/python analyze.py --dir results/sim        # reads matrix_summary.json
    .venv/bin/python analyze.py --dir results/gpu/a100   # the same, for real vLLM results

Writes ``analysis.json`` (one object under ``rows[0]``) and ``multi_gpu.{json,csv}`` (the analytic
two-GPU / multi-node model for the same model and hardware; ASSUMED link constants).
Draws no prose conclusions: ``make_report.py`` renders the tables and the report author
interprets them.
"""
from __future__ import annotations

import argparse
import dataclasses
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import bottleneck
import kv_sizing
import matrix as mx
import multi_gpu
import objective
import results_io
import stability

NOISE_METRICS = ("output_tok_s", "p95_e2e_ms", "mean_ttft_ms", "mean_tpot_ms")
ANALYSIS_NOTE = ("derived from matrix_summary.json by analyze.py; thresholds and rules are named "
                 "constants in objective.py, stability.py and bottleneck.py")


def _find(summaries: Sequence[Mapping[str, Any]], case_id: str) -> Mapping[str, Any] | None:
    return next((s for s in summaries if s["case_id"] == case_id), None)


def _decisions(
    summaries: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Decision per workload present in the data; a workload that cannot be analysed is reported, not hidden."""
    decisions, skipped = [], {}
    for workload in mx.WORKLOADS:
        if not any(s["workload"] == workload for s in summaries):
            continue
        try:
            decisions.append(dataclasses.asdict(objective.select(summaries, workload)))
        except ValueError as exc:
            skipped[workload] = str(exc)
    return decisions, skipped


def _pairs(summaries: Sequence[Mapping[str, Any]], cells: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    pairs = []
    for workload in mx.WORKLOADS:
        for conc in mx.CONCURRENCY_LEVELS:
            base = _find(cells, f"{workload}-c{conc}-baseline")
            tuned = _find(cells, f"{workload}-c{conc}-tuned")
            if base is None or tuned is None:
                continue  # a partial matrix compares only the pairs it has
            rows = objective.before_after(
                base, tuned,
                floor_pct=lambda metric, b=base, t=tuned: objective.pair_floor_pct(summaries, b, t, metric),
            )
            pairs.append({"workload": workload, "concurrency": conc, "before": base["case_id"],
                          "after": tuned["case_id"], "rows": rows})
    return pairs


def analyze(
    summaries: Sequence[Mapping[str, Any]], *, hardware: str = mx.DEFAULT_HARDWARE,
    model_preset: str = mx.DEFAULT_PRESET,
) -> dict[str, Any]:
    cells = [s for s in summaries if not s["is_control"]]
    cost = kv_sizing.cost_model_for(kv_sizing.model_spec(model_preset), kv_sizing.HARDWARE[hardware])
    decisions, skipped = _decisions(summaries)
    bottlenecks = [
        {"case_id": s["case_id"],
         **bottleneck.classify(s, weight_read_ms=cost.weight_read_ms,
                               max_num_seqs=mx.CONFIGS[s["config"]].max_num_seqs)}
        for s in cells
    ]
    unstable = {s["case_id"]: stability.unstable_metrics(s) for s in summaries}
    return {
        "k_threshold": objective.K_THRESHOLD,
        "decisions": decisions, "skipped_workloads": skipped,
        "pairs": _pairs(summaries, cells), "bottlenecks": bottlenecks,
        "unstable": {k: v for k, v in unstable.items() if v},
        "p95_low_sample": sorted({s["case_id"] for s in summaries if s["p95_low_sample"]}),
        "noise_floor_pct": {m: objective.noise_floor_pct(summaries, m) for m in NOISE_METRICS},
        "weight_read_ms": cost.weight_read_ms,
    }


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", type=Path, required=True)
    p.add_argument("--hardware", default=mx.DEFAULT_HARDWARE, choices=sorted(kv_sizing.HARDWARE))
    p.add_argument("--model-preset", default=mx.DEFAULT_PRESET, choices=sorted(kv_sizing.MODELS))
    args = p.parse_args(argv)
    summary_path = args.dir / "matrix_summary.json"
    if not summary_path.is_file():
        print(f"{summary_path} not found; run run_matrix_sim.py or ingest_gpu.py first", file=sys.stderr)
        return 2
    source = results_io.read_results(summary_path)
    meta = {**source["meta"], "analysis_note": ANALYSIS_NOTE,
            "matrix_summary_sha256": results_io.file_sha256(summary_path)}
    analysis = analyze(source["rows"], hardware=args.hardware, model_preset=args.model_preset)
    results_io.write_results("analysis", [analysis], meta, args.dir, write_csv=False)
    baseline_util = mx.CONFIGS["baseline"].gpu_memory_utilization
    rows = multi_gpu.sweep(args.model_preset, args.hardware, util=baseline_util)
    results_io.write_results("multi_gpu", rows, {**meta, "assumed_links": {
        n: dataclasses.asdict(link) for n, link in multi_gpu.LINKS.items()}}, args.dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
