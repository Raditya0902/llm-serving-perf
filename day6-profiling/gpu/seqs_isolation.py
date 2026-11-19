"""Summarise the max_num_seqs isolation grid written by gpu/run_seqs_isolation.sh and apply the pre-registered verdict rule.

    python -m gpu.seqs_isolation <dir with seqs<S>_<IN>x<OUT>_r<R>.json> [--busy-256 0.93 --busy-1024 0.91]

Hypotheses (fixed before the runs, see the plan and performance-report 9.11):
  H1  decodes riding in expensive mixed steps: the loss shrinks when the mixed phase is a smaller share of the run
      (512 output tokens, or 128 input tokens).
  H2  per-step overhead that grows with running sequences (scheduler, sampler, detokenisation): the loss persists across
      workloads, or the 1024 arm leaves the GPU idle noticeably more of the time.
Everything else is reported as "mixed".
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

NAME = re.compile(r"^seqs(\d+)_(\d+x\d+)_r(\d+)\.json$")
BASELINE_SEQS = 256
BASE_WORKLOAD, LONG_OUTPUT, SHORT_INPUT = "512x128", "512x512", "128x512"
H1_MIN_SHRINK_POINTS, H2_FLAT_POINTS = 5.0, 3.0
BUSY_SAME_POINTS, BUSY_IDLE_POINTS = 0.05, 0.10


def parse_name(name: str) -> tuple[int, str, int]:
    match = NAME.match(name)
    if not match:
        raise ValueError(f"not a seqs-isolation result name: {name}")
    return int(match.group(1)), match.group(2), int(match.group(3))


def _load(directory: Path) -> tuple[dict[tuple[int, str], list[dict]], list[str]]:
    good: dict[tuple[int, str], list[dict]] = defaultdict(list)
    bad = []
    for path in sorted(Path(directory).glob("seqs*_r*.json")):
        try:
            seqs, workload, _rep = parse_name(path.name)
        except ValueError:
            continue
        data = json.loads(path.read_text())
        if data.get("failed", 0) or not data.get("completed"):
            bad.append(path.name)
            continue
        good[(seqs, workload)].append(data)
    return good, bad


def rejected(directory: Path) -> list[str]:
    return _load(directory)[1]


def summarize(directory: Path) -> list[dict]:
    good, _bad = _load(directory)
    means = {key: float(np.mean([r["output_throughput"] for r in runs])) for key, runs in good.items()}
    rows = []
    for (seqs, workload), runs in sorted(good.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        tok = [r["output_throughput"] for r in runs]
        base = means.get((BASELINE_SEQS, workload))
        rows.append({"seqs": seqs, "workload": workload, "n": len(runs), "mean_tok_s": means[(seqs, workload)],
                     "min_tok_s": min(tok), "max_tok_s": max(tok),
                     "loss_pct_vs_256": None if base is None else (means[(seqs, workload)] / base - 1) * 100,
                     "median_itl_ms": float(np.mean([r["median_itl_ms"] for r in runs])),
                     "p99_itl_ms": float(np.mean([r["p99_itl_ms"] for r in runs]))})
    return rows


def verdict(losses: dict[str, float], busy_256: float | None, busy_1024: float | None) -> dict:
    """`losses` maps workload -> signed percent change of the 1024 arm vs the 256 arm (negative = loss)."""
    base = losses.get(BASE_WORKLOAD)
    others = {w: v for w, v in losses.items() if w != BASE_WORKLOAD}
    if base is None or not others:
        return {"supported": "undetermined", "reason": "need the baseline workload and at least one other"}
    idle_gap = None if busy_256 is None or busy_1024 is None else busy_256 - busy_1024
    flat = all(abs(v - base) <= H2_FLAT_POINTS for v in others.values())
    if idle_gap is not None and idle_gap >= BUSY_IDLE_POINTS:
        return {"supported": "H2", "reason": f"the 1024 arm leaves the GPU idle {idle_gap:.0%} more of the time"}
    if flat:
        return {"supported": "H2", "reason": f"the loss stays within {H2_FLAT_POINTS:g} points across workloads"}
    shrinks = all(w in losses and losses[w] - base >= H1_MIN_SHRINK_POINTS for w in (LONG_OUTPUT, SHORT_INPUT))
    if shrinks and idle_gap is not None and abs(idle_gap) <= BUSY_SAME_POINTS:
        return {"supported": "H1", "reason": "the loss shrinks with fewer mixed steps and the GPU busy fraction is the same"}
    if shrinks and idle_gap is None:
        return {"supported": "undetermined", "reason": "the loss shrinks as H1 predicts, but busy fractions are missing"}
    return {"supported": "mixed", "reason": "neither rule is met"}


def _table(rows: list[dict]) -> str:
    lines = ["| workload | seqs | n | mean tok/s | min-max | vs 256 | p50 ITL ms | p99 ITL ms |", "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        loss = "" if r["loss_pct_vs_256"] is None else f"{r['loss_pct_vs_256']:+.1f}%"
        lines.append(f"| {r['workload']} | {r['seqs']} | {r['n']} | {r['mean_tok_s']:,.0f} | {r['min_tok_s']:,.0f}-{r['max_tok_s']:,.0f} | "
                     f"{loss} | {r['median_itl_ms']:.1f} | {r['p99_itl_ms']:.0f} |")
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("directory", type=Path)
    p.add_argument("--busy-256", type=float)
    p.add_argument("--busy-1024", type=float)
    args = p.parse_args(argv)
    rows = summarize(args.directory)
    print(_table(rows))
    bad = rejected(args.directory)
    if bad:
        print(f"\nrejected (failed requests): {', '.join(bad)}", file=sys.stderr)
    losses = {r["workload"]: r["loss_pct_vs_256"] for r in rows if r["seqs"] == 1024 and r["loss_pct_vs_256"] is not None}
    print("\n" + json.dumps(verdict(losses, args.busy_256, args.busy_1024), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
