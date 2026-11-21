"""Summarize `run_tp_compare.sh` output into one markdown table.

    python summarize_tp_compare.py results/raw/tp_compare

Rows are tp1 / tp2 / tp2nc (TP=2 with --disable-custom-all-reduce) / dp2 (two TP=1 replicas, throughput summed)
by concurrency, mean over repeats with the CV; TP=2 rows are compared with TP=1 at the same concurrency, dp2 with tp1.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

RUN = re.compile(r"^(tp1|tp2|tp2nc|dp2)_c(\d+)_r(\d+)(?:_([ab]))?\.json$")
BASELINE = {"tp2": "tp1", "tp2nc": "tp1", "dp2": "tp1"}


def delta(value: float, base: float) -> float:
    return (value / base - 1) * 100


def _peak_mem(path: Path) -> dict[int, int]:
    peak: dict[int, int] = {}
    for line in path.read_text().splitlines()[1:]:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3 and parts[1].isdigit():
            used = int(parts[2].split()[0])
            peak[int(parts[1])] = max(peak.get(int(parts[1]), 0), used)
    return peak


def summarize(raw: Path) -> dict[str, dict[str, float]]:
    reps: dict[tuple[str, int], dict[int, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for path in sorted(raw.glob("*.json")):
        m = RUN.match(path.name)
        if not m:
            continue
        label, conc, rep = m.group(1), int(m.group(2)), int(m.group(3))
        reps[(label, conc)][rep].append(json.loads(path.read_text()))
    table: dict[str, dict[str, float]] = {}
    for (label, conc), by_rep in sorted(reps.items()):
        runs = list(by_rep.values())
        tput = [sum(r["output_throughput"] for r in rr) for rr in runs]
        tpot = [statistics.fmean(r["mean_tpot_ms"] for r in rr) for rr in runs]
        ttft = [statistics.fmean(r["mean_ttft_ms"] for r in rr) for rr in runs]
        mean = statistics.fmean(tput)
        row = {"n": len(runs), "tput": mean, "tpot": statistics.fmean(tpot), "ttft": statistics.fmean(ttft),
               "cv": statistics.stdev(tput) / mean * 100 if len(tput) > 1 else float("nan"),
               "failed": sum(r["failed"] for rr in runs for r in rr)}
        mem = next(iter(sorted(raw.glob(f"mem_{label}_c{conc}_r*.csv"))), None)
        for gpu, used in (_peak_mem(mem).items() if mem else ()):
            row[f"mem_gpu{gpu}"] = used
        table[f"{label}_c{conc}"] = row
    return table


def render(table: Mapping[str, Mapping[str, float]]) -> str:
    lines = ["| case | n | out tok/s (CV) | vs TP=1 | mean TPOT ms | vs TP=1 | mean TTFT ms | peak MiB gpu0/gpu1 | failed |",
             "|---|---|---|---|---|---|---|---|---|"]
    for case, r in table.items():
        label, conc = case.split("_c")
        base = table.get(f"{BASELINE.get(label, '')}_c{conc}")
        d = lambda key: "-" if base is None else f"{delta(r[key], base[key]):+.1f}%"
        mem = f"{r.get('mem_gpu0', float('nan')):.0f} / {r.get('mem_gpu1', float('nan')):.0f}"
        lines.append(f"| {case} | {r['n']:.0f} | {r['tput']:.0f} ({r['cv']:.1f}%) | {d('tput')} | {r['tpot']:.2f} | "
                     f"{d('tpot')} | {r['ttft']:.0f} | {mem} | {r['failed']:.0f} |")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("raw", type=Path)
    print(render(summarize(p.parse_args(argv).raw)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
