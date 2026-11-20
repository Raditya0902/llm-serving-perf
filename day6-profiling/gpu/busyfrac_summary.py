"""Summarize gpu/run_busyfrac.sh output: busy fraction (nsys window) and benchmark throughput per backend and seq limit.

    python -m gpu.busyfrac_summary OUT_DIR

Reads `<backend>_s<seqs>_<IN>x<OUT>_d<delay>.busy.json` (one nsys_busy record) and the matching bench result for output tok/s.
Per (backend, workload) it reports the busy fraction at each seq limit (mean over windows), the change from the low to the high limit,
and the median idle gap. Single-GPU runs only: the first device in the file is used.
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

NAME = re.compile(r"^(?P<backend>[A-Z_]+(?:-fa\d)?)_s(?P<seqs>\d+)_(?P<workload>\d+x\d+)_d(?P<delay>\d+)\.busy\.json$")


def _first_device(record: dict) -> dict | None:
    devices = record.get("devices") or {}
    return devices[sorted(devices)[0]] if devices else None


def gather(root: Path) -> list[dict]:
    rows = []
    for path in sorted(root.glob("*.busy.json")):
        match = NAME.match(path.name)
        if not match:
            continue
        device = _first_device(json.loads(path.read_text()))
        if device is None or device.get("busy_fraction") is None:
            continue
        bench = root / path.name.replace(".busy.json", ".json")
        tok_s = json.loads(bench.read_text()).get("output_throughput") if bench.exists() else None
        rows.append({**match.groupdict(), "seqs": int(match["seqs"]), "delay": int(match["delay"]),
                     "busy": device["busy_fraction"], "gap_p50_us": device.get("gap_p50_us"),
                     "gap_p95_us": device.get("gap_p95_us"), "tok_s": tok_s})
    return rows


def summarize(root: Path) -> dict:
    cells: dict[tuple, list[dict]] = defaultdict(list)
    for row in gather(root):
        cells[(row["backend"], row["workload"], row["seqs"])].append(row)
    table = [{"backend": b, "workload": w, "seqs": s, "windows": len(rows),
              "busy_fraction": mean(r["busy"] for r in rows),
              "busy_min": min(r["busy"] for r in rows), "busy_max": max(r["busy"] for r in rows),
              "gap_p50_us": mean(r["gap_p50_us"] for r in rows if r["gap_p50_us"] is not None) if any(r["gap_p50_us"] is not None for r in rows) else None,
              "tok_s": mean(r["tok_s"] for r in rows if r["tok_s"]) if any(r["tok_s"] for r in rows) else None}
             for (b, w, s), rows in sorted(cells.items())]
    drops = []
    for backend, workload in sorted({(r["backend"], r["workload"]) for r in table}):
        arm = sorted((r for r in table if (r["backend"], r["workload"]) == (backend, workload)), key=lambda r: r["seqs"])
        if len(arm) >= 2:
            drops.append({"backend": backend, "workload": workload, "low_seqs": arm[0]["seqs"], "high_seqs": arm[-1]["seqs"],
                          "busy_change_points": 100 * (arm[-1]["busy_fraction"] - arm[0]["busy_fraction"])})
    return {"cells": table, "busy_change_low_to_high_seqs": drops}


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print(__doc__, file=sys.stderr)
        return 2
    print(json.dumps(summarize(Path(argv[0])), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
