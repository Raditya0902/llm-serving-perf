"""All-reduce share of GPU time from an Nsight Systems capture (RUNBOOK 3a).

    nsys stats --report cuda_gpu_kern_sum t.nsys-rep      # also writes t.sqlite next to it
    python analyze_nsys.py t.sqlite > t.json

The share is all-reduce kernel time divided by the union of kernel-busy time on each GPU (overlapping kernels on
different streams are counted once in the denominator). `busy_fraction_of_window` is busy time over the captured
span, so the gaps (launch-bound idling) show up next to the communication share. Capture with
`--cuda-graph-trace=node`, otherwise kernels inside CUDA graphs (the whole decode step) are invisible and the output says so.
Kernel time of an all-reduce includes waiting for the slower peer, so the share is an upper bound on pure communication. vLLM's custom all-reduce shows up as
`cross_device_reduce_*`, NCCL's as `ncclDevKernel_AllReduce*`; other NCCL kernels are reported separately.
"""
from __future__ import annotations

import json
import re
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

ALLREDUCE = re.compile(r"cross_device_reduce|ncclDevKernel_AllReduce|ncclKernel_AllReduce", re.IGNORECASE)
NCCL_OTHER = re.compile(r"nccl", re.IGNORECASE)
SAMPLER = re.compile(r"SamplingFromProbKernel")  # one launch per decode step per GPU (a step-count proxy)
TOP_N = 8
QUERY = (
    "SELECT k.deviceId, k.start, k.end, s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k "
    "JOIN StringIds s ON s.id = k.demangledName ORDER BY k.deviceId, k.start"
)


def _has_column(con: sqlite3.Connection, table: str, column: str) -> bool:
    return any(row[1] == column for row in con.execute(f"PRAGMA table_info({table})"))


def _graph_counts(con: sqlite3.Connection) -> tuple[int, int]:
    """(graph launches recorded, kernels attributed to graph nodes). Default nsys (`--cuda-graph-trace=graph`) records
    each graph launch as one opaque entry, so a CUDA-graph decode loop shows ~no kernels."""
    tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    launches = (con.execute("SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_GRAPH_TRACE").fetchone()[0]
                if "CUPTI_ACTIVITY_KIND_GRAPH_TRACE" in tables else 0)
    in_graph = (con.execute("SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE graphId IS NOT NULL AND graphId > 0")
                .fetchone()[0] if _has_column(con, "CUPTI_ACTIVITY_KIND_KERNEL", "graphId") else 0)
    return launches, in_graph


def classify(name: str) -> str:
    if ALLREDUCE.search(name):
        return "allreduce"
    return "nccl_other" if NCCL_OTHER.search(name) else "other"


def _union_ns(intervals: list[tuple[int, int]]) -> int:
    total, cur_end = 0, None
    for start, end in sorted(intervals):
        if cur_end is None or start > cur_end:
            total += end - start
            cur_end = end
        elif end > cur_end:
            total += end - cur_end
            cur_end = end
    return total


def summarize(db_path: Path) -> dict:
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        rows = con.execute(QUERY).fetchall()
        graph_launches, graph_kernels = _graph_counts(con)
    finally:
        con.close()
    spans: dict[int, list[tuple[int, int]]] = defaultdict(list)
    per_class: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    counts: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    kernels: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for device, start, end, name in rows:
        spans[device].append((start, end))
        kind = classify(name)
        per_class[device][kind] += end - start
        counts[device][kind] += 1
        counts[device]["sampler"] += bool(SAMPLER.search(name))
        kernels[name][0] += end - start
        kernels[name][1] += 1
    per_device = {}
    for device, intervals in spans.items():
        busy = _union_ns(intervals)
        window = max(e for _, e in intervals) - min(s for s, _ in intervals)
        classes = per_class[device]
        per_device[str(device)] = {
            "kernel_instances": len(intervals), "busy_ns": busy, "window_ns": window,
            "busy_fraction_of_window": busy / window if window else None,
            "allreduce_ns": classes["allreduce"], "nccl_other_ns": classes["nccl_other"],
            "allreduce_instances": counts[device]["allreduce"], "sampler_launches": counts[device]["sampler"],
            "allreduce_share_of_busy": classes["allreduce"] / busy if busy else None,
        }
    busy_total = sum(d["busy_ns"] for d in per_device.values())
    allreduce_total = sum(d["allreduce_ns"] for d in per_device.values())
    top = sorted(kernels.items(), key=lambda kv: kv[1][0], reverse=True)[:TOP_N]
    result = {
        "allreduce_share_of_busy": allreduce_total / busy_total if busy_total else None,
        "graph_launches": graph_launches, "kernels_from_graph_nodes": graph_kernels,
        "graph_contents_visible": graph_launches == 0 or graph_kernels > 0,
        "per_device": per_device,
        "top_kernels": [{"name": n, "total_ns": t, "instances": c} for n, (t, c) in top],
    }
    if not result["graph_contents_visible"]:
        result["warning"] = ("CUDA graph launches were recorded but their kernels were not: re-capture with "
                             "`nsys profile --cuda-graph-trace=node`; this share only covers kernels launched outside graphs")
    return result


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    print(json.dumps(summarize(Path(argv[1])), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
