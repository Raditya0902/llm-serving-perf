"""GPU busy fraction and idle-gap statistics from an Nsight Systems trace (nsys export --type sqlite).

    nsys export --type sqlite --force-overwrite true -o run.sqlite run.nsys-rep
    python -m gpu.nsys_busy run.sqlite [more.sqlite ...]        # one JSON object per file

Busy fraction = length of the union of all kernel / memcpy / memset intervals on a device divided by the span from the first
interval start to the last interval end. The trace holds only the captured window (vLLM's cuda profiler start/stop), so the span is
the window. Several devices are reported separately. Gaps are the idle stretches between merged intervals.
"""
from __future__ import annotations

import json
import sqlite3
import sys
from collections import defaultdict
from collections.abc import Iterable, Sequence

Interval = tuple[int, int]  # nanoseconds
TABLES = ("CUPTI_ACTIVITY_KIND_KERNEL", "CUPTI_ACTIVITY_KIND_MEMCPY", "CUPTI_ACTIVITY_KIND_MEMSET")


def merge_intervals(intervals: Iterable[Interval]) -> list[Interval]:
    """Union of intervals, sorted, with touching or overlapping ones joined. Zero-length and inverted intervals are dropped."""
    merged: list[Interval] = []
    for start, end in sorted(i for i in intervals if i[1] > i[0]):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _percentile(sorted_values: Sequence[int], q: float) -> float:
    if not sorted_values:
        return 0.0
    rank = q / 100 * (len(sorted_values) - 1)
    low = int(rank)
    high = min(low + 1, len(sorted_values) - 1)
    return float(sorted_values[low] + (sorted_values[high] - sorted_values[low]) * (rank - low))


def busy_stats(intervals: Iterable[Interval]) -> dict:
    """Busy fraction and gap percentiles for one device. `busy_fraction` is None with no activity or a zero-length span."""
    merged = merge_intervals(intervals)
    if not merged:
        return {"busy_fraction": None, "span_ms": 0.0, "busy_ms": 0.0, "intervals": 0, "gaps": 0}
    span = merged[-1][1] - merged[0][0]
    busy = sum(end - start for start, end in merged)
    gaps = sorted(merged[i + 1][0] - merged[i][1] for i in range(len(merged) - 1))
    return {
        "busy_fraction": busy / span if span > 0 else None,
        "span_ms": span / 1e6,
        "busy_ms": busy / 1e6,
        "intervals": len(merged),
        "gaps": len(gaps),
        "gap_p50_us": _percentile(gaps, 50) / 1e3,
        "gap_p95_us": _percentile(gaps, 95) / 1e3,
        "gap_max_us": (gaps[-1] / 1e3) if gaps else 0.0,
    }


def load_by_device(path: str) -> dict[int, list[Interval]]:
    """Read (start, end) per deviceId from the kernel, memcpy and memset tables that exist in the file."""
    by_device: dict[int, list[Interval]] = defaultdict(list)
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as db:
        present = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        found = [t for t in TABLES if t in present]
        if "CUPTI_ACTIVITY_KIND_KERNEL" not in found:
            raise ValueError(f"{path}: no kernel table (was the capture window empty?)")
        for table in found:
            for device, start, end in db.execute(f"SELECT deviceId, start, end FROM {table}"):
                by_device[int(device)].append((int(start), int(end)))
    return dict(by_device)


def summarize_file(path: str) -> dict:
    return {"file": path, "devices": {str(d): busy_stats(iv) for d, iv in sorted(load_by_device(path).items())}}


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__, file=sys.stderr)
        return 2
    status = 0
    for path in argv:
        try:
            print(json.dumps(summarize_file(path)))
        except (ValueError, sqlite3.Error, OSError) as error:
            print(json.dumps({"file": path, "error": str(error)}))
            status = 1
    return status


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
