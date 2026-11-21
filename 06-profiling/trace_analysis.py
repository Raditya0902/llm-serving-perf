"""Analyse a PyTorch Chrome trace: device idle gaps and allocator events.

Gap analysis needs *device* events (CUDA kernels). On MPS/CPU the PyTorch profiler records
only CPU-side operators, so the report says `available=False` rather than inventing numbers.
Use `launch_probe.py` for the CPU-vs-device question on those backends.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from pathlib import Path

DEVICE_CATEGORIES = ("kernel", "gpu_memcpy", "gpu_memset")
LARGEST_GAPS_KEPT = 5
CUDA_DEVICE_TYPE = 1  # c10::DeviceType::CUDA in the profiler's [memory] events


@dataclass(frozen=True)
class GapReport:
    available: bool
    reason: str
    n_kernels: int = 0
    window_us: float = 0.0
    busy_fraction: float | None = None
    gaps_us: tuple[float, ...] = ()
    gap_p50_us: float = 0.0
    gap_p95_us: float = 0.0
    gap_max_us: float = 0.0


@dataclass(frozen=True)
class AllocSummary:
    n_allocs: int
    n_frees: int
    bytes_allocated: int
    bytes_freed: int
    n_device_allocs: int = 0  # allocations on a CUDA device (Device Type 1), excluding host ones


def load_trace(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def _percentile(sorted_values: list[float], q: float) -> float:
    """Linear interpolation, the same definition as measure.percentile (q in 0..1)."""
    if not sorted_values:
        return 0.0
    position = q * (len(sorted_values) - 1)
    low = int(position)
    high = min(low + 1, len(sorted_values) - 1)
    return sorted_values[low] + (sorted_values[high] - sorted_values[low]) * (position - low)


def _merge(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    merged: list[tuple[float, float]] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _busiest_device(events: list[dict]) -> list[dict]:
    counts: dict = {}
    for event in events:
        key = event.get("args", {}).get("device")
        counts[key] = counts.get(key, 0) + 1
    if len(counts) <= 1:
        return events
    busiest = max(counts, key=counts.get)
    return [e for e in events if e.get("args", {}).get("device") == busiest]


def analyze_gaps(trace: dict) -> GapReport:
    events = [
        e for e in trace.get("traceEvents", [])
        if e.get("ph") == "X" and e.get("cat") in DEVICE_CATEGORIES and "dur" in e
    ]
    events = _busiest_device(events)  # intervals from other GPUs must not hide this GPU's idle gaps
    if not events:
        return GapReport(
            available=False,
            reason="no device kernel events in trace (MPS/CPU record CPU operators only)",
        )
    merged = _merge([(e["ts"], e["ts"] + e["dur"]) for e in events])
    window = merged[-1][1] - merged[0][0]
    busy = sum(end - start for start, end in merged)
    gaps = tuple(merged[i + 1][0] - merged[i][1] for i in range(len(merged) - 1))
    ordered = sorted(gaps)
    return GapReport(
        available=True,
        reason="",
        n_kernels=sum(1 for e in events if e.get("cat") == "kernel"),
        window_us=window,
        busy_fraction=busy / window if window > 0 else 1.0,
        gaps_us=gaps,
        gap_p50_us=statistics.median(ordered) if ordered else 0.0,
        gap_p95_us=_percentile(ordered, 0.95),
        gap_max_us=ordered[-1] if ordered else 0.0,
    )


def largest_gaps(report: GapReport, n: int = LARGEST_GAPS_KEPT) -> list[float]:
    return sorted(report.gaps_us, reverse=True)[:n]


def summarize_alloc_events(trace: dict) -> AllocSummary:
    """Count `[memory]` instant events (recorded with `profile_memory=True`)."""
    allocated = freed = n_allocs = n_frees = n_device = 0
    for event in trace.get("traceEvents", []):
        if event.get("name") != "[memory]":
            continue
        size = int(event.get("args", {}).get("Bytes", 0))
        if size > 0:
            n_allocs += 1
            allocated += size
            n_device += event.get("args", {}).get("Device Type") == CUDA_DEVICE_TYPE
        elif size < 0:
            n_frees += 1
            freed += -size
    return AllocSummary(n_allocs, n_frees, allocated, freed, n_device)
