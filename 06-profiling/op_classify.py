"""Group profiler operators into coarse categories and total them.

All totals use *self* time (time inside the op excluding children). Summing total
time would count nested ops (`aten::linear` -> `aten::addmm`) twice.

Projections (q/k/v/o, MLP, lm_head) land in `matmul`; `attention` is the attention
core only (SDPA / softmax). They are kept apart so "time in matmuls" and "time in
attention" are separate, unambiguous numbers.

CAVEAT: this is only true for `sdpa`. With `attn_impl=eager`, QK^T and PV run as plain
`aten::matmul/bmm`, so they count as `matmul` here; never compare category *shares* across
attention implementations. Use module_timing's bracketed attribution for that.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

CATEGORIES = ("matmul", "attention", "elementwise", "memory", "sync", "other")
ANNOTATION = "annotation"
ANNOTATION_PREFIXES = ("d6::", "ProfilerStep", "Activity Buffer")

_MATMUL_MARKERS = ("::mm", "::addmm", "::bmm", "::baddbmm", "::matmul", "::linear", "::mv", "::einsum")
_ATTENTION_MARKERS = ("scaled_dot_product", "softmax")
_SYNC_MARKERS = ("_local_scalar_dense", "::item", "synchronize")
_MEMORY_MARKERS = (
    "::cat", "::copy_", "::to", "::_to_copy", "::clone", "::contiguous", "::empty",
    "::zeros", "::index_select", "::index_put", "::index_fill", "::slice", "::narrow",
)


@dataclass(frozen=True)
class OpRow:
    name: str
    count: int
    self_cpu_us: float
    self_device_us: float
    total_cpu_us: float


@dataclass(frozen=True)
class CategoryTotal:
    category: str
    count: int
    self_cpu_us: float
    self_device_us: float
    share: float  # fraction of all non-annotation time (device time if requested, else CPU)


def classify(name: str) -> str:
    if name.startswith(ANNOTATION_PREFIXES):
        return ANNOTATION
    if not name.startswith("aten::"):
        return "other"
    if any(m in name for m in _SYNC_MARKERS):
        return "sync"
    if any(m in name for m in _ATTENTION_MARKERS):
        return "attention"
    if any(name.endswith(m) or m + "_" in name for m in _MATMUL_MARKERS):
        return "matmul"
    if any(name.endswith(m) or name.startswith("aten" + m) for m in _MEMORY_MARKERS):
        return "memory"
    return "elementwise"


def rows_from_events(events: Iterable) -> list[OpRow]:
    """Convert `prof.key_averages()` entries into immutable rows."""
    return [
        OpRow(
            name=e.key,
            count=int(e.count),
            self_cpu_us=float(e.self_cpu_time_total),
            self_device_us=float(e.self_device_time_total),
            total_cpu_us=float(e.cpu_time_total),
        )
        for e in events
    ]


def category_totals(
    rows: Sequence[OpRow], use_device_time: bool = False
) -> dict[str, CategoryTotal]:
    """Per-category self-time totals with shares of the grand total (annotations excluded)."""
    sums: dict[str, list[float]] = {}
    for row in rows:
        category = classify(row.name)
        if category == ANNOTATION:
            continue
        acc = sums.setdefault(category, [0, 0.0, 0.0])
        acc[0] += row.count
        acc[1] += row.self_cpu_us
        acc[2] += row.self_device_us
    key = 2 if use_device_time else 1
    grand = sum(acc[key] for acc in sums.values())
    if grand <= 0:
        return {}
    return {
        cat: CategoryTotal(cat, int(acc[0]), acc[1], acc[2], acc[key] / grand)
        for cat, acc in sums.items()
    }


def top_ops(rows: Sequence[OpRow], n: int, use_device_time: bool = False) -> list[OpRow]:
    real = [r for r in rows if classify(r.name) != ANNOTATION]
    key = (lambda r: r.self_device_us) if use_device_time else (lambda r: r.self_cpu_us)
    return sorted(real, key=key, reverse=True)[:n]
