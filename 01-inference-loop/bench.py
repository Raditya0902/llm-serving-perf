"""Tiny timing harness: warmup runs are discarded, remaining runs are summarised."""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Callable, Sequence, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class Stats:
    median: float
    minimum: float
    maximum: float


def summarize(values: Sequence[float]) -> Stats:
    if not values:
        raise ValueError("cannot summarise an empty sequence")
    return Stats(
        median=statistics.median(values),
        minimum=min(values),
        maximum=max(values),
    )


def run_repeated(fn: Callable[[], T], warmup: int, repeats: int) -> tuple[T, ...]:
    """Call `fn` `warmup` times (discarded), then `repeats` times, returning those results.

    Warmup absorbs one-off costs: kernel compilation, allocator growth, caches.
    `fn` is responsible for synchronising the device around what it times.
    """
    if warmup < 0 or repeats < 1:
        raise ValueError("warmup must be >= 0 and repeats >= 1")
    for _ in range(warmup):
        fn()
    return tuple(fn() for _ in range(repeats))
