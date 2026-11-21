"""Device-agnostic memory counters.

CUDA  : torch.cuda.memory_allocated / memory_reserved / max_memory_allocated.
MPS   : torch.mps.current_allocated_memory (live tensors) and driver_allocated_memory
        (what the Metal driver holds for the process, i.e. the closest analogue of
        "reserved"). MPS has no native peak counter, so the peak is tracked by sampling.
CPU   : process RSS for both counters (there is no caching allocator to observe).

Peak on MPS/CPU is therefore a *sampled* peak: it can miss a spike between two
`snapshot()` calls, whereas CUDA's `max_memory_allocated` cannot.
"""
from __future__ import annotations

import gc
from dataclasses import dataclass

import psutil
import torch

DEVICE_PREFERENCE = ("cuda", "mps", "cpu")


@dataclass(frozen=True)
class MemorySnapshot:
    """Bytes of live tensors, bytes held by the allocator/driver, and peak live bytes."""

    allocated: int
    reserved: int
    peak_allocated: int


def _is_available(device: str) -> bool:
    if device == "cuda":
        return torch.cuda.is_available()
    if device == "mps":
        return torch.backends.mps.is_available()
    return device == "cpu"


def pick_device(preferred: str | None = None) -> str:
    """Return `preferred` if usable, else the best available of cuda > mps > cpu."""
    if preferred is not None:
        if preferred not in DEVICE_PREFERENCE:
            raise ValueError(f"unknown device {preferred!r}; choose from {DEVICE_PREFERENCE}")
        if not _is_available(preferred):
            raise RuntimeError(f"device {preferred!r} is not available on this machine")
        return preferred
    return next(d for d in DEVICE_PREFERENCE if _is_available(d))


def sync(device: str) -> None:
    """Block until queued device work finishes so memory counters are accurate."""
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


class MemoryProbe:
    """Uniform view of allocated / reserved / peak memory on one device."""

    def __init__(self, device: str) -> None:
        if device not in DEVICE_PREFERENCE:
            raise ValueError(f"unknown device {device!r}")
        self.device = device
        self._peak = 0
        self.reset_peak()

    @property
    def has_reserved_counter(self) -> bool:
        """False on CPU, where 'reserved' is just RSS and carries no allocator meaning."""
        return self.device != "cpu"

    def _read(self) -> tuple[int, int]:
        if self.device == "cuda":
            return torch.cuda.memory_allocated(), torch.cuda.memory_reserved()
        if self.device == "mps":
            return torch.mps.current_allocated_memory(), torch.mps.driver_allocated_memory()
        rss = psutil.Process().memory_info().rss
        return rss, rss

    def reset_peak(self) -> None:
        if self.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        self._peak = self._read()[0]

    def snapshot(self) -> MemorySnapshot:
        sync(self.device)
        allocated, reserved = self._read()
        self._peak = max(self._peak, allocated)
        peak = torch.cuda.max_memory_allocated() if self.device == "cuda" else self._peak
        return MemorySnapshot(allocated, reserved, max(peak, allocated))

    def empty_cache(self) -> None:
        """Drop Python garbage, then hand cached-but-unused memory back to the driver."""
        gc.collect()
        sync(self.device)
        if self.device == "cuda":
            torch.cuda.empty_cache()
        elif self.device == "mps":
            torch.mps.empty_cache()
