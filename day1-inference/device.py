"""Device selection, synchronisation and memory tracking (CUDA / MPS / CPU)."""

from __future__ import annotations

import gc
import os

import psutil
import torch

SUPPORTED_DEVICES = ("cuda", "mps", "cpu")


def pick_device(preferred: str | None = None) -> torch.device:
    """Return the requested device, or the best available (cuda > mps > cpu)."""
    if preferred is not None:
        if preferred not in SUPPORTED_DEVICES:
            raise ValueError(f"unsupported device {preferred!r}; expected one of {SUPPORTED_DEVICES}")
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sync(device: torch.device) -> None:
    """Block until all queued work on `device` finishes.

    GPU kernels are launched asynchronously, so a wall-clock timer read without
    this call measures only the launch time, not the execution time.
    """
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def release_cached_memory(device: torch.device) -> None:
    """Free allocator-cached blocks so one configuration's peak doesn't leak into the next."""
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


class MemoryTracker:
    """Tracks peak memory use of the current process on one device.

    CUDA exposes a true peak counter. MPS and CPU do not, so the peak is the
    running max of samples taken via `sample()` (call it once per step).
    """

    def __init__(self, device: torch.device) -> None:
        self._device = device
        self._peak = 0

    def reset(self) -> None:
        if self._device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self._device)
        self._peak = self._current_bytes()

    def sample(self) -> None:
        self._peak = max(self._peak, self._current_bytes())

    @property
    def peak_bytes(self) -> int:
        if self._device.type == "cuda":
            return int(torch.cuda.max_memory_allocated(self._device))
        return self._peak

    def _current_bytes(self) -> int:
        if self._device.type == "cuda":
            return int(torch.cuda.memory_allocated(self._device))
        if self._device.type == "mps":
            # Live tensor bytes. (driver_allocated_memory is a grow-only pool that sits
            # at its high-water mark after warmup, so it cannot distinguish configs.)
            return int(torch.mps.current_allocated_memory())
        # CPU RSS never shrinks reliably (malloc keeps freed pages), so it is only a
        # coarse upper bound and is not comparable across configurations.
        return int(psutil.Process(os.getpid()).memory_info().rss)
