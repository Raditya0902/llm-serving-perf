"""Peak GPU memory while a run executes (``nvidia-smi``).

Whole-GPU ``memory.used`` for one device: it includes the CUDA context and vLLM's pre-allocated
KV pool, so it mostly reflects ``gpu_memory_utilization``, not live tensors. Run on an otherwise
idle GPU. Never exercised on the dev Mac (no NVIDIA GPU); tested with stand-in queries only.
Same approach as Day 6's ``gpu/gpu_mem_sampler.py`` (a different git branch), re-implemented as
a background sampler so it can run alongside the server and the benchmark.
"""
from __future__ import annotations

import subprocess
import threading
from collections.abc import Callable
from dataclasses import dataclass

from periodic import PeriodicThread

QUERY_CMD = ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]
QUERY_TIMEOUT_S = 10
DEFAULT_INTERVAL_S = 0.5


@dataclass(frozen=True)
class PeakMemory:
    peak_mib: int | None  # None: nothing was sampled, or the sampler failed ("not measured")
    n_samples: int
    errors: int
    gpu_index: int
    thread_error: str | None = None


def parse_memory_used(text: str) -> list[int]:
    """One MiB integer per GPU from ``--format=csv,noheader,nounits`` output."""
    values = []
    for line in text.strip().splitlines():
        try:
            values.append(int(line.strip()))
        except ValueError as error:
            raise ValueError(f"unexpected nvidia-smi line: {line!r}") from error
    if not values:
        raise ValueError("nvidia-smi returned no GPUs")
    return values


def nvidia_smi() -> str:
    return subprocess.run(
        QUERY_CMD, capture_output=True, text=True, check=True, timeout=QUERY_TIMEOUT_S,
    ).stdout


class PeakMemorySampler:
    def __init__(
        self, query: Callable[[], str] = nvidia_smi, *, gpu_index: int = 0,
        interval_s: float = DEFAULT_INTERVAL_S,
    ) -> None:
        if gpu_index < 0:
            raise ValueError(f"gpu_index must be >= 0, got {gpu_index}")
        self._query, self._gpu_index = query, gpu_index
        self._thread = PeriodicThread(self.sample_once, interval_s, "gpu-mem-sampler")
        self._lock = threading.Lock()
        self._peak: int | None = None
        self._samples = self._errors = 0

    def sample_once(self) -> None:
        try:
            used = parse_memory_used(self._query())
        except (OSError, subprocess.SubprocessError, ValueError):
            with self._lock:
                self._errors += 1
            return
        if self._gpu_index >= len(used):
            raise ValueError(f"gpu_index {self._gpu_index} but only {len(used)} GPU(s)")
        with self._lock:
            self._peak = max(used[self._gpu_index], self._peak or 0)
            self._samples += 1

    def result(self) -> PeakMemory:
        with self._lock:
            if self._thread.error is not None:  # a partial peak from before the failure is not a measurement
                err = f"{type(self._thread.error).__name__}: {self._thread.error}"
                return PeakMemory(None, self._samples, self._errors, self._gpu_index, err)
            return PeakMemory(self._peak, self._samples, self._errors, self._gpu_index)

    def __enter__(self) -> PeakMemorySampler:
        self._thread.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self._thread.stop()
