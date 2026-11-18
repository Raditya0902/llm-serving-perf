"""Peak GPU memory while a command runs (Day 4's sweep has no memory metric).

    python gpu/gpu_mem_sampler.py --out mem.json --interval 0.5 -- vllm bench serve ...

Polls `nvidia-smi --query-gpu=memory.used` until the command exits. NOTE: this is whole-GPU
memory (all processes), so run on an otherwise idle GPU; it includes the CUDA context and
vLLM's pre-allocated KV pool, i.e. it reflects `gpu_memory_utilization`, not live tensors.
Never run on the dev Mac (no NVIDIA GPU); tested with stand-ins only.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

QUERY = ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]
DEFAULT_INTERVAL_S = 0.5


@dataclass(frozen=True)
class PeakMemory:
    peak_mib: int
    n_samples: int
    gpu_index: int


def parse_memory_used(text: str) -> list[int]:
    """One MiB integer per GPU from `--format=csv,noheader,nounits` output."""
    values = []
    for line in text.strip().splitlines():
        try:
            values.append(int(line.strip()))
        except ValueError as error:
            raise ValueError(f"unexpected nvidia-smi line: {line!r}") from error
    if not values:
        raise ValueError("nvidia-smi returned no GPUs")
    return values


def sample_peak(
    query: Callable[[], str],
    is_running: Callable[[], bool],
    gpu_index: int = 0,
    interval_s: float = DEFAULT_INTERVAL_S,
    sleep: Callable[[float], None] = time.sleep,
) -> PeakMemory:
    """Poll while `is_running()`, plus one final sample, and return the maximum seen."""
    peak, count = 0, 0
    while True:
        running = is_running()
        used = parse_memory_used(query())
        if gpu_index >= len(used):
            raise ValueError(f"gpu_index {gpu_index} but only {len(used)} GPU(s)")
        peak, count = max(peak, used[gpu_index]), count + 1
        if not running:
            return PeakMemory(peak, count, gpu_index)
        sleep(interval_s)


def _nvidia_smi() -> str:
    return subprocess.run(QUERY, capture_output=True, text=True, check=True, timeout=10).stdout


def _terminate(process: subprocess.Popen, grace_s: float = 10.0) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=grace_s)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=DEFAULT_INTERVAL_S)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("give a command after --")
    if args.interval <= 0:
        parser.error("--interval must be > 0")
    process = subprocess.Popen(command)
    try:
        peak = sample_peak(_nvidia_smi, lambda: process.poll() is None, args.gpu, args.interval)
    except BaseException:
        _terminate(process)  # never leave the benchmark running unobserved (and holding GPU memory)
        raise
    args.out.write_text(json.dumps({**asdict(peak), "command": command, "returncode": process.returncode}))
    return process.returncode or 0


if __name__ == "__main__":
    sys.exit(main())
