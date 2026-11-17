"""Process-group helpers shared by the Day 5 scripts.

Two ways to get a process group:

* ``torchrun`` launches N processes and sets RANK / WORLD_SIZE / LOCAL_RANK / MASTER_*.
  Scripts call ``init_distributed()`` (or the ``distributed_session()`` context manager).
* ``spawn_group(fn, world_size)`` launches N processes from plain Python (tests and the
  benchmark use this, so no torchrun is needed). Workers get the same ``DistContext``.

Backend choice: NCCL needs one CUDA device per rank; otherwise Gloo on CPU. NCCL is
therefore *never* selected on this Mac. The NCCL branch is written but untested on real
hardware (see README).
"""
from __future__ import annotations

import os
import pickle
import socket
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable, Iterator

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

SUPPORTED_BACKENDS = ("gloo", "nccl")
LOOPBACK_ADDR = "127.0.0.1"
DEFAULT_TIMEOUT_S = 120.0


@dataclass(frozen=True)
class DistContext:
    """What a rank needs to know about its place in the group."""

    rank: int
    world_size: int
    local_rank: int
    backend: str
    device: torch.device


def resolve_backend(world_size: int, cuda_device_count: int, requested: str | None = None) -> str:
    """Pick 'nccl' when every rank can own a GPU, else 'gloo'. Explicit requests are validated."""
    if world_size < 1:
        raise ValueError(f"world_size must be >= 1, got {world_size}")
    if requested is not None and requested not in SUPPORTED_BACKENDS:
        raise ValueError(f"backend must be one of {SUPPORTED_BACKENDS}, got {requested!r}")
    has_enough_gpus = cuda_device_count >= world_size
    if requested == "nccl" and not has_enough_gpus:
        raise ValueError(
            f"nccl needs one CUDA device per rank: world_size={world_size}, "
            f"visible devices={cuda_device_count}"
        )
    if requested is not None:
        return requested
    return "nccl" if has_enough_gpus else "gloo"


def find_free_port() -> int:
    """Ask the OS for an unused loopback TCP port (bound then released)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((LOOPBACK_ADDR, 0))
        return int(sock.getsockname()[1])


def loopback_interface() -> str:
    """Name of the loopback network interface on this OS ('lo0' on macOS, 'lo' on Linux)."""
    return "lo0" if sys.platform == "darwin" else "lo"


def _configure_loopback_env() -> None:
    """Point Gloo at loopback when the rendezvous is local.

    Without this, Gloo resolves the machine hostname, which hangs on hosts (e.g. this Mac)
    whose hostname does not resolve.
    """
    if os.environ.get("MASTER_ADDR") in (LOOPBACK_ADDR, "localhost"):
        os.environ.setdefault("GLOO_SOCKET_IFNAME", loopback_interface())


def init_distributed(
    backend: str | None = None, timeout_s: float = DEFAULT_TIMEOUT_S
) -> DistContext:
    """Initialise the default process group from torchrun-style environment variables."""
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    chosen = resolve_backend(world_size, torch.cuda.device_count(), backend)

    _configure_loopback_env()
    if chosen == "nccl":
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    dist.init_process_group(
        backend=chosen, rank=rank, world_size=world_size, timeout=timedelta(seconds=timeout_s)
    )
    return DistContext(rank, world_size, local_rank, chosen, device)


def cleanup() -> None:
    """Synchronise then tear the process group down. Safe to call twice."""
    if not (dist.is_available() and dist.is_initialized()):
        return
    try:
        dist.barrier()
    finally:  # a dead peer makes the barrier raise; still release the group
        dist.destroy_process_group()


@contextmanager
def distributed_session(
    backend: str | None = None, timeout_s: float = DEFAULT_TIMEOUT_S
) -> Iterator[DistContext]:
    """Init on entry; destroy the group on exit, even if the body raises."""
    ctx = init_distributed(backend, timeout_s)
    try:
        yield ctx
    except BaseException:
        # A peer may already be gone, so skip the barrier: just release local resources.
        if dist.is_initialized():
            dist.destroy_process_group()
        raise
    else:
        cleanup()


def _worker(
    local_rank: int,
    fn: Callable[..., Any],
    world_size: int,
    port: int,
    backend: str,
    timeout_s: float,
    out_dir: str,
    args: tuple[Any, ...],
) -> None:
    os.environ.update(
        MASTER_ADDR=LOOPBACK_ADDR,
        MASTER_PORT=str(port),
        RANK=str(local_rank),
        WORLD_SIZE=str(world_size),
        LOCAL_RANK=str(local_rank),
    )
    with distributed_session(backend, timeout_s) as ctx:
        result = fn(ctx, *args)
    (Path(out_dir) / f"rank_{local_rank}.pkl").write_bytes(pickle.dumps(result))


def spawn_group(
    fn: Callable[..., Any],
    world_size: int,
    *args: Any,
    backend: str = "gloo",
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> list[Any]:
    """Run ``fn(ctx, *args)`` on ``world_size`` fresh processes; return results ordered by rank.

    ``fn`` and ``args`` must be picklable (module-level function, plain data). A failure on
    any rank raises in the parent with that rank's traceback.
    """
    resolve_backend(world_size, torch.cuda.device_count(), backend)
    with tempfile.TemporaryDirectory(prefix="day5_spawn_") as out_dir:
        mp.spawn(
            _worker,
            args=(fn, world_size, find_free_port(), backend, timeout_s, out_dir, args),
            nprocs=world_size,
            join=True,
        )
        return [
            pickle.loads((Path(out_dir) / f"rank_{r}.pkl").read_bytes())
            for r in range(world_size)
        ]
