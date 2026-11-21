"""Day 5 / Session 2: PyTorch collective communication, one primitive at a time.

Run with two local processes (see README for why macOS needs ``launch.sh``)::

    torchrun --standalone --nnodes=1 --nproc-per-node=2 collectives.py

Covers: process initialisation, rank identification, AllReduce, Broadcast, AllGather,
synchronisation (barrier) and clean shutdown. Backend is NCCL when every rank has a CUDA
device, otherwise Gloo on CPU (the only one exercised on this Mac).

The functions are thin and *functional*: they return new tensors and never mutate the
caller's input. When no process group exists they behave as a world of size 1, so code
built on them (tensor_parallel.py) also runs in a single process.
"""
from __future__ import annotations

import sys

import torch
import torch.distributed as dist

from dist_utils import DistContext, distributed_session


def _in_group() -> bool:
    return dist.is_available() and dist.is_initialized()


def all_reduce_sum(tensor: torch.Tensor) -> torch.Tensor:
    """Every rank ends up with the elementwise sum over ranks (reduce + broadcast)."""
    out = tensor.clone()
    if _in_group():
        dist.all_reduce(out, op=dist.ReduceOp.SUM)
    return out


def broadcast_from(tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
    """Every rank receives ``src``'s tensor. Non-source values are only a shape/dtype template."""
    out = tensor.clone()
    if _in_group():
        dist.broadcast(out, src=src)
    return out


def all_gather_cat(tensor: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """Every rank receives all ranks' tensors, concatenated along ``dim`` in rank order.

    Shapes must match across ranks (equal shards, as in tensor parallelism).
    """
    if not _in_group():
        return tensor.clone()
    pieces = [torch.empty_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(pieces, tensor.contiguous())
    return torch.cat(pieces, dim=dim)


def synchronize() -> None:
    """Barrier: no rank continues until all ranks have arrived."""
    if _in_group():
        dist.barrier()


def _check(rank: int, name: str, detail: str, ok: bool) -> str:
    return f"[rank {rank}] {name}: {detail} -> {'PASS' if ok else 'FAIL'}"


def run_demo(ctx: DistContext) -> list[str]:
    """Exercise every collective with a known answer; return one report line per check."""
    rank, world = ctx.rank, ctx.world_size
    device = ctx.device
    lines = [
        f"[rank {rank}] identity: rank {rank} of {world}, local_rank={ctx.local_rank}, "
        f"backend={ctx.backend}, device={device}"
    ]

    local = torch.full((3,), float(rank + 1), device=device)
    reduced = all_reduce_sum(local)
    expected_sum = world * (world + 1) / 2
    lines.append(
        _check(
            rank, "all_reduce", f"{local[0].item():g} -> {reduced[0].item():g} (expect {expected_sum:g})",
            bool(torch.all(reduced == expected_sum)),
        )
    )

    payload = torch.tensor([42.0, 7.0], device=device) if rank == 0 else torch.zeros(2, device=device)
    received = broadcast_from(payload, src=0)
    lines.append(
        _check(
            rank, "broadcast", f"{payload.tolist()} -> {received.tolist()} (src=rank 0)",
            received.tolist() == [42.0, 7.0],
        )
    )

    shard = torch.full((2,), float(rank), device=device)
    gathered = all_gather_cat(shard)
    expected = torch.arange(world, dtype=torch.float32, device=device).repeat_interleave(2)
    lines.append(
        _check(rank, "all_gather", f"{shard.tolist()} -> {gathered.tolist()}", torch.equal(gathered, expected))
    )

    synchronize()
    lines.append(_check(rank, "barrier", "all ranks arrived", True))
    return lines


def main() -> int:
    with distributed_session() as ctx:
        lines = run_demo(ctx)
        print("\n".join(lines), flush=True)  # one write per rank so ranks don't interleave mid-line
        failed = any("FAIL" in line for line in lines)
        if ctx.rank == 0:
            print(f"collectives demo: {'FAILED' if failed else 'all checks passed'}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
