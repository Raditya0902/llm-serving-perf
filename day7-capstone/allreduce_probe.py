"""All-reduce latency and bus bandwidth between two GPUs with torch.distributed (NCCL), for the interconnect section of the results.

    torchrun --nproc_per_node 2 allreduce_probe.py --out probe.json            # NVLink or PCIe, whatever the pair has
    NCCL_P2P_DISABLE=1 torchrun --nproc_per_node 2 allreduce_probe.py ...      # forces the host-memory path

Bus bandwidth uses the nccl-tests convention for all-reduce: busbw = 2 (n-1)/n x bytes / time. Sizes are bytes of a float16 tensor;
8 KiB is one decode token's hidden state at 4096 wide, 256 MiB approaches the link limit. Rank 0 writes the JSON.
"""
from __future__ import annotations

import argparse
import json
import os
import time

SIZES = [8 << 10, 64 << 10, 512 << 10, 4 << 20, 32 << 20, 256 << 20]


def bus_bandwidth_gb_s(nbytes: int, seconds: float, ranks: int) -> float:
    """nccl-tests all-reduce bus bandwidth in GB/s (1e9 bytes)."""
    if seconds <= 0 or ranks < 2:
        raise ValueError("need a positive time and at least two ranks")
    return 2 * (ranks - 1) / ranks * nbytes / seconds / 1e9


def main() -> int:
    import torch
    import torch.distributed as dist

    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--warmup", type=int, default=10)
    args = p.parse_args()
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank)))
    rows = []
    for nbytes in SIZES:
        x = torch.ones(nbytes // 2, dtype=torch.float16, device="cuda")
        for _ in range(args.warmup):
            dist.all_reduce(x)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(args.iters):
            dist.all_reduce(x)
        torch.cuda.synchronize()
        seconds = (time.perf_counter() - start) / args.iters
        rows.append({"bytes": nbytes, "us": seconds * 1e6, "busbw_gb_s": bus_bandwidth_gb_s(nbytes, seconds, world)})
    if rank == 0:
        with open(args.out, "w") as handle:
            json.dump({"ranks": world, "env": {k: v for k, v in os.environ.items() if k.startswith("NCCL_")}, "rows": rows}, handle, indent=1)
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
