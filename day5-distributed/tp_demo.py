"""Day 5 / Session 3 demo: verify tensor-parallel layers against a single-device reference.

    ./launch.sh 2 tp_demo.py        # torchrun, 2 processes (Gloo on CPU here)

Every rank builds the same full weights from a shared seed, shards its own slice, runs the
parallel layer, and compares with the unsharded ``X @ W`` it computes locally.
"""
from __future__ import annotations

import sys

import torch
import torch.nn.functional as F

from dist_utils import DistContext, distributed_session
from tensor_parallel import ColumnParallelLinear, ParallelMLP, RowParallelLinear, split_last_dim

TOKENS, D_MODEL, D_FF = 6, 16, 64
DEFAULT_SEED = 1234
REL_TOLERANCE = 1e-5  # max|err| / max|reference|; fp32 rounding alone is ~1e-6 of the output scale


def _randn(shape: tuple[int, ...], seed: int, device: torch.device) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=gen).to(device)


def _line(rank: int, name: str, comm: str, got: torch.Tensor, expected: torch.Tensor) -> str:
    """Compare in *relative* terms: the unscaled randn outputs are large, so an absolute
    threshold would flag harmless floating-point reduction-order differences."""
    max_err = float((got - expected).abs().max())
    rel_err = max_err / float(expected.abs().max())
    verdict = "PASS" if rel_err < REL_TOLERANCE else "FAIL"
    return f"[rank {rank}] {name}: comm={comm}, max|err|={max_err:.2e} (rel {rel_err:.1e}) -> {verdict}"


def run_tp_demo(ctx: DistContext, seed: int = DEFAULT_SEED) -> list[str]:
    rank, world, dev = ctx.rank, ctx.world_size, ctx.device
    x = _randn((TOKENS, D_MODEL), seed, dev)
    w_up, b_up = _randn((D_MODEL, D_FF), seed + 1, dev), _randn((D_FF,), seed + 2, dev)
    w_down, b_down = _randn((D_FF, D_MODEL), seed + 3, dev), _randn((D_MODEL,), seed + 4, dev)
    lines = [f"[rank {rank}] world={world}: each rank holds 1/{world} of every weight matrix"]

    column = ColumnParallelLinear(w_up, b_up, rank, world, gather_output=True)
    ref_up = x @ w_up + b_up
    lines.append(_line(rank, "column_parallel+all_gather", f"1 {column.collective}", column(x), ref_up))

    row = RowParallelLinear(w_down, b_down, rank, world, input_is_parallel=True)
    hidden = _randn((TOKENS, D_FF), seed + 5, dev)
    ref_down = hidden @ w_down + b_down
    out_row = row(split_last_dim(hidden, world)[rank])
    lines.append(_line(rank, "row_parallel+all_reduce", f"1 {row.collective}", out_row, ref_down))

    mlp = ParallelMLP(w_up, b_up, w_down, b_down, rank, world)
    ref_mlp = F.gelu(x @ w_up + b_up) @ w_down + b_down
    comm = ", ".join(f"1 {c}" for c in mlp.collectives) or "none"
    lines.append(_line(rank, "mlp(column->gelu->row)", comm, mlp(x), ref_mlp))
    return lines


def main() -> int:
    with distributed_session() as ctx:
        lines = run_tp_demo(ctx)
        print("\n".join(lines), flush=True)
        failed = any("FAIL" in line for line in lines)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
