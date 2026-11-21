"""Module-level worker functions for multi-process tests.

They live at module level (not inside tests) because ``spawn`` pickles them by reference.
Each takes the ``DistContext`` first and returns a picklable value that ``spawn_group``
sends back to the parent, ordered by rank.
"""
from __future__ import annotations

import torch

from dist_utils import DistContext


def identity(ctx: DistContext) -> tuple[int, int, str]:
    return ctx.rank, ctx.world_size, ctx.backend


def rank_tensor_sum(ctx: DistContext) -> float:
    import torch.distributed as dist

    t = torch.tensor([float(ctx.rank + 1)])
    dist.all_reduce(t)
    return float(t.item())


def always_fails(ctx: DistContext) -> None:
    if ctx.rank == 1:
        raise RuntimeError("boom on rank 1")
    import torch.distributed as dist

    dist.barrier()


def seeded_tensor(shape: tuple[int, ...], seed: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=gen)


# --- collectives workers -----------------------------------------------------------------
def worker_all_reduce(ctx: DistContext) -> tuple[list[float], list[float]]:
    from collectives import all_reduce_sum

    local = torch.full((3,), float(ctx.rank + 1))
    reduced = all_reduce_sum(local)
    return local.tolist(), reduced.tolist()  # local must be left untouched


def worker_broadcast(ctx: DistContext, src: int) -> tuple[list[float], list[float]]:
    from collectives import broadcast_from

    local = torch.full((2,), float(ctx.rank * 10))
    out = broadcast_from(local, src=src)
    return local.tolist(), out.tolist()


def worker_all_gather(ctx: DistContext, dim: int) -> list[list[float]]:
    from collectives import all_gather_cat

    local = torch.full((2, 2), float(ctx.rank))
    return all_gather_cat(local, dim=dim).tolist()


def worker_synchronize(ctx: DistContext, delay_s: float) -> tuple[float, float]:
    import time

    from collectives import synchronize

    before = time.time()
    if ctx.rank == 0:
        time.sleep(delay_s)
    synchronize()
    return before, time.time()


def worker_demo(ctx: DistContext) -> list[str]:
    from collectives import run_demo

    return run_demo(ctx)


# --- tensor-parallel workers -------------------------------------------------------------
def _tp_inputs(tokens: int, d_in: int, d_out: int, seed: int):
    x = seeded_tensor((tokens, d_in), seed)
    w = seeded_tensor((d_in, d_out), seed + 1)
    b = seeded_tensor((d_out,), seed + 2)
    return x, w, b


def worker_column(ctx: DistContext, gather: bool, with_bias: bool) -> dict:
    from tensor_parallel import ColumnParallelLinear

    x, w, b = _tp_inputs(5, 8, 12, seed=0)
    bias = b if with_bias else None
    layer = ColumnParallelLinear(w, bias, ctx.rank, ctx.world_size, gather_output=gather)
    y = layer(x)
    ref = x @ w + (b if with_bias else 0)
    if not gather:
        shard = ref.shape[1] // ctx.world_size
        ref = ref[:, ctx.rank * shard : (ctx.rank + 1) * shard]
    return {
        "max_err": float((y - ref).abs().max()),
        "out_shape": tuple(y.shape),
        "weight_numel": layer.weight.numel(),
        "collective": layer.collective,
    }


def worker_row(ctx: DistContext, parallel_input: bool, with_bias: bool) -> dict:
    from tensor_parallel import RowParallelLinear, split_last_dim

    x, w, b = _tp_inputs(5, 8, 12, seed=10)
    bias = b if with_bias else None
    layer = RowParallelLinear(w, bias, ctx.rank, ctx.world_size, input_is_parallel=parallel_input)
    x_in = split_last_dim(x, ctx.world_size)[ctx.rank] if parallel_input else x
    y = layer(x_in)
    ref = x @ w + (b if with_bias else 0)
    return {
        "max_err": float((y - ref).abs().max()),
        "weight_numel": layer.weight.numel(),
        "collective": layer.collective,
    }


def worker_mlp(ctx: DistContext) -> dict:
    import torch.nn.functional as F

    from tensor_parallel import ParallelMLP

    x = seeded_tensor((4, 8), 20)
    w1, b1 = seeded_tensor((8, 16), 21), seeded_tensor((16,), 22)
    w2, b2 = seeded_tensor((16, 8), 23), seeded_tensor((8,), 24)
    mlp = ParallelMLP(w1, b1, w2, b2, ctx.rank, ctx.world_size)
    ref = F.gelu(x @ w1 + b1) @ w2 + b2
    return {"max_err": float((mlp(x) - ref).abs().max()), "collectives": mlp.collectives}


def worker_tp_demo(ctx: DistContext) -> list[str]:
    from tp_demo import run_tp_demo

    return run_tp_demo(ctx)


def worker_tp_demo_many_seeds(ctx: DistContext, seeds: tuple[int, ...]) -> int:
    from tp_demo import run_tp_demo

    return sum(any("FAIL" in line for line in run_tp_demo(ctx, seed=s)) for s in seeds)


def worker_layer_without_group_check(ctx: DistContext, claimed_world: int) -> str:
    from tensor_parallel import RowParallelLinear

    layer = RowParallelLinear(torch.ones(4, 4), None, 0, claimed_world, input_is_parallel=False)
    try:
        layer(torch.ones(2, 4))
    except RuntimeError as exc:
        return str(exc)
    return "no error"


def worker_prepared_ops(ctx: DistContext, size_bytes: int, repeats: int) -> dict:
    from bench_workers import COLLECTIVES, prepare_collective_op

    out = {}
    for name in COLLECTIVES:
        op = prepare_collective_op(name, size_bytes, ctx)
        for _ in range(repeats):
            op()  # must stay finite however often it runs (no value growth)
        out[name] = True
    return out
