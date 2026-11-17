"""Day 5 / Session 3: a simplified tensor-parallel linear layer (forward pass only).

Convention: ``Y = X @ W`` with ``X: [tokens, d_in]`` and ``W: [d_in, d_out]``.

Column parallel -- split W's columns (output features). Every rank sees the full X and
computes ``Y_i = X @ W_i`` = a slice of Y's columns. Assembling Y needs an **AllGather**
(concatenate); if the next op is elementwise or row-parallel it can consume the shard as is
and no communication happens.

Row parallel -- split W's rows (input features). Rank i needs the matching slice
``X_i`` of X's columns and computes ``P_i = X_i @ W_i``, a *partial sum* of the full Y
with the full shape. Y = sum_i P_i, so it needs an **AllReduce(sum)**.

Column -> elementwise activation -> row (the Megatron MLP) needs just ONE AllReduce per MLP:
the column layer's sharded output is exactly the row layer's sharded input, and the
activation acts per element, so it commutes with sharding.

Backward is the mirror image (AllReduce of dX for column parallel, identity/AllGather for
row parallel); it is not implemented here -- this is an inference-oriented exercise.

All helpers are functional: shards are fresh copies, never views into the caller's tensor.
"""
from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F

import torch.distributed as dist

from collectives import all_gather_cat, all_reduce_sum


def _shard(tensor: torch.Tensor, dim: int, index: int, parts: int) -> torch.Tensor:
    """Copy of the ``index``-th of ``parts`` equal slices of ``tensor`` along ``dim``."""
    if parts < 1:
        raise ValueError(f"parts must be >= 1, got {parts}")
    size = tensor.shape[dim]
    if size % parts != 0:
        raise ValueError(f"dim {dim} of size {size} is not divisible by {parts}")
    width = size // parts
    return tensor.narrow(dim, index * width, width).clone().contiguous()


def _split(tensor: torch.Tensor, dim: int, parts: int) -> tuple[torch.Tensor, ...]:
    if parts < 1:
        raise ValueError(f"parts must be >= 1, got {parts}")
    return tuple(_shard(tensor, dim, i, parts) for i in range(parts))


def split_columns(weight: torch.Tensor, parts: int) -> tuple[torch.Tensor, ...]:
    """W [d_in, d_out] -> ``parts`` tensors [d_in, d_out/parts] (column parallel)."""
    return _split(weight, 1, parts)


def split_rows(weight: torch.Tensor, parts: int) -> tuple[torch.Tensor, ...]:
    """W [d_in, d_out] -> ``parts`` tensors [d_in/parts, d_out] (row parallel)."""
    return _split(weight, 0, parts)


def split_last_dim(x: torch.Tensor, parts: int) -> tuple[torch.Tensor, ...]:
    """X [..., d_in] -> ``parts`` tensors [..., d_in/parts] (row-parallel input)."""
    return _split(x, x.dim() - 1, parts)


def _validate_rank(rank: int, world_size: int) -> None:
    if world_size < 1:
        raise ValueError(f"world_size must be >= 1, got {world_size}")
    if not 0 <= rank < world_size:
        raise ValueError(f"rank {rank} is outside world of size {world_size}")


def _require_group(world_size: int) -> None:
    """A layer sharded for ``world_size > 1`` must run inside a group of exactly that size;
    otherwise the collectives silently degrade to no-ops and the output is wrong."""
    if world_size == 1:
        return
    if not (dist.is_available() and dist.is_initialized()):
        raise RuntimeError(
            f"layer is sharded for world_size={world_size} but no process group is initialised"
        )
    actual = dist.get_world_size()
    if actual != world_size:
        raise RuntimeError(
            f"layer is sharded for world_size={world_size} but the process group has size {actual}"
        )


class ColumnParallelLinear:
    """Holds columns [rank*d_out/n, (rank+1)*d_out/n) of W (and of the bias)."""

    def __init__(
        self,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        rank: int,
        world_size: int,
        gather_output: bool = True,
    ) -> None:
        _validate_rank(rank, world_size)
        self.weight = _shard(weight, 1, rank, world_size)
        self.bias = None if bias is None else _shard(bias, 0, rank, world_size)
        self.gather_output = gather_output
        self.world_size = world_size
        self.collective = "all_gather" if gather_output and world_size > 1 else "none"

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        _require_group(self.world_size)
        y = x @ self.weight
        if self.bias is not None:
            y = y + self.bias  # bias is sharded like the columns, so add locally
        return all_gather_cat(y, dim=-1) if self.gather_output else y


class RowParallelLinear:
    """Holds rows [rank*d_in/n, (rank+1)*d_in/n) of W; the bias is replicated."""

    def __init__(
        self,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        rank: int,
        world_size: int,
        input_is_parallel: bool = True,
    ) -> None:
        _validate_rank(rank, world_size)
        self.rank = rank
        self.world_size = world_size
        self.weight = _shard(weight, 0, rank, world_size)
        self.bias = None if bias is None else bias.clone()
        self.input_is_parallel = input_is_parallel
        self.collective = "all_reduce" if world_size > 1 else "none"

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        _require_group(self.world_size)
        x_local = x if self.input_is_parallel else _shard(x, x.dim() - 1, self.rank, self.world_size)
        partial = x_local @ self.weight  # full-shape partial sum
        y = all_reduce_sum(partial)
        # Bias after the reduce: adding it before would count it world_size times.
        return y if self.bias is None else y + self.bias


class ParallelMLP:
    """column (no gather) -> activation -> row (input already sharded): one AllReduce total."""

    def __init__(
        self,
        w_in: torch.Tensor,
        b_in: torch.Tensor | None,
        w_out: torch.Tensor,
        b_out: torch.Tensor | None,
        rank: int,
        world_size: int,
        activation: Callable[[torch.Tensor], torch.Tensor] = F.gelu,
    ) -> None:
        self.up = ColumnParallelLinear(w_in, b_in, rank, world_size, gather_output=False)
        self.down = RowParallelLinear(w_out, b_out, rank, world_size, input_is_parallel=True)
        self.activation = activation

    @property
    def collectives(self) -> list[str]:
        return [c for c in (self.up.collective, self.down.collective) if c != "none"]

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(self.activation(self.up(x)))
