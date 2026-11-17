"""Simplified paged KV-cache allocator (metadata only) and a contiguous baseline.

GPU KV storage is modelled as a pool of fixed-size *blocks*, each holding `block_size`
tokens. A request owns a `BlockTable`: an ordered list of block ids plus the number of
tokens stored. No tensors and no attention kernel here -- only the bookkeeping that a
real system (e.g. vLLM's block manager) needs: who owns which block, what is free, and
when the pool is exhausted.

`ContiguousAllocator` is the strawman the experiments compare against: it reserves the
request's *maximum* sequence length upfront, because a contiguous buffer cannot grow in
place once other requests sit next to it.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Protocol


class InsufficientBlocksError(RuntimeError):
    """Raised when the KV pool cannot satisfy an allocate/extend request."""


@dataclass(frozen=True)
class BlockTable:
    """Immutable snapshot of one request's KV storage."""

    block_ids: tuple[int, ...]
    num_tokens: int


@dataclass(frozen=True)
class AllocatorStats:
    """Point-in-time accounting. All quantities are in token slots."""

    capacity_slots: int
    allocated_slots: int
    used_tokens: int
    num_requests: int
    failures: int

    @property
    def internal_fragmentation(self) -> int:
        """Reserved slots that hold no token yet (unused reserved space)."""
        return self.allocated_slots - self.used_tokens

    @property
    def utilization(self) -> float:
        """Fraction of the whole pool holding real tokens."""
        return self.used_tokens / self.capacity_slots

    @property
    def allocated_fraction(self) -> float:
        """Fraction of the whole pool that is reserved (used or not)."""
        return self.allocated_slots / self.capacity_slots


class KVAllocator(Protocol):
    """Interface shared by the paged and contiguous allocators."""

    def can_allocate(self, num_tokens: int) -> bool: ...
    def allocate(self, req_id: str, num_tokens: int) -> None: ...
    def extend(self, req_id: str, num_tokens: int = 1) -> None: ...
    def free(self, req_id: str) -> None: ...
    def has_request(self, req_id: str) -> bool: ...
    def stats(self) -> AllocatorStats: ...


def _require_positive(**values: int) -> None:
    for name, value in values.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")


class PagedKVAllocator:
    """Fixed-size block allocator with per-request block tables."""

    def __init__(self, num_blocks: int, block_size: int) -> None:
        _require_positive(num_blocks=num_blocks, block_size=block_size)
        self.num_blocks = num_blocks
        self.block_size = block_size
        self._free: deque[int] = deque(range(num_blocks))
        self._tables: dict[str, BlockTable] = {}
        self._failures = 0

    # ------------------------------------------------------------ queries
    @property
    def num_free_blocks(self) -> int:
        return len(self._free)

    @property
    def num_used_blocks(self) -> int:
        return self.num_blocks - len(self._free)

    def blocks_needed(self, num_tokens: int) -> int:
        return -(-num_tokens // self.block_size)  # ceil division

    def can_allocate(self, num_tokens: int) -> bool:
        if num_tokens < 0:
            raise ValueError(f"num_tokens must be >= 0, got {num_tokens}")
        return self.blocks_needed(num_tokens) <= self.num_free_blocks

    def has_request(self, req_id: str) -> bool:
        return req_id in self._tables

    def block_table(self, req_id: str) -> BlockTable:
        return self._tables[req_id]

    def stats(self) -> AllocatorStats:
        return AllocatorStats(
            capacity_slots=self.num_blocks * self.block_size,
            allocated_slots=self.num_used_blocks * self.block_size,
            used_tokens=sum(t.num_tokens for t in self._tables.values()),
            num_requests=len(self._tables),
            failures=self._failures,
        )

    # ---------------------------------------------------------- mutations
    def allocate(self, req_id: str, num_tokens: int) -> None:
        """Reserve blocks for a new request's prompt. All-or-nothing."""
        if num_tokens < 0:
            raise ValueError(f"num_tokens must be >= 0, got {num_tokens}")
        if req_id in self._tables:
            raise ValueError(f"request {req_id!r} already allocated")
        needed = self.blocks_needed(num_tokens)
        self._take_or_fail(needed, f"allocate({req_id!r}, {num_tokens})")
        ids = tuple(self._free.popleft() for _ in range(needed))
        self._tables[req_id] = BlockTable(ids, num_tokens)

    def extend(self, req_id: str, num_tokens: int = 1) -> None:
        """Append decoded tokens; grabs a new block only when the last one is full."""
        if num_tokens <= 0:
            raise ValueError(f"num_tokens must be positive, got {num_tokens}")
        table = self._tables[req_id]
        new_total = table.num_tokens + num_tokens
        extra = self.blocks_needed(new_total) - len(table.block_ids)
        self._take_or_fail(extra, f"extend({req_id!r}, {num_tokens})")
        new_ids = tuple(self._free.popleft() for _ in range(extra))
        self._tables[req_id] = BlockTable(table.block_ids + new_ids, new_total)

    def free(self, req_id: str) -> None:
        """Release all of a finished request's blocks back to the pool."""
        table = self._tables.pop(req_id)  # KeyError for unknown / double free
        self._free.extend(table.block_ids)

    def _take_or_fail(self, needed: int, what: str) -> None:
        if needed > self.num_free_blocks:
            self._failures += 1
            raise InsufficientBlocksError(
                f"{what} needs {needed} blocks but only {self.num_free_blocks} are free"
            )


class ContiguousAllocator:
    """Baseline: reserve `max_seq_len` slots per request at admission time.

    Slots are interchangeable fixed-size reservations, so this models only the
    *internal* waste of over-reserving. A real contiguous allocator would additionally
    suffer external fragmentation, which this baseline deliberately leaves out (it is
    therefore a generous baseline).
    """

    def __init__(self, total_tokens: int, max_seq_len: int) -> None:
        _require_positive(total_tokens=total_tokens, max_seq_len=max_seq_len)
        if max_seq_len > total_tokens:
            raise ValueError("max_seq_len cannot exceed total_tokens")
        self.total_tokens = total_tokens
        self.max_seq_len = max_seq_len
        self._tokens: dict[str, int] = {}
        self._failures = 0

    @property
    def free_slots(self) -> int:
        return self.total_tokens - len(self._tokens) * self.max_seq_len

    def can_allocate(self, num_tokens: int) -> bool:
        return num_tokens <= self.max_seq_len and self.free_slots >= self.max_seq_len

    def has_request(self, req_id: str) -> bool:
        return req_id in self._tokens

    def stats(self) -> AllocatorStats:
        return AllocatorStats(
            capacity_slots=self.total_tokens,
            allocated_slots=len(self._tokens) * self.max_seq_len,
            used_tokens=sum(self._tokens.values()),
            num_requests=len(self._tokens),
            failures=self._failures,
        )

    def allocate(self, req_id: str, num_tokens: int) -> None:
        if num_tokens < 0 or num_tokens > self.max_seq_len:
            raise ValueError(f"num_tokens must be in [0, {self.max_seq_len}], got {num_tokens}")
        if req_id in self._tokens:
            raise ValueError(f"request {req_id!r} already allocated")
        if self.free_slots < self.max_seq_len:
            self._failures += 1
            raise InsufficientBlocksError(
                f"allocate({req_id!r}) needs {self.max_seq_len} contiguous slots, "
                f"only {self.free_slots} free"
            )
        self._tokens[req_id] = num_tokens

    def extend(self, req_id: str, num_tokens: int = 1) -> None:
        if num_tokens <= 0:
            raise ValueError(f"num_tokens must be positive, got {num_tokens}")
        current = self._tokens[req_id]
        if current + num_tokens > self.max_seq_len:
            self._failures += 1
            raise InsufficientBlocksError(
                f"extend({req_id!r}) exceeds reserved capacity of {self.max_seq_len} tokens"
            )
        self._tokens[req_id] = current + num_tokens

    def free(self, req_id: str) -> None:
        del self._tokens[req_id]  # KeyError for unknown / double free
