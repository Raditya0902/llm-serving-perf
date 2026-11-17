"""Simulated paged-KV block accounting.

Only *counts* blocks (no tensors): a sequence holding `n` KV tokens owns
`ceil(n / block_size)` blocks out of a fixed pool, like vLLM's block manager.
Allocation is all-or-nothing so a failed request never leaves partial state.
"""
from __future__ import annotations

import math


class KVBlockManager:
    def __init__(self, num_blocks: int, block_size: int) -> None:
        if num_blocks < 1 or block_size < 1:
            raise ValueError("num_blocks and block_size must be >= 1")
        self.num_blocks = num_blocks
        self.block_size = block_size
        self._owned: dict[int, int] = {}
        self.used_blocks = 0
        self.peak_used_blocks = 0
        self.failures = 0  # try_ensure calls that could not be satisfied

    @property
    def free_blocks(self) -> int:
        return self.num_blocks - self.used_blocks

    def blocks_for(self, tokens: int) -> int:
        return math.ceil(tokens / self.block_size)

    def blocks_of(self, req_id: int) -> int:
        return self._owned.get(req_id, 0)

    def can_fit(self, req_id: int, total_tokens: int) -> bool:
        """Would `req_id` fit if it needed KV for `total_tokens`? Never mutates."""
        return self.blocks_for(total_tokens) - self.blocks_of(req_id) <= self.free_blocks

    def try_ensure(self, req_id: int, total_tokens: int) -> bool:
        """Grow `req_id` to cover `total_tokens`; all-or-nothing."""
        extra = self.blocks_for(total_tokens) - self.blocks_of(req_id)
        if extra > self.free_blocks:
            self.failures += 1
            return False
        if extra > 0:
            self._owned[req_id] = self.blocks_of(req_id) + extra
            self.used_blocks += extra
            self.peak_used_blocks = max(self.peak_used_blocks, self.used_blocks)
        return True

    def free(self, req_id: int) -> int:
        """Release every block of `req_id`; return how many were freed."""
        n = self._owned.pop(req_id, 0)
        self.used_blocks -= n
        return n
