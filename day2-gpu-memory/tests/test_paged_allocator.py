from __future__ import annotations

import pytest

from paged_allocator import (
    ContiguousAllocator,
    InsufficientBlocksError,
    PagedKVAllocator,
)


def _paged(num_blocks: int = 8, block_size: int = 4) -> PagedKVAllocator:
    return PagedKVAllocator(num_blocks=num_blocks, block_size=block_size)


# ---------------------------------------------------------------- paged
def test_blocks_needed_is_ceiling():
    a = _paged()
    assert [a.blocks_needed(n) for n in (0, 1, 4, 5, 8, 9)] == [0, 1, 1, 2, 2, 3]


def test_allocate_takes_ceil_blocks_and_tracks_free():
    a = _paged()
    a.allocate("r1", 5)
    assert a.num_used_blocks == 2
    assert a.num_free_blocks == 6
    assert len(a.block_table("r1").block_ids) == 2
    assert a.block_table("r1").num_tokens == 5


def test_block_ids_are_unique_across_requests():
    a = _paged()
    a.allocate("r1", 5)
    a.allocate("r2", 9)
    ids = a.block_table("r1").block_ids + a.block_table("r2").block_ids
    assert len(ids) == len(set(ids)) == 5


def test_extend_within_block_takes_no_new_block():
    a = _paged()
    a.allocate("r1", 5)  # 2 blocks, 3 free slots in the last one
    a.extend("r1", 3)
    assert a.num_used_blocks == 2
    assert a.block_table("r1").num_tokens == 8


def test_extend_across_boundary_takes_one_block():
    a = _paged()
    a.allocate("r1", 4)  # exactly full
    a.extend("r1")
    assert a.num_used_blocks == 2
    assert a.block_table("r1").num_tokens == 5


def test_extend_by_many_tokens_takes_multiple_blocks():
    a = _paged()
    a.allocate("r1", 1)
    a.extend("r1", 12)
    assert a.num_used_blocks == 4
    assert a.block_table("r1").num_tokens == 13


def test_allocate_failure_is_atomic():
    a = _paged(num_blocks=3)
    a.allocate("r1", 4)
    before = a.stats()
    with pytest.raises(InsufficientBlocksError):
        a.allocate("r2", 12)  # needs 3, only 2 free
    assert a.num_free_blocks == 2
    assert not a.has_request("r2")
    after = a.stats()
    assert after.allocated_slots == before.allocated_slots
    assert after.failures == before.failures + 1


def test_extend_failure_is_atomic():
    a = _paged(num_blocks=2)
    a.allocate("r1", 8)
    with pytest.raises(InsufficientBlocksError):
        a.extend("r1")
    assert a.block_table("r1").num_tokens == 8
    assert a.num_free_blocks == 0


def test_can_allocate_reflects_free_blocks():
    a = _paged(num_blocks=2)
    assert a.can_allocate(8)
    assert not a.can_allocate(9)
    a.allocate("r1", 4)
    assert not a.can_allocate(8)


def test_free_returns_blocks_for_reuse():
    a = _paged(num_blocks=2)
    a.allocate("r1", 8)
    a.free("r1")
    assert a.num_free_blocks == 2
    assert not a.has_request("r1")
    a.allocate("r2", 8)  # succeeds only if blocks were really returned
    assert a.num_used_blocks == 2


def test_free_unknown_or_twice_raises():
    a = _paged()
    with pytest.raises(KeyError):
        a.free("ghost")
    a.allocate("r1", 1)
    a.free("r1")
    with pytest.raises(KeyError):
        a.free("r1")


def test_duplicate_allocate_raises():
    a = _paged()
    a.allocate("r1", 1)
    with pytest.raises(ValueError):
        a.allocate("r1", 1)


def test_extend_unknown_raises():
    with pytest.raises(KeyError):
        _paged().extend("ghost")


@pytest.mark.parametrize("bad", [0, -1])
def test_invalid_construction_raises(bad):
    with pytest.raises(ValueError):
        PagedKVAllocator(num_blocks=bad, block_size=4)
    with pytest.raises(ValueError):
        PagedKVAllocator(num_blocks=4, block_size=bad)


def test_allocate_negative_tokens_raises():
    with pytest.raises(ValueError):
        _paged().allocate("r1", -1)


def test_block_table_is_immutable_snapshot():
    a = _paged()
    a.allocate("r1", 4)
    old = a.block_table("r1")
    a.extend("r1")
    assert old.num_tokens == 4
    assert a.block_table("r1").num_tokens == 5


def test_stats_invariants_and_internal_fragmentation():
    a = _paged(num_blocks=8, block_size=4)
    a.allocate("r1", 5)  # 8 slots, 5 used
    a.allocate("r2", 4)  # 4 slots, 4 used
    s = a.stats()
    assert s.allocated_slots == 12
    assert s.used_tokens == 9
    assert s.internal_fragmentation == 3
    assert s.used_tokens <= s.allocated_slots <= s.capacity_slots
    assert s.capacity_slots == 32
    assert s.utilization == pytest.approx(9 / 32)
    assert s.num_requests == 2


def test_internal_fragmentation_bounded_by_block_size_per_request():
    a = _paged(num_blocks=64, block_size=16)
    for i, n in enumerate([1, 17, 33, 100]):
        a.allocate(f"r{i}", n)
    s = a.stats()
    assert s.internal_fragmentation < 4 * 16


def test_zero_token_allocation_holds_no_blocks():
    a = _paged()
    a.allocate("r1", 0)
    assert a.num_used_blocks == 0
    a.extend("r1")
    assert a.num_used_blocks == 1


# ----------------------------------------------------------- contiguous
def test_contiguous_reserves_max_seq_len_upfront():
    c = ContiguousAllocator(total_tokens=100, max_seq_len=40)
    c.allocate("r1", 5)
    s = c.stats()
    assert s.allocated_slots == 40
    assert s.used_tokens == 5
    assert s.internal_fragmentation == 35
    assert s.capacity_slots == 100


def test_contiguous_fails_exactly_when_less_than_max_seq_len_left():
    c = ContiguousAllocator(total_tokens=100, max_seq_len=40)
    c.allocate("r1", 1)
    c.allocate("r2", 1)  # 20 slots left, < 40
    with pytest.raises(InsufficientBlocksError):
        c.allocate("r3", 1)
    assert not c.can_allocate(1)
    assert c.stats().failures == 1


def test_contiguous_extend_within_reservation_and_beyond():
    c = ContiguousAllocator(total_tokens=100, max_seq_len=10)
    c.allocate("r1", 9)
    c.extend("r1")  # 10 == reservation, fine
    with pytest.raises(InsufficientBlocksError):
        c.extend("r1")
    assert c.stats().used_tokens == 10


def test_contiguous_free_releases_reservation():
    c = ContiguousAllocator(total_tokens=20, max_seq_len=10)
    c.allocate("r1", 1)
    c.allocate("r2", 1)
    c.free("r1")
    c.allocate("r3", 1)
    assert c.stats().num_requests == 2
    with pytest.raises(KeyError):
        c.free("r1")


def test_contiguous_rejects_prompt_larger_than_max_seq_len():
    c = ContiguousAllocator(total_tokens=100, max_seq_len=10)
    with pytest.raises(ValueError):
        c.allocate("r1", 11)


def test_contiguous_invalid_construction():
    with pytest.raises(ValueError):
        ContiguousAllocator(total_tokens=0, max_seq_len=1)
    with pytest.raises(ValueError):
        ContiguousAllocator(total_tokens=5, max_seq_len=10)


def test_contiguous_duplicate_and_unknown():
    c = ContiguousAllocator(total_tokens=100, max_seq_len=10)
    c.allocate("r1", 1)
    with pytest.raises(ValueError):
        c.allocate("r1", 1)
    with pytest.raises(KeyError):
        c.extend("ghost")


def test_can_allocate_negative_raises():
    with pytest.raises(ValueError):
        _paged().can_allocate(-1)
