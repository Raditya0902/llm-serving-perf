from __future__ import annotations

import pytest

from kv_blocks import KVBlockManager


def test_blocks_for_rounds_up() -> None:
    m = KVBlockManager(num_blocks=10, block_size=16)
    assert [m.blocks_for(n) for n in (0, 1, 16, 17, 32)] == [0, 1, 1, 2, 2]


def test_ensure_allocates_and_grows_only_missing_blocks() -> None:
    m = KVBlockManager(10, 16)
    assert m.try_ensure(1, 20)  # 2 blocks
    assert m.used_blocks == 2
    assert m.try_ensure(1, 32)  # still 2 blocks
    assert m.used_blocks == 2
    assert m.try_ensure(1, 33)  # 3 blocks
    assert m.used_blocks == 3 and m.blocks_of(1) == 3


def test_ensure_is_all_or_nothing_and_counts_failures() -> None:
    m = KVBlockManager(4, 16)
    assert m.try_ensure(1, 48)  # 3 blocks
    assert not m.try_ensure(2, 33)  # needs 3, only 1 free
    assert m.used_blocks == 3 and m.blocks_of(2) == 0
    assert m.failures == 1


def test_can_fit_does_not_allocate_or_count_failures() -> None:
    m = KVBlockManager(4, 16)
    assert m.can_fit(1, 64) and not m.can_fit(1, 65)
    assert m.used_blocks == 0 and m.failures == 0


def test_free_returns_blocks_and_tracks_peak() -> None:
    m = KVBlockManager(10, 16)
    m.try_ensure(1, 64)
    m.try_ensure(2, 32)
    assert m.peak_used_blocks == 6
    assert m.free(1) == 4
    assert m.used_blocks == 2 and m.free_blocks == 8
    assert m.peak_used_blocks == 6
    assert m.free(99) == 0


@pytest.mark.parametrize("args", [(0, 16), (4, 0)])
def test_invalid_geometry_rejected(args) -> None:
    with pytest.raises(ValueError):
        KVBlockManager(*args)
