from __future__ import annotations

import pytest
import torch

from memory_probe import MemoryProbe, pick_device


def test_pick_device_auto_returns_available_device():
    assert pick_device() in ("cuda", "mps", "cpu")


def test_pick_device_cpu_always_ok_and_unknown_rejected():
    assert pick_device("cpu") == "cpu"
    with pytest.raises(ValueError):
        pick_device("tpu")


@pytest.mark.skipif(torch.cuda.is_available(), reason="cuda present")
def test_pick_device_unavailable_cuda_raises():
    with pytest.raises(RuntimeError):
        pick_device("cuda")


def test_cpu_probe_snapshot_is_consistent():
    probe = MemoryProbe("cpu")
    assert not probe.has_reserved_counter
    snap = probe.snapshot()
    assert snap.allocated > 0
    assert snap.peak_allocated >= snap.allocated
    probe.reset_peak()
    probe.empty_cache()


def test_unknown_device_probe_rejected():
    with pytest.raises(ValueError):
        MemoryProbe("tpu")


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS")
def test_mps_probe_tracks_live_tensor_and_reserved():
    probe = MemoryProbe("mps")
    probe.empty_cache()
    base = probe.snapshot()
    nbytes = 32 * 1024 * 1024
    x = torch.ones(nbytes, dtype=torch.uint8, device="mps")
    mid = probe.snapshot()
    assert mid.allocated - base.allocated >= nbytes
    assert mid.peak_allocated >= mid.allocated
    del x
    after = probe.snapshot()
    assert after.allocated < mid.allocated
    assert after.peak_allocated >= mid.allocated  # peak remembers the spike
    assert probe.has_reserved_counter
