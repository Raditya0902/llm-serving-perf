from __future__ import annotations

import socket

import pytest

import helpers
from dist_utils import (
    find_free_port,
    loopback_interface,
    resolve_backend,
    spawn_group,
)


def test_resolve_backend_uses_nccl_only_with_enough_cuda_devices() -> None:
    assert resolve_backend(world_size=2, cuda_device_count=2) == "nccl"
    assert resolve_backend(world_size=2, cuda_device_count=8) == "nccl"


@pytest.mark.parametrize("devices", [0, 1])
def test_resolve_backend_falls_back_to_gloo(devices: int) -> None:
    assert resolve_backend(world_size=2, cuda_device_count=devices) == "gloo"


def test_resolve_backend_respects_explicit_override() -> None:
    assert resolve_backend(world_size=2, cuda_device_count=0, requested="gloo") == "gloo"


def test_resolve_backend_rejects_unknown_backend() -> None:
    with pytest.raises(ValueError, match="backend"):
        resolve_backend(world_size=2, cuda_device_count=0, requested="mpi")


def test_resolve_backend_rejects_nccl_without_enough_gpus() -> None:
    with pytest.raises(ValueError, match="nccl"):
        resolve_backend(world_size=2, cuda_device_count=1, requested="nccl")


def test_resolve_backend_rejects_nonpositive_world_size() -> None:
    with pytest.raises(ValueError, match="world_size"):
        resolve_backend(world_size=0, cuda_device_count=0)


def test_find_free_port_returns_bindable_port() -> None:
    port = find_free_port()
    with socket.socket() as s:
        s.bind(("127.0.0.1", port))


def test_loopback_interface_is_known_name() -> None:
    assert loopback_interface() in {"lo", "lo0"}


def test_spawn_group_returns_results_ordered_by_rank() -> None:
    results = spawn_group(helpers.identity, world_size=2)
    assert results == [(0, 2, "gloo"), (1, 2, "gloo")]


def test_spawn_group_processes_really_communicate() -> None:
    assert spawn_group(helpers.rank_tensor_sum, world_size=2) == [3.0, 3.0]


def test_spawn_group_propagates_worker_failure() -> None:
    with pytest.raises(Exception, match="boom on rank 1"):
        spawn_group(helpers.always_fails, world_size=2, timeout_s=60)


def test_spawn_group_rejects_bad_world_size() -> None:
    with pytest.raises(ValueError, match="world_size"):
        spawn_group(helpers.identity, world_size=0)
