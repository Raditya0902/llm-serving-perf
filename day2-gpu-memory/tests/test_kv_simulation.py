from __future__ import annotations

import pytest

from kv_simulation import Request, WorkloadConfig, generate_requests, simulate
from paged_allocator import ContiguousAllocator, PagedKVAllocator

POOL = 4096
MAX_LEN = 512
CFG = WorkloadConfig(
    num_requests=120, max_seq_len=MAX_LEN, arrivals_per_step=0.5,
    prompt_median=32, prompt_sigma=0.8, output_median=32, output_sigma=0.6, seed=7,
)


def test_workload_is_deterministic_and_bounded():
    a, b = generate_requests(CFG), generate_requests(CFG)
    assert a == b
    assert len(a) == CFG.num_requests
    assert all(1 <= r.prompt_len < MAX_LEN and r.output_len >= 1 for r in a)
    assert all(r.total_len <= MAX_LEN for r in a)
    arrivals = [r.arrival_step for r in a]
    assert arrivals == sorted(arrivals)


def test_different_seed_changes_workload():
    other = WorkloadConfig(**{**CFG.__dict__, "seed": 8})
    assert generate_requests(CFG) != generate_requests(other)


@pytest.mark.parametrize("bad", [{"num_requests": 0}, {"max_seq_len": 1}, {"arrivals_per_step": 0}])
def test_invalid_workload_raises(bad):
    with pytest.raises(ValueError):
        generate_requests(WorkloadConfig(**{**CFG.__dict__, **bad}))


def test_single_request_lifecycle_and_pool_is_empty_afterwards():
    alloc = PagedKVAllocator(num_blocks=8, block_size=4)
    result = simulate("paged", alloc, (Request("r", 0, 5, 3),), block_size=4)
    assert result.completed == 1 and result.unfinished == 0
    assert result.preemptions == 0 and result.admission_failures == 0
    assert alloc.stats().allocated_slots == 0
    assert result.steps == 3  # one token per step


def test_all_requests_complete_for_both_allocators():
    reqs = generate_requests(CFG)
    paged = simulate("paged", PagedKVAllocator(POOL // 16, 16), reqs, block_size=16)
    contig = simulate("contiguous", ContiguousAllocator(POOL, MAX_LEN), reqs)
    assert paged.completed == contig.completed == len(reqs)
    assert paged.unfinished == contig.unfinished == 0


def test_contiguous_never_preempts_and_fails_admission_under_load():
    reqs = generate_requests(CFG)
    contig = simulate("contiguous", ContiguousAllocator(POOL, MAX_LEN), reqs)
    assert contig.preemptions == 0
    assert contig.admission_failures > 0
    assert contig.peak_running <= POOL // MAX_LEN


def test_paged_wastes_less_and_runs_more_concurrently():
    reqs = generate_requests(CFG)
    paged = simulate("paged", PagedKVAllocator(POOL // 16, 16), reqs, block_size=16)
    contig = simulate("contiguous", ContiguousAllocator(POOL, MAX_LEN), reqs)
    assert paged.unused_reserved_fraction < contig.unused_reserved_fraction
    assert paged.mean_utilization > contig.mean_utilization
    assert paged.mean_running > contig.mean_running
    assert paged.mean_queue_wait_steps < contig.mean_queue_wait_steps


def test_tiny_pool_forces_preemption_but_still_finishes():
    reqs = tuple(Request(f"r{i}", 0, 4, 12) for i in range(4))
    alloc = PagedKVAllocator(num_blocks=6, block_size=4)  # 24 slots; 4 reqs need up to 64
    result = simulate("paged", alloc, reqs, block_size=4)
    assert result.preemptions > 0
    assert result.completed == 4


def test_max_steps_cap_reports_unfinished():
    reqs = (Request("r", 0, 2, 50),)
    result = simulate("paged", PagedKVAllocator(16, 4), reqs, block_size=4, max_steps=5)
    assert result.unfinished == 1 and result.completed == 0


def test_result_row_is_flat_and_scalar():
    row = simulate("paged", PagedKVAllocator(8, 4), (Request("r", 0, 2, 2),), block_size=4).to_row()
    assert all(isinstance(v, (int, float, str, type(None))) for v in row.values())
    assert row["allocator"] == "paged" and row["block_size"] == 4


def test_request_larger_than_pool_is_rejected_up_front():
    with pytest.raises(ValueError, match="never finish|could never"):
        simulate("paged", PagedKVAllocator(2, 8), (Request("r", 0, 10, 10),), block_size=8)


def test_non_positive_max_steps_rejected():
    with pytest.raises(ValueError):
        simulate("paged", PagedKVAllocator(8, 4), (Request("r", 0, 2, 2),), max_steps=0)


def test_latency_counts_requeue_after_preemption():
    reqs = tuple(Request(f"r{i}", 0, 4, 12) for i in range(4))
    result = simulate("paged", PagedKVAllocator(num_blocks=6, block_size=4), reqs, block_size=4)
    assert result.preemptions > 0
    assert result.mean_latency_steps >= result.mean_queue_wait_steps + 12  # decode time alone is 12 steps


def test_uncontended_run_has_no_loaded_bias():
    # One request on a huge pool: nothing ever waits, stats fall back to all steps.
    result = simulate("paged", PagedKVAllocator(64, 4), (Request("r", 3, 4, 4),), block_size=4)
    assert result.mean_queue_wait_steps == 0 and result.mean_utilization > 0
