from __future__ import annotations

import pytest

from helpers import CM, admit_order, cfg, req, run
from request import FINISHED


def test_single_request_timeline_matches_cost_model() -> None:
    r = req(1, p=100, o=3)
    res = run(cfg(), [r])
    t1 = CM.iteration_time(100, 0, 0)
    t2 = CM.iteration_time(0, 1, 100)
    t3 = CM.iteration_time(0, 1, 101)
    assert r.first_token_time == pytest.approx(t1)
    assert r.finish_time == pytest.approx(t1 + t2 + t3)
    assert r.status == FINISHED and r.generated_tokens == 3
    assert r.waiting_time == 0.0
    assert len(res.iterations) == 3


def test_idle_time_is_skipped_to_next_arrival() -> None:
    a, b = req(1, 0.0, o=2), req(2, 10_000.0, o=2)
    res = run(cfg(), [a, b])
    assert b.first_scheduled_time == pytest.approx(10_000.0)
    assert res.end_time > 10_000.0
    assert a.finish_time < 100.0


def test_max_running_seqs_is_enforced_and_all_requests_finish() -> None:
    reqs = [req(i, o=10) for i in range(12)]
    res = run(cfg(max_running_seqs=3), reqs)
    assert max(it.running for it in res.iterations) <= 3
    assert all(r.status == FINISHED and r.generated_tokens == r.max_output_tokens for r in reqs)


def test_token_budget_is_enforced_every_iteration() -> None:
    reqs = [req(i, p=150, o=12) for i in range(10)]
    res = run(cfg(token_budget=400, max_running_seqs=6), reqs)
    assert all(it.prefill_tokens + it.decode_tokens <= 400 for it in res.iterations)


def test_kv_never_exceeds_capacity_and_is_released_at_the_end() -> None:
    reqs = [req(i, p=200, o=40) for i in range(20)]
    res = run(cfg(num_kv_blocks=60), reqs)
    assert all(it.kv_used_blocks <= 60 for it in res.iterations)
    assert res.kv_peak_blocks <= 60
    assert res.kv_final_used_blocks == 0


def test_fcfs_admits_in_arrival_order() -> None:
    reqs = [req(i, arrival=float(i), o=20) for i in range(6)]
    res = run(cfg(max_running_seqs=2), reqs)
    assert admit_order(res) == [0, 1, 2, 3, 4, 5]


def test_waiting_time_accumulates_while_batch_is_full() -> None:
    a, b = req(1, 0.0, o=30), req(2, 0.0, o=5)
    run(cfg(max_running_seqs=1), [a, b])
    assert b.waiting_time == pytest.approx(a.finish_time)
    assert a.waiting_time == 0.0


def test_priority_policy_admits_higher_priority_first() -> None:
    blocker = req(0, 0.0, o=30, prio=1)
    low = req(1, 1.0, o=5, prio=1)
    high = req(2, 2.0, o=5, prio=0)
    res = run(cfg(policy="priority", max_running_seqs=1), [blocker, low, high])
    assert admit_order(res) == [0, 2, 1]
    assert high.finish_time < low.finish_time


def test_fcfs_ignores_priority() -> None:
    blocker, low, high = req(0, 0.0, o=30, prio=1), req(1, 1.0, o=5, prio=1), req(2, 2.0, o=5, prio=0)
    res = run(cfg(policy="fcfs", max_running_seqs=1), [blocker, low, high])
    assert admit_order(res) == [0, 1, 2]


def test_preempt_for_priority_evicts_lower_priority_runner() -> None:
    low = req(0, 0.0, p=64, o=60, prio=1)
    high = req(1, 50.0, p=32, o=5, prio=0)
    res = run(cfg(policy="priority", preempt_for_priority=True, max_running_seqs=1), [low, high])
    assert low.preemptions == 1 and low.recomputed_tokens > 0
    assert high.finish_time < low.finish_time
    assert low.generated_tokens == low.max_output_tokens
    assert res.preemptions == 1


def test_without_preempt_flag_high_priority_waits_for_runner() -> None:
    low = req(0, 0.0, p=64, o=60, prio=1)
    high = req(1, 50.0, p=32, o=5, prio=0)
    res = run(cfg(policy="priority", max_running_seqs=1), [low, high])
    assert low.finish_time < high.first_scheduled_time + 1e-9
    assert res.preemptions == 0


def test_chunked_prefill_splits_long_prompt_across_iterations() -> None:
    long_req = req(1, p=1000, o=4)
    res = run(cfg(chunked_prefill=True, token_budget=128, max_running_seqs=4), [long_req])
    prefill_iters = [it for it in res.iterations if it.prefill_tokens > 0]
    assert len(prefill_iters) == 8  # ceil(1000 / 128)
    assert all(it.prefill_tokens <= 128 for it in prefill_iters)
    assert sum(it.prefill_tokens for it in prefill_iters) == 1000


def test_chunked_prefill_bounds_decode_stall_from_long_prompt() -> None:
    def scenario(chunked: bool, budget: int):
        decoder = req(0, 0.0, p=16, o=200)
        long_req = req(1, 100.0, p=2000, o=4)
        run(cfg(chunked_prefill=chunked, token_budget=budget, max_running_seqs=4), [decoder, long_req])
        return decoder.max_itl

    assert scenario(False, 4096) > 3 * scenario(True, 256)


def test_head_of_line_blocking_in_fcfs_admission() -> None:
    # pool of 20 blocks (320 tokens). A runs; big B (needs 14 blocks) cannot fit next to A;
    # tiny C would fit but must wait behind B.
    a = req(0, 0.0, p=100, o=60)   # peak 159 tokens -> 10 blocks (reserved)
    b = req(1, 1.0, p=200, o=20)   # 14 blocks
    c = req(2, 2.0, p=16, o=2)     # 2 blocks
    res = run(cfg(num_kv_blocks=20, kv_mode="full_reserve"), [a, b, c])
    assert admit_order(res) == [0, 1, 2]
    assert c.first_scheduled_time >= a.finish_time


def test_kv_exhaustion_on_demand_preempts_and_still_completes() -> None:
    reqs = [req(i, 0.0, p=64, o=200) for i in range(8)]
    res = run(cfg(num_kv_blocks=60, max_running_seqs=8), reqs)
    assert res.preemptions > 0
    assert sum(r.recomputed_tokens for r in reqs) > 0
    assert all(r.generated_tokens == r.max_output_tokens and r.status == FINISHED for r in reqs)
    assert res.kv_peak_blocks <= 60


def test_full_reserve_never_preempts_but_runs_fewer_concurrently() -> None:
    mk = lambda: [req(i, 0.0, p=64, o=200) for i in range(8)]
    on_demand = run(cfg(num_kv_blocks=60), mk())
    reserve = run(cfg(num_kv_blocks=60, kv_mode="full_reserve"), mk())
    assert reserve.preemptions == 0
    assert max(i.running for i in reserve.iterations) < max(i.running for i in on_demand.iterations)


def test_oldest_request_is_never_the_preemption_victim() -> None:
    reqs = [req(i, float(i), p=64, o=200) for i in range(8)]
    run(cfg(num_kv_blocks=40), reqs)
    assert reqs[0].preemptions == 0


def test_static_batching_holds_slots_until_batch_drains() -> None:
    short, long_ = req(0, 0.0, p=32, o=4), req(1, 0.0, p=32, o=60)
    late = req(2, 1.0, p=32, o=4)
    res = run(cfg(kind="static", max_running_seqs=2), [short, long_, late])
    assert late.first_scheduled_time >= long_.finish_time  # no mid-batch admission
    assert any(it.padded_slots > 0 for it in res.iterations)  # finished slot padded
    assert short.finish_time < long_.finish_time


def test_static_batch_decodes_in_lockstep_after_all_prefill() -> None:
    a, b = req(0, 0.0, p=32, o=5), req(1, 0.0, p=32, o=5)
    res = run(cfg(kind="static", max_running_seqs=2), [a, b])
    kinds = ["P" if it.prefill_tokens else "D" for it in res.iterations]
    assert kinds == sorted(kinds, reverse=True)  # all P before any D


def test_continuous_beats_sequential_and_static_on_throughput() -> None:
    def mk():
        return [req(i, arrival=i * 5.0, p=64 + 16 * (i % 5), o=20 + 15 * (i % 4)) for i in range(30)]

    def tput(c):
        res = run(c, mk())
        return sum(r.generated_tokens for r in res.requests) / (res.end_time - res.first_arrival)

    seq = tput(cfg(max_running_seqs=1))
    static = tput(cfg(kind="static", max_running_seqs=8))
    cont = tput(cfg(max_running_seqs=8))
    assert cont > static > seq


def test_runs_are_deterministic() -> None:
    mk = lambda: [req(i, float(i * 3), p=48, o=30 + i) for i in range(15)]
    a, b = run(cfg(num_kv_blocks=50), mk()), run(cfg(num_kv_blocks=50), mk())
    assert [(i.start_ms, i.duration_ms) for i in a.iterations] == [(i.start_ms, i.duration_ms) for i in b.iterations]


def test_request_larger_than_kv_pool_is_rejected() -> None:
    with pytest.raises(ValueError, match="KV"):
        run(cfg(num_kv_blocks=4), [req(1, p=200, o=10)])


def test_unchunked_prompt_larger_than_budget_is_rejected() -> None:
    with pytest.raises(ValueError, match="budget"):
        run(cfg(token_budget=64), [req(1, p=200, o=10)])


def test_reusing_a_started_request_is_rejected() -> None:
    r = req(1)
    run(cfg(), [r])
    with pytest.raises(ValueError, match="fresh"):
        run(cfg(), [r])


def test_empty_workload_returns_empty_result() -> None:
    res = run(cfg(), [])
    assert res.iterations == () and res.requests == ()


# ---- regression tests from code review ------------------------------------------------
def test_futile_priority_preemption_is_not_performed() -> None:
    # pool 60 blocks, full_reserve. Two p0 runners hold 25 blocks each, a p1 runner holds 8.
    # A new p0 request needs 25 blocks: evicting the p1 runner frees only 8 (10 free < 25).
    a = req(0, 0.0, p=300, o=101, prio=0)  # peak 400 tokens -> 25 blocks
    b = req(1, 0.0, p=300, o=101, prio=0)
    c = req(2, 0.0, p=64, o=65, prio=1)   # 8 blocks
    d = req(3, 5.0, p=300, o=101, prio=0)
    run(cfg(policy="priority", preempt_for_priority=True, kv_mode="full_reserve", num_kv_blocks=60,
            token_budget=2048), [a, b, c, d])
    assert c.preemptions == 0


def test_priority_preemption_still_happens_when_it_can_help() -> None:
    low = req(0, 0.0, p=64, o=60, prio=1)
    high = req(1, 50.0, p=32, o=5, prio=0)
    res = run(cfg(policy="priority", preempt_for_priority=True, max_running_seqs=1), [low, high])
    assert res.preemptions == 1


def test_static_batching_releases_held_kv_at_the_end() -> None:
    res = run(cfg(kind="static", max_running_seqs=2), [req(0, 0.0, o=5), req(1, 0.0, o=50)])
    assert res.kv_final_used_blocks == 0


def test_arrive_event_uses_arrival_time_not_iteration_boundary() -> None:
    res = run(cfg(), [req(0, 0.0, p=400, o=2), req(1, 1.0, p=16, o=2)])
    arrive = {e.request_id: e.time_ms for e in res.events if e.kind == "ARRIVE"}
    assert arrive[1] == 1.0


def test_static_with_priority_preemption_flag_is_rejected() -> None:
    with pytest.raises(ValueError):
        cfg(kind="static", policy="priority", preempt_for_priority=True)


def test_random_workloads_keep_invariants_across_all_modes() -> None:
    import random

    rng = random.Random(7)
    for trial in range(60):
        kind = rng.choice(["continuous", "static"])
        chunked = kind == "continuous" and rng.random() < 0.5
        policy = rng.choice(["fcfs", "priority"])
        c = cfg(
            kind=kind, policy=policy, chunked_prefill=chunked, max_running_seqs=rng.choice([1, 3, 8]),
            token_budget=rng.choice([256, 512]), num_kv_blocks=rng.choice([40, 80, 400]),
            kv_mode="on_demand" if kind == "continuous" and rng.random() < 0.6 else "full_reserve",
            preempt_for_priority=kind == "continuous" and policy == "priority" and rng.random() < 0.5,
        )
        reqs = [req(i, rng.uniform(0, 300), p=rng.randint(8, 120), o=rng.randint(1, 60), prio=rng.randint(0, 2))
                for i in range(25)]
        res = run(c, reqs)
        assert all(r.generated_tokens == r.max_output_tokens for r in reqs), trial
        assert res.kv_final_used_blocks == 0, trial
        assert res.kv_peak_blocks <= c.num_kv_blocks, trial
        assert all(it.prefill_tokens + it.decode_tokens + it.padded_slots <= max(c.token_budget, it.decode_tokens + it.padded_slots)
                   for it in res.iterations), trial
