"""Closed-loop (fixed-concurrency) simulation on top of `03-batching-simulator`'s engine."""
from __future__ import annotations

import pytest
from cost_model import CostModel
from request import Request
from scheduler import SchedulerConfig

import closed_loop_sim as cl
import matrix as mx

COST = CostModel(overhead_ms=0.0, compute_ms_per_token=0.01, weight_read_ms=1.0, kv_read_ms_per_token=0.0)


def _config(**kw) -> SchedulerConfig:
    base = dict(max_running_seqs=8, token_budget=512, chunked_prefill=True, num_kv_blocks=400, block_size=16)
    return SchedulerConfig(**{**base, **kw})


def _jobs(n: int, prompt: int = 64, out: int = 8) -> list[Request]:
    return [Request(i, 0.0, prompt, out) for i in range(n)]


def _max_in_flight(result) -> int:
    edges = sorted(
        [(r.arrival_time, 1) for r in result.requests] + [(r.finish_time, -1) for r in result.requests],
        key=lambda e: (e[0], e[1]),  # a finish at time t frees its slot before an arrival at t
    )
    level = peak = 0
    for _, delta in edges:
        level += delta
        peak = max(peak, level)
    return peak


@pytest.mark.parametrize("concurrency", [1, 3, 8])
def test_never_exceeds_concurrency_and_finishes_everything(concurrency):
    result = cl.ClosedLoopSimulator(_config(), COST, concurrency).run(_jobs(20))
    assert all(r.finish_time is not None and r.generated_tokens == r.max_output_tokens for r in result.requests)
    assert _max_in_flight(result) <= concurrency


def test_concurrency_one_is_strictly_sequential_with_no_queueing():
    result = cl.ClosedLoopSimulator(_config(), COST, 1).run(_jobs(4))
    reqs = sorted(result.requests, key=lambda r: r.request_id)
    for prev, nxt in zip(reqs, reqs[1:]):
        assert nxt.arrival_time == pytest.approx(prev.finish_time)
    assert all(r.waiting_time == pytest.approx(0.0) for r in reqs)
    assert len({round(r.latency, 6) for r in reqs}) == 1  # identical jobs, identical solo latency


def test_higher_concurrency_raises_throughput_until_saturation():
    low = cl.ClosedLoopSimulator(_config(), COST, 1).run(_jobs(24))
    high = cl.ClosedLoopSimulator(_config(), COST, 8).run(_jobs(24))
    assert high.end_time < low.end_time  # batching amortises the weight read


def test_static_scheduler_is_rejected():
    with pytest.raises(ValueError, match="continuous"):
        cl.ClosedLoopSimulator(_config(kind="static", chunked_prefill=False), COST, 2)


def test_concurrency_must_be_positive():
    with pytest.raises(ValueError, match="concurrency"):
        cl.ClosedLoopSimulator(_config(), COST, 0)


def test_small_kv_pool_with_high_concurrency_preempts_but_still_finishes():
    tight = _config(num_kv_blocks=40)
    result = cl.ClosedLoopSimulator(tight, COST, 8).run(_jobs(16, prompt=96, out=24))
    assert result.preemptions > 0
    assert all(r.finish_time is not None for r in result.requests)


def test_ample_kv_pool_never_preempts():
    result = cl.ClosedLoopSimulator(_config(num_kv_blocks=4000), COST, 8).run(_jobs(16))
    assert result.preemptions == 0


def test_request_too_big_for_pool_is_rejected_up_front():
    with pytest.raises(ValueError, match="KV blocks"):
        cl.ClosedLoopSimulator(_config(num_kv_blocks=4), COST, 2).run(_jobs(2, prompt=500))


def test_same_seed_is_deterministic_and_seeds_differ():
    case = next(c for c in mx.build_matrix() if c.case_id == "A-c16-baseline")
    a, b, c = (cl.row_for(case, rep) for rep in (0, 0, 1))
    assert a == b
    assert a["p95_e2e_ms"] != c["p95_e2e_ms"]


def test_sample_requests_respects_range_ratio_and_count():
    case = next(c for c in mx.build_matrix() if c.case_id == "B-c16-baseline")
    reqs = cl.sample_requests(case, rep=0)
    assert len(reqs) == case.num_prompts
    w = case.workload
    assert all(round(w.input_len * 0.75) <= r.prompt_tokens <= round(w.input_len * 1.25) for r in reqs)
    assert all(round(w.output_len * 0.75) <= r.max_output_tokens <= round(w.output_len * 1.25) for r in reqs)


def test_row_metrics_hand_checked_on_one_request():
    result = cl.ClosedLoopSimulator(_config(), COST, 1).run([Request(0, 0.0, 100, 11)])
    row = cl.row_from_result(result, mx.build_matrix()[0], rep=0)
    ttft = max(0.01 * 100, 1.0)  # prefill iteration: max(compute 1.0, memory 1.0)
    decode_step = max(0.01 * 1, 1.0)  # memory-bound: one weight read
    assert row["mean_ttft_ms"] == pytest.approx(ttft)
    assert row["mean_tpot_ms"] == pytest.approx(decode_step)
    assert row["p95_e2e_ms"] == pytest.approx(ttft + 10 * decode_step)
    assert row["output_tok_s"] == pytest.approx(11 / (ttft + 10 * decode_step) * 1000)


def test_row_schema_marks_simulation_and_unmeasured_memory():
    case = mx.build_matrix()[0]
    row = cl.row_for(case, rep=0)
    assert row["source"] == "sim-day3" and row["simulated"] is True
    assert row["peak_gpu_mem_gib"] is None  # not measured in a simulation
    assert row["modeled_kv_pool_gib"] > 0
    for key in ("case_id", "config_id", "workload", "config", "concurrency", "rep", "mean_ttft_ms",
                "p95_ttft_ms", "mean_tpot_ms", "p95_tpot_ms", "p95_e2e_ms", "req_per_s",
                "output_tok_s", "kv_peak_frac", "preemptions", "peak_waiting"):
        assert key in row


def test_pressure_prediction_matches_simulated_preemption():
    """The analytic Day-2 check and the Day-3 engine should agree on where KV pressure appears."""
    for case in mx.build_matrix():
        row = cl.row_for(case, rep=0)
        predicted = case.predicted_pressure()["pressured"]
        assert (row["preemptions"] > 0) == predicted, (case.case_id, row["preemptions"], predicted)


def test_single_token_outputs_have_no_tpot_not_zero():
    result = cl.ClosedLoopSimulator(_config(), COST, 1).run([Request(0, 0.0, 50, 1), Request(1, 0.0, 50, 1)])
    row = cl.row_from_result(result, mx.build_matrix()[0], rep=0)
    assert row["mean_tpot_ms"] is None and row["p95_tpot_ms"] is None
