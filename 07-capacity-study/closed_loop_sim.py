"""Closed-loop (fixed-concurrency) client on top of `03-batching-simulator`'s discrete-event engine.

`03-batching-simulator`'s `Simulator` replays a fixed arrival schedule (open loop). `vllm bench serve
--max-concurrency N --request-rate inf` is closed loop: N clients, each sending its next
request the instant its previous one finishes. That is what the matrix measures, so this
module subclasses the `03-batching-simulator` engine and replaces only the arrival logic; iteration
planning, the cost model, KV accounting, preemption and the per-iteration bookkeeping
(`_record`, `_apply`) are `03-batching-simulator`'s, unchanged.

SIMULATED: every number produced here comes from the `03-batching-simulator` roofline cost model, not a GPU.
Known simplifications (see README): no attention-quadratic prefill cost, no prefix cache,
vLLM's full-prompt capacity check at admission (`04-vllm-internals` COMPARISON.md) is not modelled, and
the client has zero think time and perfect pacing.
"""
from __future__ import annotations

import random

import numpy as np

import _prior  # noqa: F401  (puts folders 02-05 on sys.path)

# isort: split
from cost_model import CostModel
from request import Request
from scheduler import EngineState, SchedulerConfig, make_scheduler
from simulator import SimResult, Simulator

import kv_sizing
from matrix import MatrixCase

MS_PER_S = 1000.0
SOURCE = "sim-day3"
MIN_TPOT_TOKENS = 2  # TPOT needs at least one inter-token gap


class ClosedLoopSimulator(Simulator):
    def __init__(self, config: SchedulerConfig, cost_model: CostModel | None, concurrency: int) -> None:
        if config.kind != "continuous":
            raise ValueError("closed-loop simulation supports the continuous scheduler only")
        if concurrency < 1:
            raise ValueError("concurrency must be >= 1")
        super().__init__(config, cost_model)
        self.concurrency = concurrency

    def run(self, requests: list[Request]) -> SimResult:
        self.validate(requests)
        jobs = list(requests)  # issue order
        state = EngineState(self.config)
        scheduler = make_scheduler(self.config)
        records = []
        t, next_job = 0.0, 0

        while next_job < len(jobs) or state.waiting or state.running:
            state.now = t
            in_flight = len(state.waiting) + len(state.running)
            while in_flight < self.concurrency and next_job < len(jobs):
                job = jobs[next_job]
                job.arrival_time = job.waiting_since = t  # the client sends it now
                state.arrive(job)
                next_job += 1
                in_flight += 1

            plan = scheduler.plan(state)
            if plan.is_empty:
                raise RuntimeError("scheduler produced no work with requests pending (deadlock)")
            duration = self.cost_model.iteration_time(
                plan.prefill_tokens, plan.billed_decode_tokens, plan.decode_ctx_tokens
            )
            records.append(self._record(t, duration, plan, state))
            t += duration
            self._apply(plan, state, t)

        return SimResult(
            config=self.config, cost_model=self.cost_model, requests=tuple(jobs),
            iterations=tuple(records), events=tuple(state.events), first_arrival=0.0, end_time=t,
            preemptions=state.preemptions, kv_peak_blocks=state.kv.peak_used_blocks,
            kv_failures=state.kv.failures, kv_final_used_blocks=state.kv.used_blocks,
        )


def sample_requests(case: MatrixCase, rep: int) -> list[Request]:
    """Seeded prompt/output lengths ~ U[len(1-r), len(1+r)], like vLLM's random dataset."""
    w = case.workload
    rng = random.Random(case.seed_for(rep))

    def draw(mean: int) -> int:
        return rng.randint(round(mean * (1 - w.range_ratio)), round(mean * (1 + w.range_ratio)))

    return [Request(i, 0.0, draw(w.input_len), draw(w.output_len)) for i in range(case.num_prompts)]


def scheduler_config_for(case: MatrixCase) -> SchedulerConfig:
    cfg = case.config
    pool = kv_sizing.kv_pool_blocks(
        kv_sizing.model_spec(case.model_preset), kv_sizing.HARDWARE[case.hardware],
        cfg.gpu_memory_utilization,
    )
    return SchedulerConfig(
        kind="continuous", policy="fcfs", max_running_seqs=cfg.max_num_seqs,
        token_budget=cfg.max_num_batched_tokens,
        chunked_prefill=True,  # vLLM v1 always chunks prefill
        num_kv_blocks=pool, block_size=kv_sizing.BLOCK_SIZE, kv_mode="on_demand",
    )


def _pct(values: list[float], q: float) -> float | None:
    return float(np.percentile(values, q)) if values else None


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def row_from_result(result: SimResult, case: MatrixCase, rep: int) -> dict:
    """Unified metric row (same schema `ingest_gpu.py` produces for real vLLM runs)."""
    reqs = result.requests
    makespan = max(result.end_time - result.first_arrival, 1e-9)
    ttfts = [r.ttft for r in reqs]
    e2es = [r.latency for r in reqs]
    tpots = [
        (r.latency - r.ttft) / (r.generated_tokens - 1)
        for r in reqs if r.generated_tokens >= MIN_TPOT_TOKENS
    ]
    pool_blocks = result.config.num_kv_blocks
    block_bytes = result.config.block_size * kv_sizing.kv_bytes_per_token_for(
        kv_sizing.model_spec(case.model_preset)
    )
    return {
        "case_id": case.case_id, "config_id": case.config_id,
        "workload": case.workload.key, "config": case.config.name,
        "concurrency": case.concurrency, "rep": rep, "is_control": case.is_control,
        "source": SOURCE, "simulated": True, "completed": len(reqs),
        "mean_ttft_ms": _mean(ttfts), "p95_ttft_ms": _pct(ttfts, 95),
        "mean_tpot_ms": _mean(tpots), "p95_tpot_ms": _pct(tpots, 95),
        "mean_e2e_ms": _mean(e2es), "p95_e2e_ms": _pct(e2es, 95),
        "req_per_s": len(reqs) / makespan * MS_PER_S,
        "output_tok_s": sum(r.generated_tokens for r in reqs) / makespan * MS_PER_S,
        "peak_gpu_mem_gib": None,  # not measured: a simulation has no GPU
        "modeled_kv_pool_gib": pool_blocks * block_bytes / kv_sizing.GIB,
        "kv_peak_frac": result.kv_peak_blocks / pool_blocks,
        "preemptions": result.preemptions,
        "recomputed_tokens": sum(r.recomputed_tokens for r in reqs),
        "peak_waiting": max((i.waiting for i in result.iterations), default=0),
        "mean_batch": (
            sum(i.running * i.duration_ms for i in result.iterations)
            / max(sum(i.duration_ms for i in result.iterations), 1e-9)
        ),
    }


def simulate_case(case: MatrixCase, rep: int) -> SimResult:
    sim = ClosedLoopSimulator(
        scheduler_config_for(case),
        kv_sizing.cost_model_for(kv_sizing.model_spec(case.model_preset), kv_sizing.HARDWARE[case.hardware]),
        case.concurrency,
    )
    return sim.run(sample_requests(case, rep))


def row_for(case: MatrixCase, rep: int) -> dict:
    return row_from_result(simulate_case(case, rep), case, rep)
