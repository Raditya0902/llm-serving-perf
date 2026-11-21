"""Metrics computed from a finished simulation.

All times are simulated milliseconds; throughputs are per simulated second.

* throughput_tok_s   : generated output tokens / makespan
* latency            : finish_time - arrival_time (end to end, includes queueing)
* wait               : total time spent WAITING (before admission and after preemptions)
* ttft               : first_token_time - arrival_time
* token gap          : time between two consecutive output tokens of one request.
                       `p95_token_gap_ms` / `max_token_gap_ms` are taken over *every* gap of
                       every request (a per-token tail; stalls from long prefills and
                       preemption show up here).
                       `worst_itl_p95_ms` is different: the 95th percentile, across
                       requests, of each request's single worst gap (a per-request stall
                       measure, much larger than a typical decode step)
* util_proxy         : sum(useful_tokens * compute_ms_per_token) / makespan, i.e. the
                       fraction of wall time the "tensor cores" would be busy with
                       *useful* work (no idle gaps, no padded static slots, and KV that a
                       preemption threw away and had to recompute is not counted again). It is a
                       roofline-derived PROXY, not hardware GPU utilisation: decode is
                       memory-bound so even a perfect decode-only schedule scores low.
* busy_frac          : fraction of makespan the engine was running an iteration
* mean_batch         : time-weighted mean of occupied batch slots over busy time
* padded_slot_frac   : static batching only; share of billed decode slots wasted
                       on already-finished sequences
* peak_waiting       : deepest waiting queue seen at the start of any iteration
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from simulator import SimResult

MS_PER_S = 1000.0
EPS_MS = 1e-9  # avoid division by zero for empty runs


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(values, q)) if len(values) else 0.0


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if len(values) else 0.0


def summarize(result: SimResult) -> dict[str, float]:
    reqs = result.requests
    makespan = max(result.end_time - result.first_arrival, EPS_MS)
    out_tokens = sum(r.generated_tokens for r in reqs)
    latencies = [r.latency for r in reqs if r.latency is not None]
    ttfts = [r.ttft for r in reqs if r.ttft is not None]
    worst_itls = [r.max_itl for r in reqs if r.token_gaps]
    gaps = [g for r in reqs for g in r.token_gaps]
    its = result.iterations
    c_tok = result.cost_model.compute_ms_per_token
    busy = sum(i.duration_ms for i in its)
    recomputed = sum(r.recomputed_tokens for r in reqs)
    useful = (sum(i.prefill_tokens + i.decode_tokens for i in its) - recomputed) * c_tok
    billed_decode = sum(i.decode_tokens + i.padded_slots for i in its)
    padded = sum(i.padded_slots for i in its)
    return {
        "n_requests": len(reqs),
        "makespan_ms": makespan,
        "output_tokens": out_tokens,
        "throughput_tok_s": out_tokens / makespan * MS_PER_S,
        "throughput_req_s": len(reqs) / makespan * MS_PER_S,
        "mean_latency_ms": _mean(latencies),
        "p50_latency_ms": percentile(latencies, 50),
        "p95_latency_ms": percentile(latencies, 95),
        "p99_latency_ms": percentile(latencies, 99),
        "mean_wait_ms": _mean([r.waiting_time for r in reqs]),
        "p95_wait_ms": percentile([r.waiting_time for r in reqs], 95),
        "mean_ttft_ms": _mean(ttfts),
        "p95_ttft_ms": percentile(ttfts, 95),
        "p95_token_gap_ms": percentile(gaps, 95),
        "max_token_gap_ms": max(gaps, default=0.0),
        "worst_itl_p95_ms": percentile(worst_itls, 95),
        "util_proxy": useful / makespan,
        "busy_frac": busy / makespan,
        "mean_batch": (sum(i.running * i.duration_ms for i in its) / busy) if busy else 0.0,
        "padded_slot_frac": padded / billed_decode if billed_decode else 0.0,
        "peak_waiting": max((i.waiting for i in its), default=0),
        "kv_peak_blocks": result.kv_peak_blocks,
        "kv_peak_frac": result.kv_peak_blocks / result.config.num_kv_blocks,
        "preemptions": result.preemptions,
        "recomputed_tokens": recomputed,
        "iterations": len(its),
    }


def by_priority(result: SimResult) -> dict[int, dict[str, float]]:
    groups: dict[int, list] = defaultdict(list)
    for r in result.requests:
        groups[r.priority].append(r)
    return {
        prio: {
            "n": len(rs),
            "mean_latency_ms": _mean([r.latency for r in rs]),
            "p95_latency_ms": percentile([r.latency for r in rs], 95),
            "mean_wait_ms": _mean([r.waiting_time for r in rs]),
            "mean_ttft_ms": _mean([r.ttft for r in rs]),
            "preemptions": sum(r.preemptions for r in rs),
        }
        for prio, rs in sorted(groups.items())
    }
