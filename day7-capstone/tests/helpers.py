"""Synthetic vLLM-shaped result files for tests.

EVERYTHING HERE IS SYNTHETIC TEST DATA. No vLLM or GPU produced these numbers, and they must
never appear in a report. Key names follow ``vllm bench serve`` (v0.34.0, serve.py ~1315-1440).
"""
from __future__ import annotations

import json
from pathlib import Path

import matrix as mx


def make_bench_result(case: mx.MatrixCase, rep: int = 0, **overrides) -> dict:
    result = {
        "synthetic_test_fixture": True,
        "duration": 40.0, "completed": case.num_prompts, "failed": 0,
        "request_throughput": 2.5, "output_throughput": 800.0, "total_token_throughput": 4000.0,
        "mean_ttft_ms": 300.0, "median_ttft_ms": 280.0, "p50_ttft_ms": 280.0,
        "p95_ttft_ms": 900.0, "p99_ttft_ms": 1200.0,
        "mean_tpot_ms": 25.0, "p95_tpot_ms": 40.0, "p99_tpot_ms": 55.0,
        "mean_e2el_ms": 9000.0, "p95_e2el_ms": 15000.0, "p99_e2el_ms": 18000.0,
        "max_concurrent_requests": case.concurrency,
        **{kv.split("=", 1)[0]: kv.split("=", 1)[1] for kv in case.metadata(rep)},
    }
    result.update(overrides)
    return result


def write_bench_result(out_dir: Path, case: mx.MatrixCase, rep: int = 0, **overrides) -> Path:
    path = out_dir / f"{case.case_id}__r{rep}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(make_bench_result(case, rep, **overrides)))
    return path


def write_sidecar(
    out_dir: Path, case: mx.MatrixCase, rep: int = 0, *, poll=None, gpu_mem=None, **overrides,
) -> Path:
    path = out_dir / "sidecar" / f"{case.case_id}__r{rep}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    body = {
        "run_id": f"{case.case_id}__r{rep}", "matrix_config_id": case.config_id,
        "poll": poll if poll is not None else {
            "kv_peak_frac": 0.8, "waiting_peak": 3.0, "preemptions": 2.0, "samples": 10, "errors": 0},
        "gpu_mem": gpu_mem if gpu_mem is not None else {
            "peak_mib": 36864, "n_samples": 10, "errors": 0, "gpu_index": 0},
        **overrides,
    }
    path.write_text(json.dumps(body))
    return path
