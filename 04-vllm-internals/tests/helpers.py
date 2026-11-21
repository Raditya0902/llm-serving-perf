"""Synthetic ``vllm bench serve`` result files for tests.

The key names mirror ``vllm/benchmarks/serve.py`` at v0.34.0 (scalar metrics,
``p<N>_<metric>_ms`` percentiles, ``--metadata`` keys).  The *numbers* are made
up -- they exist only to exercise the parsing and plotting code and must never
be reported as measurements.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SYNTHETIC_NOTE = "SYNTHETIC TEST DATA - not a measurement"


def make_result(
    run_id: str = "baseline",
    knob: str = "baseline",
    value: str = "baseline",
    *,
    max_num_seqs: int = 256,
    max_num_batched_tokens: int = 8192,
    gpu_memory_utilization: float = 0.85,
    request_rate: float | str = 2.0,
    input_len: int = 512,
    output_len: int = 128,
    output_throughput: float = 250.0,
    p99_ttft_ms: float = 80.0,
    p99_itl_ms: float = 12.0,
    failed: int = 0,
    config_id: str = "cfg-test",
    **overrides: Any,
) -> dict[str, Any]:
    """Return a dict shaped like a saved vLLM bench result."""
    data: dict[str, Any] = {
        "date": "20260101-000000",
        "backend": "vllm",
        "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "num_prompts": 100,
        "request_rate": request_rate if request_rate == "inf" else float(request_rate),
        "max_concurrency": None,
        "duration": 50.0,
        "completed": 100 - failed,
        "failed": failed,
        "total_input_tokens": 100 * input_len,
        "total_output_tokens": 100 * output_len,
        "request_throughput": 2.0,
        "output_throughput": output_throughput,
        "total_token_throughput": output_throughput * 5,
        "mean_ttft_ms": p99_ttft_ms / 3,
        "median_ttft_ms": p99_ttft_ms / 3,
        "std_ttft_ms": 5.0,
        "p50_ttft_ms": p99_ttft_ms / 3,
        "p99_ttft_ms": p99_ttft_ms,
        "mean_tpot_ms": p99_itl_ms / 2,
        "p99_tpot_ms": p99_itl_ms,
        "mean_itl_ms": p99_itl_ms / 2,
        "median_itl_ms": p99_itl_ms / 2,
        "p50_itl_ms": p99_itl_ms / 2,
        "p99_itl_ms": p99_itl_ms,
        "mean_e2el_ms": 1500.0,
        "p50_e2el_ms": 1450.0,
        "p99_e2el_ms": 2100.0,
        # --metadata items are stored as strings by vLLM.
        "sweep_run_id": run_id,
        "sweep_knob": knob,
        "sweep_value": value,
        "sweep_config_id": config_id,
        "p_max_num_seqs": str(max_num_seqs),
        "p_max_num_batched_tokens": str(max_num_batched_tokens),
        "p_gpu_memory_utilization": str(gpu_memory_utilization),
        "p_max_model_len": "4096",
        "p_request_rate": str(float(request_rate)),
        "p_input_len": str(input_len),
        "p_output_len": str(output_len),
        # vLLM also stores big per-request arrays; the parser must ignore them.
        "ttfts": [0.01, 0.02],
        "itls": [[0.005, 0.006], [0.005]],
        "synthetic_note": SYNTHETIC_NOTE,
    }
    data.update(overrides)
    return data


def write_result(directory: Path, data: dict[str, Any], rep: int = 0) -> Path:
    """Write ``data`` as ``<run_id>__r<rep>.json`` and return the path."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{data['sweep_run_id']}__r{rep}.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path
