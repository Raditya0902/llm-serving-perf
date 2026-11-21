"""Turn `vllm bench serve` result JSON (+ optional peak-memory JSON) into report rows.

P95 sources, in order of preference:
  1. `p95_ttft_ms` / `p95_tpot_ms` written by vLLM itself - needs the bench run with
     `--metric-percentiles 50,95,99 --percentile-metrics ttft,tpot,itl,e2el`
  2. recomputed from per-request arrays `ttfts` / `itls` - needs `--save-detailed`
     (per-request TPOT is then the mean of that request's inter-token latencies, an
     approximation of vLLM's own definition)
Otherwise it fails and names the flags to add.

WARNING: Day 4's run_sweep.py requests percentiles 50,99 and does NOT pass --save-detailed
(vLLM v0.34.0 deletes ttfts/itls from the saved JSON without it, see vllm/benchmarks/serve.py),
so Day 4 sweep files cannot provide P95. Use the direct commands in RUNBOOK.md for the
baseline-vs-modified table. Key names follow the v0.34.0 source; never run on a real result.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

MS = 1e3
P95 = 95


@dataclass(frozen=True)
class VllmRow:
    run_id: str
    requests_per_s: float
    output_tok_s: float
    ttft_ms_p95: float
    tpot_ms_p95: float
    peak_mem_mb: float | None


def _p95_ms(values: Sequence[float]) -> float:
    if not len(values):
        raise ValueError("no per-request values to take a percentile of")
    return float(np.percentile(np.asarray(values, dtype=float) * MS, P95))


def per_request_tpots(itls: Sequence[Sequence[float]]) -> list[float]:
    return [sum(request) / len(request) for request in itls if len(request)]


def _p95_from_result(data: dict, direct_key: str, from_arrays, flag_hint: str) -> float:
    if data.get(direct_key) is not None:
        return float(data[direct_key])
    values = from_arrays(data)
    if values:
        return _p95_ms(values)
    raise ValueError(f"no {direct_key} and no per-request arrays; re-run the bench with {flag_hint}")


def row_from_result(path: Path, peak_mem_path: Path | None = None) -> VllmRow:
    data = json.loads(Path(path).read_text())
    for key in ("request_throughput", "output_throughput"):
        if key not in data:
            raise ValueError(f"{path}: missing key {key!r} (result file not from vllm bench serve?)")
    peak = None
    if peak_mem_path is not None:
        mem = json.loads(Path(peak_mem_path).read_text())
        if mem.get("returncode", 0) != 0:
            raise ValueError(f"{peak_mem_path}: the benchmark exited with code {mem['returncode']}; its memory is not valid")
        peak = float(mem["peak_mib"])  # MiB == MB of 2**20 bytes
    return VllmRow(
        run_id=str(data.get("sweep_run_id", Path(path).stem)),
        requests_per_s=float(data["request_throughput"]),
        output_tok_s=float(data["output_throughput"]),
        ttft_ms_p95=_p95_from_result(
            data, "p95_ttft_ms", lambda d: d.get("ttfts") or [], "--metric-percentiles 50,95,99"
        ),
        tpot_ms_p95=_p95_from_result(
            data, "p95_tpot_ms", lambda d: per_request_tpots(d.get("itls") or []),
            "--metric-percentiles 50,95,99 (or --save-detailed)",
        ),
        peak_mem_mb=peak,
    )
