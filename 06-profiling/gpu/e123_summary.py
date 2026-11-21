"""E1-E3 of Day 1 (experiments.py) on vLLM: evaluate the unchanged hypotheses from gpu/run_e123_vllm.sh results.

    python -m gpu.e123_summary <root>        # <root>/<model>/e1_c<C>.json, e2_in<L>.json, e3_out<L>.json

  E1  concurrency 8 vs 1: median TPOT at most 1.5x, and output tok/s at least 5x
  E2  TTFT(2048) / TTFT(512) in [3.8, 5.0]
  E3  median TPOT changes by at most 5% from 32 to 256 output tokens
A run with failed requests counts as missing.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

E1_STEP_MAX, E1_THROUGHPUT_MIN = 1.5, 5.0
E2_LOW, E2_HIGH = 3.8, 5.0
E3_MAX_CHANGE = 0.05


def _load(directory: Path, name: str) -> dict | None:
    path = directory / f"{name}.json"
    if not path.is_file():
        return None
    data = json.loads(path.read_text())
    return None if data.get("failed", 0) or not data.get("completed") else data


def _result(directory: Path, names: list[str], build) -> dict:
    data = {n: _load(directory, n) for n in names}
    missing = [n for n, d in data.items() if d is None]
    if missing:
        return {"confirmed": None, "missing": missing}
    return build(data)


def _e1(directory: Path) -> dict:
    def build(d):
        step = d["e1_c8"]["median_tpot_ms"] / d["e1_c1"]["median_tpot_ms"]
        thr = d["e1_c8"]["output_throughput"] / d["e1_c1"]["output_throughput"]
        return {"step_time_ratio": step, "throughput_ratio": thr, "confirmed": step <= E1_STEP_MAX and thr >= E1_THROUGHPUT_MIN}
    return _result(directory, ["e1_c1", "e1_c8"], build)


def _e2(directory: Path) -> dict:
    def build(d):
        ratio = d["e2_in2048"]["median_ttft_ms"] / d["e2_in512"]["median_ttft_ms"]
        return {"ratio": ratio, "confirmed": E2_LOW <= ratio <= E2_HIGH}
    return _result(directory, ["e2_in512", "e2_in2048"], build)


def _e3(directory: Path) -> dict:
    def build(d):
        change = d["e3_out256"]["median_tpot_ms"] / d["e3_out32"]["median_tpot_ms"] - 1
        return {"change": change, "confirmed": abs(change) <= E3_MAX_CHANGE}
    return _result(directory, ["e3_out32", "e3_out256"], build)


def summarize(root: Path) -> dict:
    out = {}
    for directory in sorted(p for p in Path(root).iterdir() if p.is_dir()):
        out[directory.name] = {"E1": _e1(directory), "E2": _e2(directory), "E3": _e3(directory)}
    return out


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        raise SystemExit("usage: python -m gpu.e123_summary <root>")
    print(json.dumps(summarize(Path(argv[0])), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
