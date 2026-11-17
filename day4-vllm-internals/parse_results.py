"""Parse saved ``vllm bench serve`` JSON files into a sweep summary.

Input : ``results/raw/<run_id>__r<rep>.json`` written by ``run_sweep.py``
        (``vllm bench serve --save-result --metadata sweep_*=... p_*=...``).
Output: ``results/sweep_summary.json`` and ``.csv`` -- one row per ``run_id``
        with metrics averaged over repetitions (and ``*_std`` columns).

Percentile key names (``p99_ttft_ms``) and ``--metadata`` handling were checked
against ``vllm/benchmarks/serve.py`` at v0.34.0.  A missing required key raises
``ResultFormatError`` naming the key and the file instead of guessing, so a
vLLM version that renames a metric is noticed rather than silently mis-plotted.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

REQUIRED_METRICS: tuple[str, ...] = (
    "completed", "failed", "duration",
    "request_throughput", "output_throughput", "total_token_throughput",
    "mean_ttft_ms", "p99_ttft_ms",
    "mean_itl_ms", "p99_itl_ms",
    "mean_e2el_ms",
)
OPTIONAL_METRICS: tuple[str, ...] = (
    "p50_ttft_ms", "mean_tpot_ms", "p99_tpot_ms", "p50_itl_ms",
    "p50_e2el_ms", "p99_e2el_ms",
    "num_preemptions",  # not in the bench JSON: scraped from the server's /metrics
)
# Prometheus counter `vllm:num_preemptions` (exposed with a `_total` suffix);
# the `_created` timestamp series must not match.
PREEMPTIONS_RE = re.compile(
    r"^vllm:num_preemptions(?:_total)?(?:\{[^}]*\})?\s+([0-9.eE+-]+)\s*$", re.MULTILINE
)
METRICS_SUBDIR = "metrics"
# metadata key -> (summary column, converter)
PARAM_KEYS: dict[str, tuple[str, Any]] = {
    "p_max_num_seqs": ("max_num_seqs", int),
    "p_max_num_batched_tokens": ("max_num_batched_tokens", int),
    "p_gpu_memory_utilization": ("gpu_memory_utilization", float),
    "p_request_rate": ("request_rate", float),  # float("inf") parses "inf"
    "p_input_len": ("input_len", int),
    "p_output_len": ("output_len", int),
}
SUM_COLUMNS = ("completed", "failed")
FILENAME_RE = re.compile(r"^(?P<run_id>.+)__r(?P<rep>\d+)\.json$")
SUMMARY_STEM = "sweep_summary"


class ResultFormatError(ValueError):
    """A result file is missing data this tooling needs."""


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    knob: str
    value: str
    config_id: str
    rep: int
    model: str
    params: Mapping[str, float | int]
    metrics: Mapping[str, float | None]


def _require(data: Mapping[str, Any], key: str, path: Path) -> Any:
    if key not in data or data[key] is None:
        raise ResultFormatError(f"{path.name}: missing required key {key!r}")
    return data[key]


def _convert(raw: Any, conv: Any, key: str, path: Path) -> float | int:
    try:
        return conv(raw)
    except (TypeError, ValueError) as exc:
        raise ResultFormatError(f"{path.name}: bad value {raw!r} for {key!r}") from exc


def parse_preemptions(metrics_text: str) -> float | None:
    """Total preemptions from Prometheus text (summed over label sets), or None."""
    values = [float(m.group(1)) for m in PREEMPTIONS_RE.finditer(metrics_text)]
    return sum(values) if values else None


def _read_preemptions(result_path: Path) -> float | None:
    metrics_file = result_path.parent / METRICS_SUBDIR / f"{result_path.stem}.metrics.txt"
    if not metrics_file.is_file():
        return None
    return parse_preemptions(metrics_file.read_text(encoding="utf-8"))


def load_result(path: Path) -> RunRecord:
    """Load one saved bench result."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ResultFormatError(f"{path.name}: not valid JSON ({exc})") from exc
    if not isinstance(data, dict):
        raise ResultFormatError(f"{path.name}: expected a JSON object")

    params = {
        column: _convert(_require(data, key, path), conv, key, path)
        for key, (column, conv) in PARAM_KEYS.items()
    }
    metrics: dict[str, float | None] = {
        key: _convert(_require(data, key, path), float, key, path)
        for key in REQUIRED_METRICS
    }
    for key in OPTIONAL_METRICS:
        if data.get(key) is not None:
            metrics[key] = _convert(data[key], float, key, path)
        else:
            metrics[key] = None
    metrics["num_preemptions"] = _read_preemptions(path)

    run_id = str(_require(data, "sweep_run_id", path))
    match = FILENAME_RE.match(path.name)
    if match is None or match.group("run_id") != run_id:
        raise ResultFormatError(
            f"{path.name}: file name does not match its run id {run_id!r} "
            "(expected '<run_id>__r<rep>.json'); was the file copied or renamed?"
        )
    return RunRecord(
        run_id=run_id,
        knob=str(_require(data, "sweep_knob", path)),
        value=str(_require(data, "sweep_value", path)),
        config_id=str(_require(data, "sweep_config_id", path)),
        rep=int(match.group("rep")),
        model=str(data.get("model_id", "")),
        params=params,
        metrics=metrics,
    )


def load_dir(raw_dir: Path) -> list[RunRecord]:
    """Load every ``*.json`` in ``raw_dir``, sorted by (run_id, rep)."""
    paths = sorted(raw_dir.glob("*.json")) if raw_dir.is_dir() else []
    if not paths:
        raise ResultFormatError(f"no result files found in {raw_dir}")
    records = [load_result(p) for p in paths]
    return sorted(records, key=lambda r: (r.run_id, r.rep))


def aggregate(records: Sequence[RunRecord]) -> list[dict[str, Any]]:
    """One row per run_id: metric means, ``*_std``, summed counts, ``n_reps``."""
    groups: dict[str, list[RunRecord]] = {}
    for rec in records:
        groups.setdefault(rec.run_id, []).append(rec)

    rows: list[dict[str, Any]] = []
    for run_id, recs in groups.items():
        first = recs[0]
        if len({r.config_id for r in recs}) > 1:
            raise ResultFormatError(
                f"run {run_id!r} has repetitions from different configurations "
                f"({sorted({r.config_id for r in recs})}); results from different "
                "load/model/prompt settings were mixed in one directory"
            )
        row: dict[str, Any] = {
            "run_id": run_id, "knob": first.knob, "value": first.value,
            "model": first.model, "n_reps": len(recs), **first.params,
        }
        for key in (*REQUIRED_METRICS, *OPTIONAL_METRICS):
            values = [r.metrics[key] for r in recs if r.metrics[key] is not None]
            if key in SUM_COLUMNS:
                row[key] = int(sum(values))
                continue
            # A metric missing from some repetitions is unknown, not the mean of the rest.
            complete = len(values) == len(recs)
            row[key] = float(np.mean(values)) if complete else None
            row[f"{key}_std"] = float(np.std(values, ddof=1)) if complete and len(values) > 1 else None
        rows.append(row)
    return rows


def _jsonable(value: Any) -> Any:
    """Strict JSON/CSV has no Infinity: spell non-finite floats as strings."""
    if isinstance(value, float) and math.isnan(value):
        return "nan"
    if isinstance(value, float) and not math.isfinite(value):
        return "inf" if value > 0 else "-inf"
    return value


def write_summary(rows: Sequence[Mapping[str, Any]], out_dir: Path) -> tuple[Path, Path]:
    """Write ``sweep_summary.json`` and ``.csv``; return their paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    clean = [{k: _jsonable(v) for k, v in row.items()} for row in rows]
    json_path = out_dir / f"{SUMMARY_STEM}.json"
    csv_path = out_dir / f"{SUMMARY_STEM}.csv"
    json_path.write_text(json.dumps({"rows": clean}, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(clean[0]) if clean else [])
        writer.writeheader()
        writer.writerows(clean)
    return json_path, csv_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw", type=Path, default=Path("results/raw"))
    parser.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args(argv)
    try:
        rows = aggregate(load_dir(args.raw))
    except ResultFormatError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    json_path, csv_path = write_summary(rows, args.out)
    for row in rows:
        flag = "  <-- FAILED REQUESTS" if row["failed"] else ""
        print(f"{row['run_id']:<32} out_tok/s={row['output_throughput']:>9.1f} "
              f"p99_ttft={row['p99_ttft_ms']:>8.1f}ms p99_itl={row['p99_itl_ms']:>7.1f}ms{flag}")
    print(f"wrote {json_path} and {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
