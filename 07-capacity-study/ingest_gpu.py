"""Turn ``vllm bench serve`` result JSON (+ the runner's sidecar) into the matrix row schema.

The rows are identical in shape to the simulator's (``closed_loop_sim.row_from_result``), with
``source = "vllm-gpu"`` and ``simulated = False``, so ``stability``/``objective``/``plot_matrix``
treat both the same. Anything the run could not measure stays ``None``.

Result files may come from another machine, so they are treated as untrusted. A run enters an
average only if: the file is a regular, size-capped JSON object; its name is canonical
(``<case_id>__r<N>.json``) and agrees with its metadata; it was produced by the same configuration
(``matrix_config_id``); it has no failed or missing requests; every metric is a finite, non-negative
number; the observed concurrency did not exceed the target; and its sidecar (if any) belongs to the
same run and holds only valid numbers. Anything else is rejected with a message, never averaged.

    .venv/bin/python ingest_gpu.py --raw results/raw/<tag> --out results/gpu/<tag>
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import kv_sizing
import matrix as mx
import results_io
import stability

logger = logging.getLogger(__name__)

SOURCE = "vllm-gpu"
SIDECAR_DIR = "sidecar"
MAX_RESULT_BYTES = 256 * 1024 * 1024  # --save-detailed output is tens of MB; this is a sanity cap
MAX_SIDECAR_BYTES = 1024 * 1024
CANONICAL_NAME_RE = re.compile(r"^(?P<case_id>[A-Za-z0-9-]+)__r(?P<rep>0|[1-9][0-9]*)\.json$")
CANONICAL_INT_RE = re.compile(r"^(0|[1-9][0-9]*)$")
CANDIDATE_GLOB = "*__r*.json"
MIB_PER_GIB = 1024.0
# bench JSON key -> row key, all required, finite and non-negative
REQUIRED = {
    "mean_ttft_ms": "mean_ttft_ms", "p95_ttft_ms": "p95_ttft_ms",
    "mean_tpot_ms": "mean_tpot_ms", "p95_tpot_ms": "p95_tpot_ms",
    "mean_e2el_ms": "mean_e2e_ms", "p95_e2el_ms": "p95_e2e_ms",
    "request_throughput": "req_per_s", "output_throughput": "output_tok_s",
}
# sidecar section -> key -> (lower bound, upper bound or None)
MIN_POLL_SAMPLES = 2  # the runner always takes a baseline and a final sample
MAX_POLL_ERROR_FRACTION = 0.5
POLL_BOUNDS = {"kv_peak_frac": (0.0, 1.0), "waiting_peak": (0.0, None), "preemptions": (0.0, None)}


class ResultFormatError(ValueError):
    """A result file cannot be trusted as one repeat of its case."""


def case_index(**matrix_kwargs: str | bool) -> dict[str, mx.MatrixCase]:
    runs = mx.build_runs(1, 1, **matrix_kwargs)
    return {r.case.case_id: r.case for r in runs}


def read_json_object(path: Path, max_bytes: int, what: str) -> dict[str, Any]:
    """Read a JSON object from a regular, size-capped file; every failure is a ResultFormatError."""
    if path.is_symlink() or not path.is_file():
        raise ResultFormatError(f"{path.name}: {what} must be a regular file")
    if path.stat().st_size > max_bytes:
        raise ResultFormatError(f"{path.name}: {what} is too large (> {max_bytes} bytes)")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, RecursionError) as exc:
        raise ResultFormatError(f"{path.name}: {what} is unreadable ({type(exc).__name__})") from exc
    if not isinstance(data, dict):
        raise ResultFormatError(f"{path.name}: {what} must be a JSON object")
    return data


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _number(data: Mapping[str, Any], key: str, name: str) -> float:
    if key not in data:
        raise ResultFormatError(f"{name}: missing key {key!r}")
    value = data[key]
    if not _is_number(value) or value < 0:
        raise ResultFormatError(f"{name}: {key!r} must be a finite non-negative number, got {value!r}")
    return float(value)


def _filename_identity(path: Path) -> tuple[str, int]:
    match = CANONICAL_NAME_RE.match(path.name)
    if match is None:
        raise ResultFormatError(f"{path.name}: non-canonical filename (expected <case_id>__r<N>.json)")
    return match.group("case_id"), int(match.group("rep"))


def _parse_rep(raw: Any, name: str) -> int:
    """vLLM stores ``--metadata key=value`` pairs as strings, so accept a canonical digit string or an int."""
    if isinstance(raw, str) and CANONICAL_INT_RE.match(raw):
        return int(raw)
    if isinstance(raw, int) and not isinstance(raw, bool) and raw >= 0:
        return raw
    raise ResultFormatError(f"{name}: matrix_rep must be a non-negative integer, got {raw!r}")


def _identify(data: Mapping[str, Any], path: Path, cases: Mapping[str, mx.MatrixCase]) -> tuple[mx.MatrixCase, int]:
    file_case, file_rep = _filename_identity(path)
    case_id = data.get("matrix_case_id")
    case = cases.get(case_id) if isinstance(case_id, str) else None
    if case is None:
        raise ResultFormatError(f"{path.name}: unknown case {case_id!r} in metadata")
    rep = _parse_rep(data.get("matrix_rep"), path.name)
    if (case.case_id, rep) != (file_case, file_rep):
        raise ResultFormatError(
            f"{path.name}: metadata ({case.case_id}, rep {rep}) disagrees with the filename"
        )
    if data.get("matrix_config_id") != case.config_id:
        raise ResultFormatError(
            f"{path.name}: config_id {data.get('matrix_config_id')!r} != {case.config_id!r} "
            "(produced by a different configuration)"
        )
    return case, rep


def _check_completeness(data: Mapping[str, Any], case: mx.MatrixCase, name: str) -> int:
    completed, failed = int(_number(data, "completed", name)), int(_number(data, "failed", name))
    if failed or completed != case.num_prompts:
        raise ResultFormatError(
            f"{name}: incomplete run (completed {completed}/{case.num_prompts}, failed {failed})"
        )
    observed = _number(data, "max_concurrent_requests", name)
    exact = _exact_peak_concurrency(data)
    if exact is not None:
        # vLLM's own figure buckets requests by whole second, so a request ending and its replacement
        # starting in the same second both count: it read 68 against a cap of 64 on the real A40 run.
        observed = exact
    if observed > case.concurrency:
        raise ResultFormatError(f"{name}: observed concurrency {observed:g} exceeds the target {case.concurrency}")
    return completed


def _exact_peak_concurrency(data: Mapping[str, Any]) -> int | None:
    """Peak number of simultaneously open requests from per-request start times and latencies, if present."""
    starts, latencies = data.get("start_times"), data.get("latencies")
    if not isinstance(starts, list) or not isinstance(latencies, list) or not starts or len(starts) != len(latencies):
        return None
    try:
        # A request that ends exactly when another starts is not concurrent with it (ends sort first on ties).
        events = sorted([(float(s), 1) for s in starts] + [(float(s) + float(l), -1) for s, l in zip(starts, latencies)],
                        key=lambda e: (e[0], e[1]))
    except (TypeError, ValueError):
        return None
    open_now = peak = 0
    for _, delta in events:
        open_now += delta
        peak = max(peak, open_now)
    return peak


def _bounded(section: Mapping[str, Any], key: str, name: str, bounds: tuple[float, float | None]) -> Any:
    value = section.get(key)
    if value is None:
        return None
    lo, hi = bounds
    if not _is_number(value) or value < lo or (hi is not None and value > hi):
        raise ResultFormatError(f"{name}: sidecar {key} must be a number in [{lo}, {hi}], got {value!r}")
    return value


def _section(data: Mapping[str, Any], key: str, name: str) -> Mapping[str, Any]:
    section = data.get(key)
    if section is None:
        return {}
    if not isinstance(section, dict):
        raise ResultFormatError(f"{name}: sidecar {key} must be an object")
    return section


def _count(section: Mapping[str, Any], key: str, name: str) -> int | None:
    value = section.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ResultFormatError(f"{name}: sidecar {key} must be a non-negative integer, got {value!r}")
    return value


def _poll_is_healthy(poll: Mapping[str, Any], name: str) -> bool:
    """Server-side metrics are trusted only if the poller kept running and mostly succeeded."""
    if poll.get("thread_error"):
        return False
    samples, errors = _count(poll, "samples", name), _count(poll, "errors", name)
    if samples is None or samples < MIN_POLL_SAMPLES:
        return False
    return (errors or 0) / (samples + (errors or 0)) <= MAX_POLL_ERROR_FRACTION


def load_sidecar(path: Path, case: mx.MatrixCase, rep: int) -> tuple[dict[str, Any], dict[str, Any]]:
    """(poll, gpu_mem) for this run: validated numbers only; ({}, {}) if the run has no sidecar."""
    side_path = path.parent / SIDECAR_DIR / path.name
    if not side_path.exists() and not side_path.is_symlink():
        return {}, {}
    data = read_json_object(side_path, MAX_SIDECAR_BYTES, "sidecar")
    run_id = f"{case.case_id}__r{rep}"
    if data.get("run_id") != run_id:
        raise ResultFormatError(f"{path.name}: sidecar run_id {data.get('run_id')!r} != {run_id!r}")
    if data.get("matrix_config_id") != case.config_id:
        raise ResultFormatError(f"{path.name}: sidecar config_id does not match this run")
    poll, mem = _section(data, "poll", path.name), _section(data, "gpu_mem", path.name)
    clean_poll = {k: _bounded(poll, k, path.name, b) for k, b in POLL_BOUNDS.items()}
    if poll and not _poll_is_healthy(poll, path.name):
        clean_poll = dict.fromkeys(POLL_BOUNDS)  # unmeasured, not a partial peak
    peak = mem.get("peak_mib")
    if peak is not None and (isinstance(peak, bool) or not isinstance(peak, int) or peak < 0):
        raise ResultFormatError(f"{path.name}: sidecar peak_mib must be a non-negative integer, got {peak!r}")
    return clean_poll, {"peak_mib": None if mem.get("thread_error") else peak}


def load_run(path: Path, cases: Mapping[str, mx.MatrixCase] | None = None) -> dict[str, Any]:
    path = Path(path)
    cases = cases if cases is not None else case_index()
    data = read_json_object(path, MAX_RESULT_BYTES, "result")
    case, rep = _identify(data, path, cases)
    metrics = {row_key: _number(data, key, path.name) for key, row_key in REQUIRED.items()}
    completed = _check_completeness(data, case, path.name)
    poll, mem = load_sidecar(path, case, rep)
    peak_mib = mem.get("peak_mib")
    return {
        "case_id": case.case_id, "config_id": case.config_id,
        "workload": case.workload.key, "config": case.config.name,
        "concurrency": case.concurrency, "rep": rep, "is_control": case.is_control,
        "source": SOURCE, "simulated": False, "completed": completed,
        **metrics,
        "peak_gpu_mem_gib": None if peak_mib is None else peak_mib / MIB_PER_GIB,
        "modeled_kv_pool_gib": None,
        "kv_peak_frac": poll.get("kv_peak_frac"),
        "preemptions": poll.get("preemptions"),
        "recomputed_tokens": None,
        "peak_waiting": poll.get("waiting_peak"),
        "mean_batch": None,
    }


def load_dir(raw_dir: Path, cases: Mapping[str, mx.MatrixCase] | None = None) -> tuple[list[dict], list[str]]:
    """(valid rows, problem messages) for every ``*__r*.json`` directly in `raw_dir`.

    Non-canonical names (``__r00``, ``__rX``) are reported as problems, so one run cannot be
    counted twice under two spellings."""
    cases = cases if cases is not None else case_index()
    rows, problems = [], []
    for path in sorted(Path(raw_dir).glob(CANDIDATE_GLOB)):
        try:
            rows.append(load_run(path, cases))
        except ResultFormatError as exc:
            problems.append(str(exc))
    return rows, problems


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--hardware", default=mx.DEFAULT_HARDWARE, choices=sorted(kv_sizing.HARDWARE))
    p.add_argument("--model", default=mx.DEFAULT_HUB_MODEL)
    p.add_argument("--model-preset", default=mx.DEFAULT_PRESET, choices=sorted(kv_sizing.MODELS))
    p.add_argument("--cap-kv-pool", action="store_true", help="the runs were made with --cap-kv-pool")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cases = case_index(model=args.model, model_preset=args.model_preset, hardware=args.hardware,
                       cap_kv_pool=args.cap_kv_pool)
    rows, problems = load_dir(args.raw, cases)
    for message in problems:
        logger.error("REJECTED %s", message)
    if not rows:
        logger.error("no valid results in %s", args.raw)
        return 1
    meta = results_io.run_meta(
        simulated=False, note="measured with vllm bench serve; see sidecar files for server-side metrics",
        extra={"raw_dir": args.raw.name, "rejected": len(problems), "hardware": args.hardware,
               "model": args.model, "cap_kv_pool": args.cap_kv_pool},
    )
    results_io.write_results("matrix", rows, meta, args.out)
    results_io.write_results("matrix_summary", stability.aggregate_cases(rows), meta, args.out)
    logger.info("ingested %d runs (%d rejected) -> %s", len(rows), len(problems), args.out)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
