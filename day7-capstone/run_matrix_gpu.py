"""Run the 12-case matrix against real vLLM on a Linux + NVIDIA host.

    .venv/bin/python run_matrix_gpu.py --dry-run                       # print every command
    .venv/bin/python run_matrix_gpu.py --out results/raw/a100 --reps 3 --key-reps 5
    .venv/bin/python ingest_gpu.py --raw results/raw/a100 --out results/gpu/a100

NOT RUN in this repo's history: the Mac that built it has no NVIDIA GPU. The orchestration is
tested end to end with stand-in server/bench/nvidia-smi (tests/test_run_matrix_gpu.py); the
real subprocess path reuses Day 4's `default_backend()`.

Per run: a FRESH server (no cache or fragmentation carried over), wait for ``/health``, a
baseline ``/metrics`` sample, the benchmark with a background poller (peak KV usage, peak queue,
preemption delta) and an ``nvidia-smi`` sampler (peak memory), then guaranteed teardown. Runs are
issued round by round with the case order rotated each round, so no case always goes first (the
position bias Day 6 found). Existing results are skipped only if they were produced by the same
configuration (``matrix_config_id``).
"""
from __future__ import annotations

import argparse
import contextlib
import http.client
import logging
import os
import random
import shlex
import subprocess
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import _prior  # noqa: F401  (puts Days 2-5 on sys.path)

# isort: split
from run_sweep import (  # Day 4
    Backend,
    ServerHandle,
    ServerStartError,
    _interrupt_on_signals,
    default_backend,
    stop_server,
    wait_until_healthy,
)
from sweep_config import LOOPBACK  # Day 4: the only host the sweep ever talks to

import gpu_mem
import ingest_gpu
import kv_sizing
import matrix as mx
import prom_metrics
import results_io

logger = logging.getLogger("run_matrix_gpu")

DEFAULT_PORT = 8000
DEFAULT_STARTUP_TIMEOUT_S = 900.0  # first start downloads weights and compiles
DEFAULT_OUT = Path(__file__).resolve().parent / "results" / "raw" / "gpu"
SIDECAR_DIR = "sidecar"
ORDER_SEED = 20260101  # fixed: the run order is reproducible
LOOPBACK_ENTRIES = ("127.0.0.1", "localhost", "::1")
MAX_PORT = 65535
MAX_REPS = 100
EXIT_OK, EXIT_RUN_FAILED, EXIT_INTERRUPTED = 0, 1, 130
# Everything that can go wrong around one run without being a bug in this script.
_RUN_ERRORS = (ServerStartError, TimeoutError, OSError, subprocess.SubprocessError, http.client.HTTPException)


@dataclass(frozen=True)
class RunnerDeps:
    """Side effects, injectable for tests."""

    backend: Backend
    gpu_query: Callable[[], str] = gpu_mem.nvidia_smi
    gpu_index: int = 0
    poll_interval_s: float = prom_metrics.DEFAULT_INTERVAL_S
    mem_interval_s: float = gpu_mem.DEFAULT_INTERVAL_S


@dataclass(frozen=True)
class Outcome:
    run_id: str
    ok: bool
    skipped: bool
    error: str | None


def ordered_runs(runs: Sequence[mx.RunSpec]) -> list[mx.RunSpec]:
    """Round by round (all rep 0, then rep 1, ...), each round shuffled with a fixed seed.

    A fixed rotation would keep `tuned` right behind `baseline` every round; a per-round shuffle
    varies which of a compared pair runs first and what ran before it (Day 6's position bias)."""
    rounds: dict[int, list[mx.RunSpec]] = {}
    for run in runs:
        rounds.setdefault(run.rep, []).append(run)
    ordered: list[mx.RunSpec] = []
    for rep in sorted(rounds):
        batch = list(rounds[rep])
        random.Random(ORDER_SEED + rep).shuffle(batch)
        ordered += batch
    return ordered


class RunFailure(RuntimeError):
    """The server ran but the benchmark did not produce a usable result."""


@dataclass(frozen=True)
class ResumeDecision:
    skip: bool = False
    problem: str | None = None  # an existing result that must not be silently replaced or reused


def _resume_decision(case: mx.MatrixCase, result_path: Path, sidecar_path: Path) -> ResumeDecision:
    """Reuse an existing result only if it is complete and valid; refuse a foreign configuration."""
    if not result_path.exists():
        return ResumeDecision()
    try:
        found = ingest_gpu.read_json_object(result_path, ingest_gpu.MAX_RESULT_BYTES, "result").get(
            "matrix_config_id")
    except ingest_gpu.ResultFormatError as exc:
        return ResumeDecision(problem=f"existing result {result_path.name} is unreadable ({exc}); "
                                      "use --force to redo it")
    if found != case.config_id:
        return ResumeDecision(problem=(
            f"existing result {result_path.name} was produced by a different configuration "
            f"(config_id {found!r}, expected {case.config_id!r}); use --force or another --out"))
    try:
        if not sidecar_path.is_file():
            raise ingest_gpu.ResultFormatError(f"{result_path.name}: no sidecar (interrupted before it was written)")
        ingest_gpu.load_run(result_path, {case.case_id: case})
    except ingest_gpu.ResultFormatError as exc:
        logger.warning("%s: existing result is not usable (%s); redoing it", result_path.name, exc)
        return ResumeDecision()
    return ResumeDecision(skip=True)


@contextlib.contextmanager
def _server(backend: Backend, cmd: list[str], log_path: Path) -> Iterator[ServerHandle]:
    """Start a server; stop its whole process group on every exit path."""
    handle = backend.start_server(cmd, log_path)
    try:
        yield handle
    finally:
        stop_server(handle)


def _serve_and_bench(
    run: mx.RunSpec, out_dir: Path, deps: RunnerDeps, *, port: int, startup_timeout_s: float,
) -> tuple[dict, dict]:
    """One fresh server, one benchmark, telemetry around it. Returns (poll summary, gpu-mem summary)."""
    case, backend = run.case, deps.backend
    base_url = f"http://{LOOPBACK}:{port}"
    log_path = out_dir / "logs" / f"{run.run_id}.server.log"
    sampler = gpu_mem.PeakMemorySampler(deps.gpu_query, gpu_index=deps.gpu_index, interval_s=deps.mem_interval_s)
    # Exit order: poller -> server teardown -> memory sampler, so teardown is never delayed by telemetry.
    with sampler, _server(backend, case.serve_command(port), log_path) as handle:  # sampler sees weight load
        wait_until_healthy(backend, handle, base_url, timeout_s=startup_timeout_s)
        if handle.poll() is not None:  # a foreign server may have answered /health for ours
            raise ServerStartError("server exited right after /health answered (is another server on this port?)")
        poller = prom_metrics.MetricsPoller(f"{base_url}/metrics", backend.fetch_text, deps.poll_interval_s)
        poller.poll_once()  # baseline for the preemption delta
        with poller:
            returncode = backend.run_bench(case.bench_command(str(out_dir), run.rep, port))
    if returncode != 0:
        raise RunFailure(f"vllm bench serve failed with exit code {returncode}")
    return poller.summary(), asdict(sampler.result())


def execute_run(
    run: mx.RunSpec, out_dir: Path, deps: RunnerDeps, *, port: int, startup_timeout_s: float,
    force: bool = False,
) -> Outcome:
    """Serve, benchmark, tear down. Failures are returned, never swallowed, and never leave a
    result file behind that a later resume could mistake for a finished run."""
    case = run.case
    result_path = out_dir / f"{run.run_id}.json"
    sidecar_path = out_dir / SIDECAR_DIR / f"{run.run_id}.json"

    def refuse(message: str) -> Outcome:  # nothing of ours is on disk to clean up
        logger.error("%s FAILED: %s", run.run_id, message)
        return Outcome(run.run_id, False, False, message)

    def failed(message: str) -> Outcome:
        for leftover in (result_path, sidecar_path):
            leftover.unlink(missing_ok=True)
        return refuse(message)

    if not force:
        decision = _resume_decision(case, result_path, sidecar_path)
        if decision.problem:
            return refuse(decision.problem)
        if decision.skip:
            logger.info("%s: complete result exists, skipping (use --force)", run.run_id)
            return Outcome(run.run_id, True, True, None)
    if not deps.backend.is_port_free(port):
        return refuse(f"port {port} is already in use (stale or foreign server?); refusing to "
                      "benchmark a server this run did not start")
    for stale in (result_path, sidecar_path):
        stale.unlink(missing_ok=True)
    try:
        poll, mem = _serve_and_bench(run, out_dir, deps, port=port, startup_timeout_s=startup_timeout_s)
        if not result_path.exists():
            return failed(f"bench finished but result file {result_path} was not written")
        results_io.write_json_atomic(sidecar_path, {
            "run_id": run.run_id, "matrix_config_id": case.config_id, "poll": poll, "gpu_mem": mem})
    except (*_RUN_ERRORS, RunFailure) as exc:
        return failed(str(exc))
    return Outcome(run.run_id, True, False, None)


def bypass_proxy_for_loopback(environ: Mapping[str, str]) -> dict[str, str]:
    """Copy of `environ` whose NO_PROXY/no_proxy always include loopback.

    `vllm bench serve` trusts the proxy environment, so without this an HTTP(S)_PROXY could route
    the benchmark's requests through a proxy (wrong latencies, prompts leaked). Proxy settings
    themselves are kept: the server may still need them to download weights."""
    out = dict(environ)
    entries: list[str] = []
    for key in ("NO_PROXY", "no_proxy"):  # tools read either spelling, so merge both
        entries += [e.strip() for e in out.get(key, "").split(",") if e.strip()]
    merged = ",".join(dict.fromkeys([*entries, *LOOPBACK_ENTRIES]))
    out["NO_PROXY"] = out["no_proxy"] = merged
    return out


def run_all(
    runs: Sequence[mx.RunSpec], out_dir: Path, deps: RunnerDeps, *, port: int, startup_timeout_s: float,
    force: bool,
) -> list[Outcome]:
    outcomes = []
    for index, run in enumerate(ordered_runs(runs), start=1):
        logger.info("[%d/%d] %s", index, len(runs), run.run_id)
        outcomes.append(execute_run(run, out_dir, deps, port=port,
                                    startup_timeout_s=startup_timeout_s, force=force))
    return outcomes


def exit_code(outcomes: Sequence[Outcome]) -> int:
    return EXIT_OK if all(o.ok for o in outcomes) else EXIT_RUN_FAILED


def _bounded_int(lo: int, hi: int):
    def parse(text: str) -> int:
        value = int(text)
        if not lo <= value <= hi:
            raise argparse.ArgumentTypeError(f"must be in {lo}..{hi}")
        return value
    return parse


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dry-run", action="store_true", help="print the commands; start nothing")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--reps", type=_bounded_int(1, MAX_REPS), default=3)
    p.add_argument("--key-reps", type=_bounded_int(1, MAX_REPS), default=5)
    p.add_argument("--no-control", action="store_true", help="skip the A/A control")
    p.add_argument("--only", nargs="+", metavar="CASE_ID", help="run only these case ids")
    p.add_argument("--force", action="store_true", help="redo runs that already have results")
    p.add_argument("--port", type=_bounded_int(1, MAX_PORT), default=DEFAULT_PORT)
    p.add_argument("--startup-timeout", type=float, default=DEFAULT_STARTUP_TIMEOUT_S)
    p.add_argument("--gpu-index", type=_bounded_int(0, 63), default=0)
    p.add_argument("--model", default=mx.DEFAULT_HUB_MODEL, help="Hub id or local path for vllm serve")
    p.add_argument("--model-preset", default=mx.DEFAULT_PRESET, choices=sorted(kv_sizing.MODELS),
                   help="Day 2 preset used for sizing; must match --model's architecture")
    p.add_argument("--hardware", default=mx.DEFAULT_HARDWARE, choices=sorted(kv_sizing.HARDWARE))
    return p


def _select_runs(args: argparse.Namespace, parser: argparse.ArgumentParser) -> list[mx.RunSpec]:
    try:
        runs = mx.build_runs(args.reps, max(args.key_reps, args.reps),
                             include_control=not args.no_control, model=args.model,
                             model_preset=args.model_preset, hardware=args.hardware)
    except ValueError as exc:
        parser.error(str(exc))
    if args.only:
        known = {r.case.case_id for r in runs}
        unknown = sorted(set(args.only) - known)
        if unknown:
            parser.error(f"unknown case id(s) {unknown}; valid: {sorted(known)}")
        runs = [r for r in runs if r.case.case_id in set(args.only)]
    return runs


def main(argv: Sequence[str] | None = None, deps: RunnerDeps | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    runs = ordered_runs(_select_runs(args, parser))
    if args.dry_run:
        for run in runs:
            print(f"# {run.run_id}  (predicted KV pressure: {run.case.predicted_pressure()['pressured']})")
            print(shlex.join(run.case.serve_command(args.port)))
            print(shlex.join(run.case.bench_command(str(args.out), run.rep, args.port)))
        print(f"# {len(runs)} runs; fresh server each")
        return EXIT_OK
    os.environ.update(bypass_proxy_for_loopback(os.environ))
    deps = deps or RunnerDeps(default_backend(), gpu_index=args.gpu_index)
    args.out = args.out.resolve()  # an --out starting with '-' must never reach vllm as a flag
    args.out.mkdir(parents=True, exist_ok=True)
    try:
        with _interrupt_on_signals():
            outcomes = run_all(runs, args.out, deps, port=args.port,
                               startup_timeout_s=args.startup_timeout, force=args.force)
    except KeyboardInterrupt:
        logger.error("interrupted; the current server was torn down")
        return EXIT_INTERRUPTED
    for o in outcomes:
        if not o.ok:
            logger.error("FAILED %s: %s", o.run_id, o.error)
    return exit_code(outcomes)


if __name__ == "__main__":
    raise SystemExit(main())
