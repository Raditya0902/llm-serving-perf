"""Data-parallel baseline for the 8B model: two TP=1 replicas, one per GPU, against one TP=2 server.

    .venv/bin/python run_dp_gpu.py --out results/raw/dp --hardware a100-40gb --cap-kv-pool --model ... --model-preset llama3-8b
    .venv/bin/python run_dp_gpu.py --dry-run --out /tmp/x

For each repeat it starts a FRESH server per GPU (pinned with CUDA_VISIBLE_DEVICES, ports PORT and PORT+1; never 8001,
which RunPod's nginx owns), runs workload B at half the total concurrency and half the prompts on each replica at the same
time, and writes one result + sidecar per replica (`B-c32-dp2a`, `B-c32-dp2b`). `summarize_followup.py` combines each repeat
into one row. Same per-run telemetry and teardown as `run_matrix_gpu.py`.
"""
from __future__ import annotations

import argparse
import contextlib
import logging
import os
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

import _prior  # noqa: F401  (puts folders 02-05 on sys.path)

# isort: split
from run_sweep import default_backend, wait_until_healthy

import gpu_mem
import matrix as mx
import prom_metrics
import results_io
import run_matrix_gpu as rg

DEFAULT_TOTALS = (16, 64)
REPLICAS = 2
SEED_STRIDE = 100  # replica i draws prompts from seed BASE_SEED + rep + i * SEED_STRIDE
DEFAULT_PORT = 18000
DEFAULT_REPS = 3
logger = logging.getLogger("run_dp_gpu")


def replica_cases(model: str, model_preset: str, hardware: str, cap_kv_pool: bool,
                  total_concurrency: int) -> list[mx.MatrixCase]:
    """The two replicas' cases: workload B, baseline knobs, half the concurrency each (like `05-tensor-parallel`'s DP arm)."""
    if total_concurrency < REPLICAS or total_concurrency % REPLICAS:
        raise ValueError(f"total concurrency must be even and >= {REPLICAS}, got {total_concurrency}")
    half = total_concurrency // REPLICAS
    return [
        mx.MatrixCase(mx.WORKLOADS["B"], mx.ServingConfig(f"dp2{'ab'[i]}", 0.85, 8192), half, model, model_preset,
                      hardware, cap_kv_pool, seed_offset=i * SEED_STRIDE)
        for i in range(REPLICAS)
    ]


def pinned_serve_command(case: mx.MatrixCase, port: int, *, gpu: int) -> list[str]:
    return ["env", f"CUDA_VISIBLE_DEVICES={gpu}", *case.serve_command(port)]


def _paths(case: mx.MatrixCase, rep: int, out_dir: Path) -> tuple[Path, Path]:
    run_id = f"{case.case_id}__r{rep}"
    return out_dir / f"{run_id}.json", out_dir / rg.SIDECAR_DIR / f"{run_id}.json"


def _bench(deps: rg.RunnerDeps, case: mx.MatrixCase, rep: int, out_dir: Path, port: int) -> int:
    return deps.backend.run_bench(case.bench_command(str(out_dir), rep, port))


def execute_dp_run(
    cases: Sequence[mx.MatrixCase], rep: int, out_dir: Path, deps: rg.RunnerDeps, *, port: int,
    startup_timeout_s: float, force: bool,
) -> rg.Outcome:
    run_id = f"{cases[0].case_id.rsplit('-', 1)[0]}__r{rep}"  # e.g. B-c32__r0
    paths = [_paths(c, rep, out_dir) for c in cases]
    ports = [port + i for i in range(len(cases))]
    backend = deps.backend

    def fail(message: str) -> rg.Outcome:
        for leftover in (p for pair in paths for p in pair):
            leftover.unlink(missing_ok=True)
        logger.error("%s FAILED: %s", run_id, message)
        return rg.Outcome(run_id, False, False, message)

    if not force:
        decisions = [rg._resume_decision(c, *p) for c, p in zip(cases, paths, strict=True)]
        problems = [d.problem for d in decisions if d.problem]
        if problems:
            logger.error("%s FAILED: %s", run_id, problems[0])
            return rg.Outcome(run_id, False, False, problems[0])
        if all(d.skip for d in decisions):
            logger.info("%s: complete results exist, skipping (use --force)", run_id)
            return rg.Outcome(run_id, True, True, None)
    busy = [p for p in ports if not backend.is_port_free(p)]
    if busy:
        return fail(f"port {busy[0]} is already in use (stale or foreign server?); refusing to benchmark "
                    "a server this run did not start")
    for pair in paths:
        for stale in pair:
            stale.unlink(missing_ok=True)
    try:
        polls, mems = _serve_and_bench_all(cases, rep, out_dir, deps, ports, startup_timeout_s)
        for case, (result_path, sidecar_path), poll, mem in zip(cases, paths, polls, mems, strict=True):
            if not result_path.exists():
                return fail(f"bench finished but result file {result_path} was not written")
            results_io.write_json_atomic(sidecar_path, {
                "run_id": f"{case.case_id}__r{rep}", "matrix_config_id": case.config_id, "poll": poll, "gpu_mem": mem})
    except (*rg._RUN_ERRORS, rg.RunFailure) as exc:
        return fail(str(exc))
    return rg.Outcome(run_id, True, False, None)


def _serve_and_bench_all(cases, rep, out_dir, deps, ports, startup_timeout_s):
    backend = deps.backend
    with contextlib.ExitStack() as stack:
        samplers = [stack.enter_context(gpu_mem.PeakMemorySampler(
            deps.gpu_query, gpu_index=i, interval_s=deps.mem_interval_s)) for i in range(len(cases))]
        handles = [stack.enter_context(rg._server(
            backend, pinned_serve_command(c, p, gpu=i), out_dir / "logs" / f"{c.case_id}__r{rep}.server.log"))
            for i, (c, p) in enumerate(zip(cases, ports, strict=True))]
        for handle, port in zip(handles, ports, strict=True):
            wait_until_healthy(backend, handle, f"http://{rg.LOOPBACK}:{port}", timeout_s=startup_timeout_s)
            if handle.poll() is not None:
                raise rg.ServerStartError("server exited right after /health answered (is another server on this port?)")
        pollers = [prom_metrics.MetricsPoller(f"http://{rg.LOOPBACK}:{p}/metrics", backend.fetch_text, deps.poll_interval_s)
                   for p in ports]
        for poller in pollers:
            poller.poll_once()
        for poller in pollers:
            stack.enter_context(poller)
        with ThreadPoolExecutor(max_workers=len(cases)) as pool:
            codes = list(pool.map(lambda cp: _bench(deps, cp[0], rep, out_dir, cp[1]), zip(cases, ports, strict=True)))
    if any(codes):
        raise rg.RunFailure(f"vllm bench serve failed with exit code {next(c for c in codes if c)}")
    return [p.summary() for p in pollers], [asdict(s.result()) for s in samplers]


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--totals", nargs="+", type=int, default=list(DEFAULT_TOTALS), help="total concurrency (split in two)")
    p.add_argument("--reps", type=rg._bounded_int(1, rg.MAX_REPS), default=DEFAULT_REPS)
    p.add_argument("--model", default=mx.DEFAULT_HUB_MODEL)
    p.add_argument("--model-preset", default=mx.DEFAULT_PRESET)
    p.add_argument("--hardware", default=mx.DEFAULT_HARDWARE)
    p.add_argument("--cap-kv-pool", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--port", type=rg._bounded_int(1, rg.MAX_PORT - 1), default=DEFAULT_PORT)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    groups = [replica_cases(args.model, args.model_preset, args.hardware, args.cap_kv_pool, t) for t in args.totals]
    if args.dry_run:
        for cases in groups:
            for i, case in enumerate(cases):
                print(f"# {case.case_id} (GPU {i}, {args.reps} reps)")
                print(" ".join(pinned_serve_command(case, args.port + i, gpu=i)))
        return rg.EXIT_OK
    os.environ.update(rg.bypass_proxy_for_loopback(os.environ))
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    deps = rg.RunnerDeps(default_backend())
    outcomes = []
    for rep in range(args.reps):
        for cases in groups:
            outcomes.append(execute_dp_run(cases, rep, out, deps, port=args.port,
                                           startup_timeout_s=rg.DEFAULT_STARTUP_TIMEOUT_S, force=False))
    for o in outcomes:
        if not o.ok:
            logging.error("FAILED %s: %s", o.run_id, o.error)
    return rg.exit_code(outcomes)


if __name__ == "__main__":
    raise SystemExit(main())
