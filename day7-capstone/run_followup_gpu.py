"""Follow-up runs for the Day 7 matrix: H6 (admission caps) and a decomposition of the `tuned` config.

`tuned` changes two knobs at once (gpu_memory_utilization 0.85 -> 0.92, max_num_batched_tokens 8192 -> 2048). With
--cap-kv-pool the first one becomes a pool-size change (17.0 -> 19.8 GiB). These arms change ONE knob at a time on
workload B at concurrency 64, using the same fresh-server / bench / telemetry path as `run_matrix_gpu.py`:

  cap16 / cap24 / cap32 : baseline + --max-num-seqs N              (H6, 1 rep each)
  budget2048            : baseline + --max-num-batched-tokens 2048 (3 reps)
  pool092               : baseline + gpu_memory_utilization 0.92   (3 reps; with --cap-kv-pool = bigger pool only)

Results land in --out in the same layout as the matrix (result JSON + sidecar). They are NOT ingestible by
`ingest_gpu.py` (its case index only knows the 12 matrix cases); `summarize_followup.py` reads them directly.
"""
from __future__ import annotations

import argparse
import logging
import os
from collections.abc import Sequence
from pathlib import Path

import matrix as mx
import run_matrix_gpu as rm
from run_sweep import default_backend

ARMS: dict[str, tuple[mx.ServingConfig, int]] = {
    "cap16": (mx.ServingConfig("cap16", 0.85, 8192, max_num_seqs=16), 1),
    "cap24": (mx.ServingConfig("cap24", 0.85, 8192, max_num_seqs=24), 1),
    "cap32": (mx.ServingConfig("cap32", 0.85, 8192, max_num_seqs=32), 1),
    "budget2048": (mx.ServingConfig("budget2048", 0.85, 2048), 3),
    "pool092": (mx.ServingConfig("pool092", 0.92, 8192), 3),
}
CONCURRENCY = 64


def build_runs(model: str, model_preset: str, hardware: str, cap_kv_pool: bool,
               arms: Sequence[str]) -> list[mx.RunSpec]:
    runs: list[mx.RunSpec] = []
    for name in arms:
        cfg, reps = ARMS[name]
        case = mx.MatrixCase(mx.WORKLOADS["B"], cfg, CONCURRENCY, model, model_preset, hardware, cap_kv_pool)
        runs += [mx.RunSpec(case, rep) for rep in range(reps)]
    return runs


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--arms", nargs="+", choices=sorted(ARMS), default=list(ARMS))
    p.add_argument("--model", default=mx.DEFAULT_HUB_MODEL)
    p.add_argument("--model-preset", default=mx.DEFAULT_PRESET)
    p.add_argument("--hardware", default=mx.DEFAULT_HARDWARE)
    p.add_argument("--cap-kv-pool", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--port", type=int, default=rm.DEFAULT_PORT)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    runs = build_runs(args.model, args.model_preset, args.hardware, args.cap_kv_pool, args.arms)
    if args.dry_run:
        for run in runs:
            print(f"# {run.run_id}")
            print(" ".join(run.case.serve_command(args.port)))
        return rm.EXIT_OK
    os.environ.update(rm.bypass_proxy_for_loopback(os.environ))
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    deps = rm.RunnerDeps(default_backend())
    outcomes = rm.run_all(runs, out, deps, port=args.port, startup_timeout_s=rm.DEFAULT_STARTUP_TIMEOUT_S,
                          force=False)
    for o in outcomes:
        if not o.ok:
            logging.error("FAILED %s: %s", o.run_id, o.error)
    return rm.exit_code(outcomes)


if __name__ == "__main__":
    raise SystemExit(main())
