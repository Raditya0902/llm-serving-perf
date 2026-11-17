"""SIMULATED predictions for E4-E6 using the Day 3 scheduler simulator.

These are NOT measurements. The Day 3 cost model uses illustrative constants (an 8B fp16 /
A100-class guess), so only the *direction and mechanism* of an effect carry information:
throughput plateaus, queueing, preemption onset. Absolute numbers must not be quoted as
vLLM results; the real measurement path is RUNBOOK.md + day4's run_sweep.py.

Mapping to vLLM flags (same table as day4-vllm-internals/COMPARISON.md):
  max_num_batched_tokens -> SchedulerConfig.token_budget (chunked prefill on, as in vLLM v1)
  max_num_seqs           -> SchedulerConfig.max_running_seqs
  gpu_memory_utilization -> SchedulerConfig.num_kv_blocks  (only monotonic, not a conversion)

Usage:
    python sim_predictions.py                # writes results/sim/sim_e4|e5|e6.{json,csv}
    python sim_predictions.py --smoke        # tiny, for tests
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import _day1  # noqa: F401  (also provides write_results)
import numpy as np
from run_experiments import write_results  # Day 1's JSON + CSV writer

DAY3_DIR = Path(__file__).resolve().parent.parent / "day3-scheduling"
if str(DAY3_DIR) not in sys.path:
    sys.path.append(str(DAY3_DIR))

from metrics import summarize  # noqa: E402  (Day 3)
from scheduler import SchedulerConfig  # noqa: E402
from simulator import Simulator  # noqa: E402
from workload import WorkloadConfig, generate_workload  # noqa: E402

from experiments import get_experiment  # noqa: E402

RESULTS_DIR = Path(__file__).parent / "results" / "sim"
SIM_NOTE = (
    "SIMULATED: Day 3 cost model with illustrative constants; direction/mechanism only, "
    "not a vLLM or GPU measurement"
)
BASELINE_SEQS, BASELINE_BUDGET = 256, 8192  # the Day 4 sweep baseline
AMPLE_KV_BLOCKS = 8192
KEEP = (
    "throughput_tok_s", "throughput_req_s", "p95_ttft_ms", "p95_token_gap_ms", "p95_wait_ms",
    "mean_batch", "util_proxy", "kv_peak_frac", "preemptions", "recomputed_tokens",
)


@dataclass(frozen=True)
class SimConfig:
    seeds: tuple[int, ...]
    workload: WorkloadConfig
    e6_blocks: tuple[int, ...]


FULL = SimConfig(
    seeds=(0, 1, 2, 3, 4),
    workload=WorkloadConfig(num_requests=100, mean_gap_ms=2.0),  # saturating arrival rate
    e6_blocks=(8192, 4096, 2048, 1024, 512, 256),
)
SMOKE = SimConfig(seeds=(0, 1), workload=WorkloadConfig(num_requests=20, mean_gap_ms=2.0), e6_blocks=(8192, 512))


def _base_config(**changes) -> SchedulerConfig:
    base = SchedulerConfig(
        max_running_seqs=BASELINE_SEQS, token_budget=BASELINE_BUDGET,
        chunked_prefill=True, num_kv_blocks=AMPLE_KV_BLOCKS,
    )
    return replace(base, **changes)


def case_configs(exp_id: str, cfg: SimConfig) -> list[tuple[str, SchedulerConfig, bool]]:
    """(label, scheduler config, is_baseline) for each swept value."""
    exp = get_experiment(exp_id)
    if exp_id == "E4":
        return [(str(v), _base_config(token_budget=v), v == BASELINE_BUDGET) for v in exp.values]
    if exp_id == "E5":
        return [(str(v), _base_config(max_running_seqs=v), v == BASELINE_SEQS) for v in exp.values]
    if exp_id == "E6":
        return [(str(b), _base_config(num_kv_blocks=b), b == AMPLE_KV_BLOCKS) for b in cfg.e6_blocks]
    raise KeyError(f"{exp_id} has no simulator mapping")


def run_sim_experiment(exp_id: str, cfg: SimConfig) -> list[dict]:
    exp = get_experiment(exp_id)
    rows = []
    for label, sched, is_baseline in case_configs(exp_id, cfg):
        per_seed = [
            summarize(Simulator(sched).run(generate_workload(replace(cfg.workload, seed=seed))))
            for seed in cfg.seeds
        ]
        row = {
            "exp_id": exp_id, "variable": exp.variable if exp_id != "E6" else "num_kv_blocks",
            "label": label, "is_baseline": is_baseline, "n_seeds": len(cfg.seeds),
            "simulated": True, "note": SIM_NOTE,
        }
        for key in KEEP:
            values = [float(s[key]) for s in per_seed]
            row[key] = float(np.mean(values))
            row[f"{key}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        rows.append(row)
    return rows


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exp", nargs="+", default=["E4", "E5", "E6"], choices=["E4", "E5", "E6"])
    parser.add_argument("--out", type=Path, default=RESULTS_DIR)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = SMOKE if args.smoke else FULL
    for exp_id in args.exp:
        rows = run_sim_experiment(exp_id, cfg)
        meta = {"simulated": True, "note": SIM_NOTE, "smoke": args.smoke,
                "experiment": get_experiment(exp_id).hypothesis, "workload": vars(cfg.workload)}
        write_results(f"sim_{exp_id.lower()}", rows, meta, args.out)
        print(f"== {exp_id} (SIMULATED) ==")
        for row in rows:
            print({k: round(v, 2) if isinstance(v, float) else v for k, v in row.items()
                   if k in ("label", "throughput_tok_s", "p95_ttft_ms", "p95_token_gap_ms", "preemptions")})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
