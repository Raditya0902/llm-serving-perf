"""`03-batching-simulator` experiments: scheduling strategies on the simulated serving engine.

A  Strategy comparison : sequential / static / continuous / continuous + chunked prefill
B  Full active batch   : sweep max_running_seqs (what happens when the batch is full?)
C  KV exhaustion       : sweep KV pool size, on-demand growth + preemption vs full reservation
D  Long prompt vs decode: a 2k-token prompt arrives while decoders run; sweep token budget
E  Priorities          : FCFS vs priority vs priority + preemption, per-class latency
F  Throughput vs P95   : can throughput improve while tail latency gets worse?
T  Timeline            : ASCII/Gantt data for the 8-request demo scenario

All experiments are simulations (see cost_model.py); times are simulated, not measured on
a GPU. Per-seed rows go to results/exp_x[_smoke].{json,csv}, seed-averaged rows to
results/exp_x_summary[_smoke].{json,csv}.
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
from collections import defaultdict
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

from metrics import by_priority, percentile, summarize
from request import Request
from scheduler import SchedulerConfig
from simulator import SimResult, Simulator
from strategies import STRATEGY_NAMES, StrategyParams, strategy_configs
from timeline import demo_scenario, render_ascii
from workload import WorkloadConfig, generate_workload, long_prompt_scenario

RESULTS_DIR = Path(__file__).parent / "results"
Row = dict[str, object]
EXPERIMENTS = ("A", "B", "C", "D", "E", "F", "T")
KEY_COLUMNS = (
    "throughput_tok_s", "mean_latency_ms", "p95_latency_ms", "util_proxy",
    "mean_ttft_ms", "p95_token_gap_ms", "worst_itl_p95_ms", "mean_wait_ms",
)
# Experiment F only counts a "throughput up, tail worse" event when both changes are material,
# so noise-level wiggles at the saturation plateau are not reported as trade-offs.
MIN_THROUGHPUT_GAIN_PCT = 2.0
MIN_TAIL_WORSENING_PCT = 2.0


@dataclass(frozen=True)
class ExperimentConfig:
    workload: WorkloadConfig
    seeds: tuple[int, ...]
    params: StrategyParams
    b_batch_sizes: tuple[int, ...]
    c_kv_blocks: tuple[int, ...]
    d_budgets: tuple[int, ...]
    d_num_decoders: int
    e_max_running: int
    f_continuous_sizes: tuple[int, ...]
    f_static_sizes: tuple[int, ...]


FULL = ExperimentConfig(
    workload=WorkloadConfig(),
    seeds=(0, 1, 2, 3, 4),
    params=StrategyParams(),
    b_batch_sizes=(1, 2, 4, 8, 16, 32, 64, 128),
    c_kv_blocks=(3072, 1024, 768, 512, 384, 256, 192),
    d_budgets=(2048, 1024, 512, 256, 128),
    d_num_decoders=16,
    e_max_running=16,
    f_continuous_sizes=(4, 8, 16, 24, 32, 48),  # below the saturation plateau (>= ~48 slots)
    f_static_sizes=(8, 16, 32, 64),
)
SMOKE = replace(
    FULL,
    workload=WorkloadConfig(num_requests=20),
    seeds=(0, 1),
    b_batch_sizes=(1, 8, 32),
    c_kv_blocks=(3072, 384),
    d_budgets=(1024, 256),
    d_num_decoders=4,
    f_continuous_sizes=(16, 32),
    f_static_sizes=(8, 16),
)


# ------------------------------------------------------------------ helpers
def _workload(cfg: ExperimentConfig, seed: int) -> list[Request]:
    return generate_workload(replace(cfg.workload, seed=seed))


def _simulate_cases(
    exp: str, cases: list[tuple[Row, SchedulerConfig]], cfg: ExperimentConfig
) -> list[Row]:
    """Run every (labels, scheduler config) case on every seed's workload."""
    rows: list[Row] = []
    for seed in cfg.seeds:
        for labels, sched in cases:
            result = Simulator(sched).run(_workload(cfg, seed))
            rows.append({"exp": exp, **labels, "seed": seed, **summarize(result)})
    return rows


def aggregate(rows: list[Row], group_keys: tuple[str, ...]) -> list[Row]:
    """Mean and sample std (ddof=1; 0 for a single seed) over seeds, grouped by `group_keys`."""
    groups: dict[tuple, list[Row]] = defaultdict(list)
    for r in rows:
        groups[tuple(r[k] for k in group_keys)].append(r)
    out: list[Row] = []
    for key, members in groups.items():
        row: Row = dict(zip(group_keys, key))
        row["n_seeds"] = len(members)
        numeric = [
            c for c, v in members[0].items()
            if c not in group_keys and c != "seed" and isinstance(v, (int, float)) and not isinstance(v, bool)
        ]
        for col in numeric:
            vals = [float(m[col]) for m in members]
            row[col] = float(np.mean(vals))
            row[f"{col}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        out.append(row)
    return out


def write_results(name: str, rows: list[Row], meta: dict, out_dir: Path) -> None:
    """Write {meta, rows} JSON plus a CSV of the scalar columns."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{name}.json").write_text(json.dumps({"meta": meta, "rows": rows}, indent=2))
    if not rows:
        return
    columns = list(dict.fromkeys(k for r in rows for k in r))
    with (out_dir / f"{name}.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def run_metadata(cfg: ExperimentConfig, smoke: bool) -> dict:
    return {
        "smoke": smoke,
        "platform": platform.platform(),
        "note": "simulated times from cost_model.CostModel; utilisation is a proxy, not GPU utilisation",
        "config": asdict(cfg),
    }


def _fmt(v: object) -> str:
    if isinstance(v, float):
        return f"{v:,.3f}" if abs(v) < 10 else f"{v:,.0f}"
    return str(v)


def print_table(title: str, rows: list[Row], columns: list[str]) -> None:
    print(f"\n== {title}")
    widths = [max(len(c), *(len(_fmt(r.get(c, ""))) for r in rows)) for c in columns]
    print("  ".join(c.ljust(w) for c, w in zip(columns, widths)))
    for r in rows:
        print("  ".join(_fmt(r.get(c, "")).ljust(w) for c, w in zip(columns, widths)))


def _save(exp: str, rows: list[Row], group_keys: tuple[str, ...], meta: dict, out: Path, suffix: str,
          show: list[str]) -> None:
    summary = aggregate(rows, group_keys)
    write_results(f"exp_{exp.lower()}{suffix}", rows, meta, out)
    write_results(f"exp_{exp.lower()}_summary{suffix}", summary, meta, out)
    print_table(f"Experiment {exp} (mean over {len(set(r['seed'] for r in rows))} seeds)", summary,
                list(group_keys) + show)


# -------------------------------------------------------------- experiments
def exp_a(cfg: ExperimentConfig) -> tuple[list[Row], tuple[str, ...], list[str]]:
    cases = [({"strategy": n}, c) for n, c in strategy_configs(cfg.params).items()]
    cols = list(KEY_COLUMNS) + ["mean_batch", "busy_frac", "padded_slot_frac", "kv_peak_frac"]
    return _simulate_cases("A", cases, cfg), ("strategy",), cols


def exp_b(cfg: ExperimentConfig) -> tuple[list[Row], tuple[str, ...], list[str]]:
    cases = [
        ({"max_running_seqs": n}, SchedulerConfig(max_running_seqs=n, token_budget=cfg.params.unchunked_budget,
                                                  num_kv_blocks=cfg.params.num_kv_blocks,
                                                  block_size=cfg.params.block_size))
        for n in cfg.b_batch_sizes
    ]
    cols = ["throughput_tok_s", "mean_wait_ms", "p95_wait_ms", "peak_waiting", "mean_batch",
            "p95_latency_ms", "p95_token_gap_ms", "preemptions", "kv_peak_frac"]
    return _simulate_cases("B", cases, cfg), ("max_running_seqs",), cols


def exp_c(cfg: ExperimentConfig) -> tuple[list[Row], tuple[str, ...], list[str]]:
    cases = [
        ({"kv_mode": mode, "num_kv_blocks": blocks},
         SchedulerConfig(max_running_seqs=cfg.params.max_running_seqs, token_budget=cfg.params.unchunked_budget,
                         num_kv_blocks=blocks, block_size=cfg.params.block_size, kv_mode=mode))
        for mode in ("on_demand", "full_reserve")
        for blocks in cfg.c_kv_blocks
    ]
    cols = ["throughput_tok_s", "p95_latency_ms", "mean_batch", "preemptions", "recomputed_tokens",
            "kv_peak_frac", "mean_wait_ms"]
    return _simulate_cases("C", cases, cfg), ("kv_mode", "num_kv_blocks"), cols


def _decoder_stats(result: SimResult, num_decoders: int) -> Row:
    decoders = [r for r in result.requests if r.request_id < num_decoders]
    longs = [r for r in result.requests if r.request_id >= num_decoders]
    itls = [r.max_itl for r in decoders]
    return {
        "decoder_max_itl_ms": max(itls),
        "decoder_p95_itl_ms": percentile(itls, 95),
        "decoder_mean_latency_ms": float(np.mean([r.latency for r in decoders])),
        "long_mean_ttft_ms": float(np.mean([r.ttft for r in longs])) if longs else 0.0,
        "long_mean_latency_ms": float(np.mean([r.latency for r in longs])) if longs else 0.0,
    }


def exp_d(cfg: ExperimentConfig) -> tuple[list[Row], tuple[str, ...], list[str]]:
    n = cfg.d_num_decoders
    p = cfg.params
    base = dict(max_running_seqs=p.max_running_seqs, num_kv_blocks=p.num_kv_blocks, block_size=p.block_size)
    cases: list[tuple[Row, SchedulerConfig, tuple[float, ...]]] = [
        ({"mode": "no long prompt (reference)", "token_budget": p.unchunked_budget},
         SchedulerConfig(token_budget=p.unchunked_budget, **base), ()),
        ({"mode": "unchunked", "token_budget": p.unchunked_budget},
         SchedulerConfig(token_budget=p.unchunked_budget, **base), (400.0, 1200.0)),
    ] + [
        ({"mode": "chunked", "token_budget": b},
         SchedulerConfig(token_budget=b, chunked_prefill=True, **base), (400.0, 1200.0))
        for b in cfg.d_budgets
    ]
    rows: list[Row] = []
    for labels, sched, arrivals in cases:
        result = Simulator(sched).run(long_prompt_scenario(num_decoders=n, long_arrivals_ms=arrivals))
        rows.append({"exp": "D", **labels, "seed": 0, **summarize(result), **_decoder_stats(result, n)})
    cols = ["decoder_max_itl_ms", "decoder_p95_itl_ms", "decoder_mean_latency_ms", "long_mean_ttft_ms",
            "throughput_tok_s"]
    return rows, ("mode", "token_budget"), cols


def exp_e(cfg: ExperimentConfig) -> tuple[list[Row], tuple[str, ...], list[str]]:
    base = dict(max_running_seqs=cfg.e_max_running, token_budget=cfg.params.unchunked_budget,
                num_kv_blocks=cfg.params.num_kv_blocks, block_size=cfg.params.block_size)
    cases = [
        ("fcfs", SchedulerConfig(policy="fcfs", **base)),
        ("priority", SchedulerConfig(policy="priority", **base)),
        ("priority+preempt", SchedulerConfig(policy="priority", preempt_for_priority=True, **base)),
    ]
    rows: list[Row] = []
    for seed in cfg.seeds:
        for name, sched in cases:
            result = Simulator(sched).run(_workload(cfg, seed))
            overall = summarize(result)
            for prio, stats in by_priority(result).items():
                rows.append({
                    "exp": "E", "policy": name, "priority_class": prio, "seed": seed,
                    **{f"class_{k}": v for k, v in stats.items()},
                    "throughput_tok_s": overall["throughput_tok_s"],
                    "overall_p95_latency_ms": overall["p95_latency_ms"],
                    "recomputed_tokens": overall["recomputed_tokens"],
                })
    cols = ["class_n", "class_mean_latency_ms", "class_p95_latency_ms", "class_mean_wait_ms",
            "class_mean_ttft_ms", "class_preemptions", "throughput_tok_s", "recomputed_tokens"]
    return rows, ("policy", "priority_class"), cols


def exp_f(cfg: ExperimentConfig) -> tuple[list[Row], tuple[str, ...], list[str]]:
    p = cfg.params
    cases: list[tuple[Row, SchedulerConfig]] = []
    for n in cfg.f_continuous_sizes:
        cases.append(({"family": "continuous", "max_running_seqs": n},
                      SchedulerConfig(max_running_seqs=n, token_budget=p.unchunked_budget, num_kv_blocks=p.num_kv_blocks, block_size=p.block_size)))
        cases.append(({"family": "continuous_chunked", "max_running_seqs": n},
                      SchedulerConfig(max_running_seqs=n, token_budget=p.chunked_budget, chunked_prefill=True,
                                      num_kv_blocks=p.num_kv_blocks, block_size=p.block_size)))
    for n in cfg.f_static_sizes:
        cases.append(({"family": "static", "max_running_seqs": n},
                      SchedulerConfig(kind="static", max_running_seqs=n, token_budget=p.unchunked_budget,
                                      num_kv_blocks=p.num_kv_blocks, block_size=p.block_size)))
    cols = ["throughput_tok_s", "mean_latency_ms", "p95_latency_ms", "p95_token_gap_ms", "worst_itl_p95_ms", "mean_batch"]
    return _simulate_cases("F", cases, cfg), ("family", "max_running_seqs"), cols


def _pct_change(before: float, after: float) -> float:
    return (after - before) / before * 100 if before else 0.0


def tradeoff_pairs(rows: list[Row]) -> list[Row]:
    """For consecutive sizes in each family: did throughput rise while the tail got worse?

    Counted per seed, and only when both moves are material (throughput up by at least
    MIN_THROUGHPUT_GAIN_PCT and the tail up by at least MIN_TAIL_WORSENING_PCT). Two tails:
    end-to-end P95 latency, and the per-token P95 gap (`p95_token_gap_ms`).
    """
    by_key: dict[tuple, dict[int, Row]] = defaultdict(dict)
    for r in rows:
        by_key[(r["family"], r["seed"])][int(r["max_running_seqs"])] = r
    families = sorted({k[0] for k in by_key})
    seeds = sorted({k[1] for k in by_key})
    out: list[Row] = []
    for fam in families:
        sizes = sorted(by_key[(fam, seeds[0])])
        for lo, hi in zip(sizes, sizes[1:]):
            pairs = [(by_key[(fam, s)][lo], by_key[(fam, s)][hi]) for s in seeds]

            def mean_change(col: str) -> float:
                return float(np.mean([_pct_change(a[col], b[col]) for a, b in pairs]))

            def seeds_worse(tail: str) -> int:
                return sum(
                    _pct_change(a["throughput_tok_s"], b["throughput_tok_s"]) >= MIN_THROUGHPUT_GAIN_PCT
                    and _pct_change(a[tail], b[tail]) >= MIN_TAIL_WORSENING_PCT
                    for a, b in pairs
                )

            out.append({
                "family": fam, "from_seqs": lo, "to_seqs": hi,
                "throughput_change_pct": mean_change("throughput_tok_s"),
                "p95_latency_change_pct": mean_change("p95_latency_ms"),
                "p95_token_gap_change_pct": mean_change("p95_token_gap_ms"),
                "seeds": len(seeds),
                "seeds_tput_up_and_p95_latency_worse": seeds_worse("p95_latency_ms"),
                "seeds_tput_up_and_tail_gap_worse": seeds_worse("p95_token_gap_ms"),
            })
    return out


def run_timeline(out_dir: Path, suffix: str) -> None:
    requests, sched = demo_scenario()
    result = Simulator(sched).run(requests)
    text = render_ascii(result, width=100)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"timeline{suffix}.txt").write_text(text + "\n")
    print("\n== Timeline (demo scenario)\n" + text)


RUNNERS = {"A": exp_a, "B": exp_b, "C": exp_c, "D": exp_d, "E": exp_e, "F": exp_f}


def run_experiment(name: str, cfg: ExperimentConfig, out: Path, suffix: str, smoke: bool) -> None:
    meta = run_metadata(cfg, smoke)
    if name == "T":
        run_timeline(out, suffix)
        return
    rows, group_keys, show = RUNNERS[name](cfg)
    _save(name, rows, group_keys, meta, out, suffix, show)
    if name == "F":
        pairs = tradeoff_pairs(rows)
        write_results(f"exp_f_pairs{suffix}", pairs, meta, out)
        print_table("Experiment F: consecutive-size trade-offs", pairs,
                    ["family", "from_seqs", "to_seqs", "throughput_change_pct", "p95_latency_change_pct",
                     "p95_token_gap_change_pct", "seeds_tput_up_and_p95_latency_worse",
                     "seeds_tput_up_and_tail_gap_worse"])


def _main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exp", default="all", choices=[*EXPERIMENTS, "all"])
    parser.add_argument("--smoke", action="store_true", help="tiny sizes; files get a _smoke suffix")
    parser.add_argument("--out", type=Path, default=RESULTS_DIR)
    parser.add_argument("--seed", type=int, default=0, help="first workload seed (others follow)")
    args = parser.parse_args(argv)
    base = SMOKE if args.smoke else FULL
    cfg = replace(base, seeds=tuple(args.seed + i for i in range(len(base.seeds))))
    suffix = "_smoke" if args.smoke else ""
    for name in EXPERIMENTS if args.exp == "all" else (args.exp,):
        run_experiment(name, cfg, args.out, suffix, args.smoke)


if __name__ == "__main__":
    _main()
