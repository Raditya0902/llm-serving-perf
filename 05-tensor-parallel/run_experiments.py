"""Day 5 experiments: what does communication cost, and when does tensor parallelism pay?

A  Collective benchmark : all_reduce / all_gather / broadcast, latency + bandwidth vs message
                          size, for 2 and 4 ranks; also a fitted alpha (latency) / beta (per byte)
B  TP MLP vs one process: compute vs communication time of a column->row MLP, sweeping
                          hidden size and token count (does halving the FLOPs beat the AllReduce?)
C  Communication model  : does A's alpha/beta predict the AllReduce time B actually saw, and the
                          speedup built from it? Plus GPU rows from ASSUMED hardware numbers:
                          hypotheses, not measurements.

A and B are real measurements on CPU with the Gloo backend, one intra-op thread per rank,
all ranks on this one machine (loopback sockets), median of `repeats` runs with the relative
spread recorded. They say nothing quantitative about NVLink, PCIe or NCCL.
Results go to results/exp_x[_smoke].{json,csv}.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, replace
from typing import Callable
from pathlib import Path

import torch

from torch.multiprocessing.spawn import ProcessExitedException

from bench_model import (
    bus_bandwidth_factor,
    fit_alpha_beta,
    fit_max_rel_error,
    median_across_runs,
    merge_ranks,
    mlp_cost,
    predict_tp_speedup,
)
from bench_workers import bench_collectives_worker, bench_tp_mlp_worker
from dist_utils import spawn_group

RESULTS_DIR = Path(__file__).parent / "results"
Row = dict[str, object]
EXPERIMENTS = ("A", "B", "C")
KB = 1024
FLOAT32_BYTES = 4
SPAWN_ATTEMPTS = 2  # Gloo occasionally kills a rank at start-up; one retry, then fail loudly


@dataclass(frozen=True)
class ExperimentConfig:
    worlds: tuple[int, ...]
    message_bytes: tuple[int, ...]
    collective_iters: int
    warmup: int
    mlp_worlds: tuple[int, ...]
    mlp_d_models: tuple[int, ...]
    mlp_tokens: tuple[int, ...]
    mlp_iters: int
    repeats: int
    ff_mult: int = 4
    seed: int = 0


FULL = ExperimentConfig(
    worlds=(2, 4),
    message_bytes=tuple(KB * 4**i for i in range(9)),  # 1 KiB .. 64 MiB per rank
    collective_iters=20,
    warmup=3,
    mlp_worlds=(2, 4),
    mlp_d_models=(256, 512, 1024, 2048),
    mlp_tokens=(1, 16, 128, 512),
    mlp_iters=10,
    repeats=3,
)
SMOKE = ExperimentConfig(
    worlds=(2,),
    message_bytes=(KB, 16 * KB),
    collective_iters=3,
    warmup=1,
    mlp_worlds=(2,),
    mlp_d_models=(32,),
    mlp_tokens=(4, 16),
    mlp_iters=3,
    repeats=1,
)

# ASSUMED, not measured: round numbers in the range of datacentre GPUs, for the hypothesis rows.
QWEN_1_5B_MLP = {"d_model": 1536, "d_ff": 8960, "n_matrices": 3, "bytes_per_elem": 2}  # SwiGLU, fp16
GPU_TOKEN_COUNTS = (1, 8, 64, 512, 4096)
GPU_WORLDS = (2, 4)
GPU_ASSUMPTIONS = {
    "nvlink_class": {"flops": 200e12, "mem_bw": 1.5e12, "alpha_s": 15e-6, "busbw": 200e9},
    "pcie_class": {"flops": 200e12, "mem_bw": 1.5e12, "alpha_s": 30e-6, "busbw": 12e9},
}


# ------------------------------------------------------------------ row builders (pure)
def collective_rows(world: int, merged: list[dict]) -> list[Row]:
    rows: list[Row] = []
    for m in merged:
        # all_gather moves world x the per-rank shard; the other two move one message.
        transfer = m["size_bytes"] * (world if m["collective"] == "all_gather" else 1)
        algbw = transfer / m["median_s"] / 1e9
        rows.append(
            {
                "world": world,
                "collective": m["collective"],
                "size_bytes": m["size_bytes"],
                "transfer_bytes": transfer,
                "latency_us": m["median_s"] * 1e6,
                "latency_spread": m.get("median_s_spread", 0.0),
                "min_latency_us": m["min_s"] * 1e6,
                "algbw_GBps": algbw,
                "busbw_GBps": algbw * bus_bandwidth_factor(m["collective"], world),
            }
        )
    return rows


def fit_rows(rows: list[Row]) -> list[Row]:
    """One alpha/beta fit per (world, collective) over all message sizes."""
    fits: list[Row] = []
    for key in dict.fromkeys((r["world"], r["collective"]) for r in rows):
        group = [r for r in rows if (r["world"], r["collective"]) == key]
        sizes = [r["transfer_bytes"] for r in group]
        times = [r["latency_us"] / 1e6 for r in group]
        alpha_s, beta_s = fit_alpha_beta(sizes, times)
        fits.append(
            {
                "world": key[0],
                "collective": key[1],
                "alpha_us": alpha_s * 1e6,
                "beta_ns_per_byte": beta_s * 1e9,
                "implied_bw_GBps": (1.0 / beta_s / 1e9) if beta_s > 0 else float("nan"),
                "points": len(group),
                "max_rel_err": fit_max_rel_error(sizes, times, alpha_s, beta_s),
            }
        )
    return fits


def mlp_rows(world: int, merged: list[dict]) -> list[Row]:
    return [
        {
            "world": world,
            "d_model": m["d_model"],
            "tokens": m["tokens"],
            "ref_ms": m["ref_s"] * 1e3,
            "ref_spread": m.get("ref_s_spread", 0.0),
            "tp_total_ms": m["tp_total_s"] * 1e3,
            "tp_total_spread": m.get("tp_total_s_spread", 0.0),
            "tp_compute_ms": m["tp_compute_s"] * 1e3,
            "tp_comm_ms": m["tp_comm_s"] * 1e3,
            "speedup": m["ref_s"] / m["tp_total_s"],
            "compute_speedup": m["ref_s"] / m["tp_compute_s"],
            "comm_fraction": m["tp_comm_s"] / m["tp_total_s"],
            "weight_fraction": m["shard_weight_bytes"] / m["full_weight_bytes"],
            "max_err": m["max_err"],
        }
        for m in merged
    ]


def cpu_model_rows(fits: list[Row], measured: list[Row]) -> list[Row]:
    """Test A's alpha/beta against B: predict each case's AllReduce time, then the speedup that
    results from B's *measured* compute time plus that predicted communication time."""
    ar_fit = {f["world"]: f for f in fits if f["collective"] == "all_reduce"}
    rows: list[Row] = []
    for m in measured:
        fit = ar_fit.get(m["world"])
        if fit is None:
            continue
        message_bytes = m["tokens"] * m["d_model"] * FLOAT32_BYTES
        predicted_comm_ms = (fit["alpha_us"] + fit["beta_ns_per_byte"] * 1e-3 * message_bytes) / 1e3
        predicted_total_ms = m["tp_compute_ms"] + predicted_comm_ms
        rows.append(
            {
                "source": "measured_cpu_model",
                "hardware": "this_mac_gloo_1thread",
                "world": m["world"], "d_model": m["d_model"], "tokens": m["tokens"],
                "predicted_comm_ms": predicted_comm_ms,
                "measured_comm_ms": m["tp_comm_ms"],
                "comm_model_ratio": predicted_comm_ms / m["tp_comm_ms"],
                "predicted_speedup": m["ref_ms"] / predicted_total_ms,
                "measured_speedup": m["speedup"],
            }
        )
    return rows


def gpu_assumption_rows() -> list[Row]:
    """Hypothesis rows: Qwen2.5-1.5B-sized MLP on assumed GPU/interconnect numbers."""
    dims = QWEN_1_5B_MLP
    rows: list[Row] = []
    for hardware, hw in GPU_ASSUMPTIONS.items():
        for world in GPU_WORLDS:
            beta = bus_bandwidth_factor("all_reduce", world) / hw["busbw"]
            for tokens in GPU_TOKEN_COUNTS:
                flops, weight_bytes, comm_bytes = mlp_cost(
                    tokens, dims["d_model"], dims["d_ff"], dims["n_matrices"], dims["bytes_per_elem"]
                )
                p = predict_tp_speedup(
                    flops=flops, weight_bytes=weight_bytes, comm_bytes=comm_bytes,
                    device_flops=hw["flops"], device_bw=hw["mem_bw"],
                    comm_alpha_s=hw["alpha_s"], comm_beta_s_per_byte=beta, world=world,
                )
                rows.append(
                    {
                        "source": "assumption", "hardware": hardware, "world": world,
                        "d_model": dims["d_model"], "tokens": tokens,
                        "t_single_ms": p.t_single * 1e3, "predicted_comm_ms": p.t_comm * 1e3,
                        "predicted_comm_fraction": p.comm_fraction,
                        "predicted_speedup": p.speedup, "measured_speedup": None,
                    }
                )
    return rows


# ---------------------------------------------------------------------- experiments
def _progress(message: str) -> None:
    print(f"[progress] {message}", file=sys.stderr, flush=True)


SPAWN_RETRY_LOG: list[str] = []  # one entry per retried worker death; copied into the run metadata


def core_split(read_sysctl: Callable[[str], int | None]) -> tuple[int | None, int | None]:
    """(performance cores, efficiency cores) on Apple silicon; (None, None) elsewhere."""
    return read_sysctl("hw.perflevel0.logicalcpu"), read_sysctl("hw.perflevel1.logicalcpu")


def _sysctl_int(name: str) -> int | None:
    try:
        out = subprocess.run(["sysctl", "-n", name], capture_output=True, text=True, check=True, timeout=5)
        return int(out.stdout.strip())
    except (OSError, subprocess.SubprocessError, ValueError):
        return None  # not macOS / key absent: the metadata field is simply null


def _spawn_retrying(fn, world: int, *args) -> list:
    """``spawn_group`` with a bounded retry for *process death* (signal exits) only.

    One full-size run once died inside Gloo ("Unexpected opcode", SIGABRT) and did not
    reproduce. Python exceptions from the workers are real bugs and are never retried.
    """
    for attempt in range(1, SPAWN_ATTEMPTS + 1):
        try:
            return spawn_group(fn, world, *args)
        except ProcessExitedException as exc:
            if attempt == SPAWN_ATTEMPTS:
                raise
            SPAWN_RETRY_LOG.append(f"{getattr(fn, '__name__', fn)} world={world}: {exc}")
            _progress(f"worker process died ({exc}); retrying (attempt {attempt + 1}/{SPAWN_ATTEMPTS})")
    raise AssertionError("unreachable")


def exp_a(cfg: ExperimentConfig) -> tuple[list[Row], list[Row]]:
    rows: list[Row] = []
    for world in cfg.worlds:
        runs = []
        for repeat in range(cfg.repeats):
            _progress(f"A: collectives, world={world}, repeat {repeat + 1}/{cfg.repeats}")
            per_rank = _spawn_retrying(
                bench_collectives_worker, world, cfg.message_bytes, cfg.collective_iters, cfg.warmup
            )
            runs.append(merge_ranks(per_rank, ("collective", "size_bytes"), ("median_s", "min_s")))
        merged = median_across_runs(runs, ("collective", "size_bytes"), ("median_s", "min_s"))
        rows.extend(collective_rows(world, merged))
    return rows, fit_rows(rows)


def exp_b(cfg: ExperimentConfig) -> list[Row]:
    cases = tuple((d, t) for d in cfg.mlp_d_models for t in cfg.mlp_tokens)
    time_fields = ("ref_s", "tp_total_s", "tp_compute_s", "tp_comm_s")
    rows: list[Row] = []
    for world in cfg.mlp_worlds:
        runs = []
        for repeat in range(cfg.repeats):
            _progress(f"B: TP MLP, world={world}, repeat {repeat + 1}/{cfg.repeats}")
            per_rank = _spawn_retrying(
                bench_tp_mlp_worker, world, cases, cfg.ff_mult, cfg.mlp_iters, cfg.warmup, cfg.seed
            )
            runs.append(merge_ranks(per_rank, ("d_model", "tokens"), (*time_fields, "max_err")))
        merged = median_across_runs(runs, ("d_model", "tokens"), time_fields, ("max_err",))
        rows.extend(mlp_rows(world, merged))
    return rows


def exp_c(
    cfg: ExperimentConfig, fits: list[Row] | None = None, measured: list[Row] | None = None
) -> list[Row]:
    """CPU model-vs-measurement rows plus assumed-GPU rows. Runs A / B if not supplied."""
    fits = exp_a(cfg)[1] if fits is None else fits
    measured = exp_b(cfg) if measured is None else measured
    return cpu_model_rows(fits, measured) + gpu_assumption_rows()


# ------------------------------------------------------------------------- output
def write_results(name: str, rows: list[Row], meta: dict, out_dir: Path) -> None:
    """Write {meta, rows} JSON plus a CSV of the columns."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{name}.json").write_text(json.dumps({"meta": meta, "rows": rows}, indent=2))
    if not rows:
        return
    columns = list(dict.fromkeys(k for r in rows for k in r))
    with (out_dir / f"{name}.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def run_metadata(cfg: ExperimentConfig, smoke: bool, **extra: object) -> dict:
    performance_cores, efficiency_cores = core_split(_sysctl_int)
    return {
        "smoke": smoke,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "backend": "gloo",
        "device": "cpu",
        "threads_per_rank": 1,
        "cpu_count": os.cpu_count(),
        "performance_cores": performance_cores,
        "efficiency_cores": efficiency_cores,
        "spawn_retries": list(SPAWN_RETRY_LOG),
        "note": (
            "Measured on CPU with Gloo over loopback, all ranks on one machine, no CPU affinity "
            "(macOS has none), so ranks may land on performance or efficiency cores. Not NCCL, "
            "not NVLink/PCIe. Collective timings are bare torch.distributed calls on preallocated "
            "buffers; each timed call starts right after a barrier, so latencies include "
            "barrier-exit skew between ranks, and per-case results take the slowest rank's "
            "median. Rows with source=assumption are model predictions from assumed hardware "
            "numbers, not measurements."
        ),
        "config": asdict(cfg),
        **extra,
    }


def _fmt(v: object) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:,.3f}" if abs(v) < 10 else f"{v:,.0f}"
    return str(v)


def print_table(title: str, rows: list[Row], columns: list[str]) -> None:
    print(f"\n== {title}")
    if not rows:
        print("(no rows)")
        return
    widths = [max(len(c), *(len(_fmt(r.get(c))) for r in rows)) for c in columns]
    print("  ".join(c.ljust(w) for c, w in zip(columns, widths)))
    for r in rows:
        print("  ".join(_fmt(r.get(c)).ljust(w) for c, w in zip(columns, widths)))


def run(names: tuple[str, ...], cfg: ExperimentConfig, out: Path, suffix: str, smoke: bool) -> None:
    SPAWN_RETRY_LOG.clear()
    fits: list[Row] | None = None
    mlp: list[Row] | None = None
    if "A" in names or "C" in names:
        rows_a, fits = exp_a(cfg)
        if "A" in names:
            write_results(f"exp_a{suffix}", rows_a, run_metadata(cfg, smoke), out)
            write_results(f"exp_a_fits{suffix}", fits, run_metadata(cfg, smoke), out)
            print_table("A: collectives", rows_a, ["world", "collective", "size_bytes", "latency_us", "latency_spread", "algbw_GBps", "busbw_GBps"])
            print_table("A: alpha/beta fits", fits, ["world", "collective", "alpha_us", "beta_ns_per_byte", "implied_bw_GBps", "max_rel_err"])
    if "B" in names or "C" in names:
        mlp = exp_b(cfg)
        if "B" in names:
            write_results(f"exp_b{suffix}", mlp, run_metadata(cfg, smoke), out)
            print_table("B: TP MLP vs one process", mlp, ["world", "d_model", "tokens", "ref_ms", "tp_total_ms", "tp_comm_ms", "speedup", "comm_fraction", "ref_spread", "tp_total_spread"])
    if "C" in names:
        rows_c = exp_c(cfg, fits, mlp)
        write_results(f"exp_c{suffix}", rows_c, run_metadata(cfg, smoke), out)
        print_table("C: comm model vs measurement (and assumed GPUs)", rows_c, ["source", "hardware", "world", "d_model", "tokens", "predicted_comm_ms", "measured_comm_ms", "comm_model_ratio", "predicted_speedup", "measured_speedup"])


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exp", default="all", choices=[*EXPERIMENTS, "all"])
    parser.add_argument("--smoke", action="store_true", help="tiny sizes; files get a _smoke suffix")
    parser.add_argument("--out", type=Path, default=RESULTS_DIR)
    parser.add_argument("--seed", type=int, default=0, help="seed for the benchmark tensors")
    args = parser.parse_args(argv)
    cfg = replace(SMOKE if args.smoke else FULL, seed=args.seed)
    names = EXPERIMENTS if args.exp == "all" else (args.exp,)
    run(names, cfg, args.out, "_smoke" if args.smoke else "", args.smoke)


if __name__ == "__main__":
    main()
