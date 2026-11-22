"""`02-kv-cache-memory` experiments.

A  Memory scaling   : theoretical KV bytes vs. bytes actually allocated as PyTorch tensors.
B  Memory profiling : allocated vs. reserved vs. peak through a scripted alloc/free sequence.
C  Paged allocation : block allocator vs. reserve-max-upfront on a simulated workload.

A and B need a device (CUDA, MPS or CPU); C is pure simulation. Results go to
results/exp_{a,b,c}[_smoke].{json,csv}.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import platform
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from kv_simulation import WorkloadConfig, generate_requests, simulate
from memory_calculator import (
    PRESETS,
    DeploymentSpec,
    kv_bytes_per_token,
    kv_cache_bytes,
)
from memory_probe import MemoryProbe, pick_device
from paged_allocator import ContiguousAllocator, PagedKVAllocator

RESULTS_DIR = Path(__file__).parent / "results"
BYTES_PER_MB = 1024 * 1024
EXP_A_MODEL = "qwen2.5-0.5b"  # same model as day 1: 12,288 B/token in fp16
TORCH_DTYPES = {"fp16": torch.float16, "fp32": torch.float32}

Row = dict[str, object]


@dataclass(frozen=True)
class ExperimentConfig:
    a_contexts: tuple[int, ...]
    a_seqs: tuple[int, ...]
    a_max_kv_bytes: int  # skip sweep points whose theoretical KV exceeds this
    b_tensor_mb: int
    b_small_pieces: int
    c_pool_tokens: int
    c_block_sizes: tuple[int, ...]
    c_workload: WorkloadConfig


FULL = ExperimentConfig(
    a_contexts=(256, 512, 1024, 2048, 4096, 8192),
    a_seqs=(1, 2, 4, 8, 16, 32),
    a_max_kv_bytes=3 * 1024**3,
    b_tensor_mb=256,
    b_small_pieces=16,
    c_pool_tokens=16_384,
    c_block_sizes=(8, 16, 32, 64, 128),
    c_workload=WorkloadConfig(
        num_requests=600, max_seq_len=2048, arrivals_per_step=0.5,
        prompt_median=128, prompt_sigma=1.0, output_median=128, output_sigma=0.8, seed=0,
    ),
)

SMOKE = ExperimentConfig(
    a_contexts=(16, 32),
    a_seqs=(1, 2),
    a_max_kv_bytes=64 * BYTES_PER_MB,
    b_tensor_mb=4,
    b_small_pieces=4,
    c_pool_tokens=1024,
    c_block_sizes=(8, 32),
    c_workload=WorkloadConfig(
        num_requests=40, max_seq_len=128, arrivals_per_step=0.5,
        prompt_median=16, prompt_sigma=0.8, output_median=16, output_sigma=0.6, seed=0,
    ),
)


def _mb(n: float) -> float:
    return round(n / BYTES_PER_MB, 3)


def _make_buffer(nbytes: int, device: str) -> torch.Tensor:
    """Allocate and touch `nbytes` (ones, not empty, so CPU pages are really committed)."""
    return torch.ones(nbytes, dtype=torch.uint8, device=device)


# ------------------------------------------------------------------ Experiment A
def run_exp_a(device: str, cfg: ExperimentConfig) -> list[Row]:
    """Sweep context x concurrency; compare theory with the live-tensor delta."""
    model = PRESETS[EXP_A_MODEL]
    probe = MemoryProbe(device)
    rows: list[Row] = []
    for ctx in cfg.a_contexts:
        for seqs in cfg.a_seqs:
            spec = DeploymentSpec(model, ctx, seqs, "fp16")
            theory = kv_cache_bytes(spec)
            row: Row = {
                "context_len": ctx,
                "concurrent_seqs": seqs,
                "theoretical_kv_mb": _mb(theory),
                "skipped": theory > cfg.a_max_kv_bytes,
            }
            if not row["skipped"]:
                row.update(_measure_kv_allocation(probe, spec, theory))
            rows.append(row)
    return rows


def _measure_kv_allocation(probe: MemoryProbe, spec: DeploymentSpec, theory: int) -> Row:
    """Allocate one K and one V tensor per layer, as a real KV cache does."""
    m = spec.model
    probe.empty_cache()
    probe.reset_peak()
    before = probe.snapshot()
    shape = (spec.concurrent_seqs, m.num_kv_heads, spec.context_len, m.head_dim)
    cache = [
        torch.ones(shape, dtype=TORCH_DTYPES[spec.kv_dtype], device=probe.device)
        for _ in range(2 * m.num_layers)
    ]
    after = probe.snapshot()
    measured = after.allocated - before.allocated
    row: Row = {
        "num_tensors": len(cache),
        "measured_kv_mb": _mb(measured),
        "measured_reserved_delta_mb": _mb(after.reserved - before.reserved),
        "peak_over_baseline_mb": _mb(after.peak_allocated - before.allocated),
        "measured_over_theory": round(measured / theory, 4),
        "error_pct": round(100 * (measured - theory) / theory, 3),
    }
    del cache
    probe.empty_cache()
    return row


# ------------------------------------------------------------------ Experiment B
def run_exp_b(device: str, cfg: ExperimentConfig) -> list[Row]:
    """Scripted alloc/free sequence showing allocated vs reserved vs peak."""
    probe = MemoryProbe(device)
    size = cfg.b_tensor_mb * BYTES_PER_MB
    rows: list[Row] = []

    def record(stage: str, note: str) -> None:
        snap = probe.snapshot()
        rows.append({
            "stage": stage,
            "allocated_mb": _mb(snap.allocated),
            "reserved_mb": _mb(snap.reserved),
            "peak_allocated_mb": _mb(snap.peak_allocated),
            "reserved_minus_allocated_mb": _mb(snap.reserved - snap.allocated),
            "reserved_is_meaningful": probe.has_reserved_counter,
            "note": note,
        })

    probe.empty_cache()
    probe.reset_peak()
    record("0 baseline", "after empty_cache + reset_peak")
    a = _make_buffer(size, device)
    record("1 alloc A", f"live tensor of {cfg.b_tensor_mb} MB")
    b = _make_buffer(size, device)
    record("2 alloc B", "second live tensor; peak follows allocated")
    del a
    gc.collect()
    record("3 free A", "allocated drops; reserved should stay (block is cached for reuse)")
    c = _make_buffer(size // 2, device)
    record("4 alloc C (half size)", "served from the cached block if the allocator reuses it")
    del b, c
    gc.collect()
    record("5 free all", "no live tensors, memory still reserved; peak remembers the high-water mark")
    probe.empty_cache()
    record("6 empty_cache", "cached blocks returned to the driver; reserved falls")

    probe.reset_peak()
    pieces = [_make_buffer(size // cfg.b_small_pieces, device) for _ in range(cfg.b_small_pieces)]
    record("7 many small tensors", f"{cfg.b_small_pieces} tensors of 1/{cfg.b_small_pieces} size")
    kept = [p for i, p in enumerate(pieces) if i % 2 == 0]
    del pieces
    gc.collect()
    record("8 free every other", "half the bytes are free but scattered between live tensors")
    big = _make_buffer(size // 2 + 1, device)
    record("9 alloc one large", "needs contiguous space; cannot reuse the scattered holes")
    del kept, big
    probe.empty_cache()
    record("10 cleanup", "all freed and cache emptied")
    return rows


# ------------------------------------------------------------------ Experiment C
def run_exp_c(cfg: ExperimentConfig, seed: int | None = None) -> list[Row]:
    """Paged (several block sizes) vs reserve-max-upfront on one shared workload."""
    wl = cfg.c_workload if seed is None else WorkloadConfig(**{**asdict(cfg.c_workload), "seed": seed})
    requests = generate_requests(wl)
    pool = cfg.c_pool_tokens
    results = [simulate("contiguous-max", ContiguousAllocator(pool, wl.max_seq_len), requests)]
    for bs in cfg.c_block_sizes:
        results.append(simulate(f"paged-bs{bs}", PagedKVAllocator(pool // bs, bs), requests, block_size=bs))
    return [{**r.to_row(), "pool_tokens": pool} for r in results]


# ------------------------------------------------------------------ output
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


def run_metadata(device: str, cfg: ExperimentConfig, smoke: bool) -> dict:
    kv_spec = DeploymentSpec(PRESETS[EXP_A_MODEL], 1, 1, "fp16")
    return {
        "device": device,
        "torch": torch.__version__,
        "platform": platform.platform(),
        "smoke": smoke,
        "model": EXP_A_MODEL,
        "kv_bytes_per_token": kv_bytes_per_token(kv_spec),
        "config": asdict(cfg),
    }


def _print_table(title: str, rows: list[Row], columns: list[str]) -> None:
    print(f"\n== {title} ==")
    print(" | ".join(columns))
    for r in rows:
        print(" | ".join(str(r.get(c, "")) for c in columns))


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exp", choices=["A", "B", "C", "all"], default="all")
    parser.add_argument("--smoke", action="store_true", help="tiny sizes; files get a _smoke suffix")
    parser.add_argument("--device", default=None, help="cuda | mps | cpu (default: best available)")
    parser.add_argument("--out", type=Path, default=RESULTS_DIR)
    parser.add_argument("--seed", type=int, default=None, help="override the Experiment C workload seed")
    args = parser.parse_args()

    cfg = SMOKE if args.smoke else FULL
    device = pick_device(args.device)
    suffix = "_smoke" if args.smoke else ""
    meta = run_metadata(device, cfg, args.smoke)
    print(f"device={device} torch={torch.__version__} smoke={args.smoke}")

    if args.exp in ("A", "all"):
        rows = run_exp_a(device, cfg)
        write_results(f"exp_a{suffix}", rows, meta, args.out)
        _print_table("A: memory scaling", rows,
                     ["context_len", "concurrent_seqs", "theoretical_kv_mb", "measured_kv_mb",
                      "measured_over_theory", "skipped"])
    if args.exp in ("B", "all"):
        rows = run_exp_b(device, cfg)
        write_results(f"exp_b{suffix}", rows, meta, args.out)
        _print_table("B: allocated vs reserved", rows,
                     ["stage", "allocated_mb", "reserved_mb", "peak_allocated_mb"])
    if args.exp in ("C", "all"):
        rows = run_exp_c(cfg, args.seed)
        write_results(f"exp_c{suffix}", rows, meta, args.out)
        _print_table("C: paged vs contiguous", rows,
                     ["allocator", "completed", "mean_running", "mean_utilization",
                      "unused_reserved_fraction", "admission_failures", "preemptions",
                      "throughput_tokens_per_step"])


if __name__ == "__main__":
    _main()
