"""Render PNGs from results/*.json (and one calculator-only growth chart)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from memory_calculator import PRESETS, DeploymentSpec, memory_growth  # noqa: E402

RESULTS_DIR = Path(__file__).parent / "results"
COLORS = {"blue": "#2a78b5", "orange": "#c4552d", "grey": "#8a8a8a", "green": "#3a8f5c"}
BYTES_PER_GIB = 1024**3
BYTES_PER_MB = 1024**2
DPI = 150
GPU_80_GIB = 80


def _load(results_dir: Path, name: str) -> dict:
    return json.loads((results_dir / f"{name}.json").read_text())


def _save(fig: plt.Figure, out: Path) -> None:
    fig.tight_layout()
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"wrote {out}")


def plot_growth(out_dir: Path) -> None:
    """Theoretical total memory vs context for Llama-3-8B at several concurrencies."""
    contexts = [512 * 2**i for i in range(9)]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for seqs, color in zip((1, 4, 16, 64), (COLORS["blue"], COLORS["green"], COLORS["orange"], COLORS["grey"])):
        rows = memory_growth(DeploymentSpec(PRESETS["llama3-8b"], 1, seqs), contexts)
        ax.plot(contexts, [r["total_bytes"] / BYTES_PER_GIB for r in rows], marker="o", color=color,
                label=f"{seqs} seq")
    ax.axhline(GPU_80_GIB, color="black", linestyle="--", linewidth=1)
    ax.annotate("80 GiB GPU", (contexts[0], GPU_80_GIB), xytext=(4, 4), textcoords="offset points")
    ax.set(xscale="log", yscale="log", xlabel="context length (tokens)", ylabel="weights + KV (GiB)",
           title="Llama-3-8B (fp16): theoretical memory vs context")
    ax.legend(title="concurrency")
    _save(fig, out_dir / "memory_growth.png")


def plot_exp_a(results_dir: Path, out_dir: Path) -> None:
    rows = [r for r in _load(results_dir, "exp_a")["rows"] if not r["skipped"]]
    seqs = sorted({r["concurrent_seqs"] for r in rows})
    cmap = plt.get_cmap("viridis")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, s in enumerate(seqs):
        pts = sorted((r for r in rows if r["concurrent_seqs"] == s), key=lambda r: r["context_len"])
        color = cmap(i / max(len(seqs) - 1, 1))
        ctx = [p["context_len"] for p in pts]
        ax.plot(ctx, [p["theoretical_kv_mb"] for p in pts], color=color, linewidth=1.5, label=f"{s} seq (theory)")
        ax.scatter(ctx, [p["measured_kv_mb"] for p in pts], color=color, marker="x", s=40, zorder=3)
    ax.set(xscale="log", yscale="log", xlabel="context length (tokens)", ylabel="KV cache (MB)",
           title="Exp A: theoretical (lines) vs measured allocation (x)")
    ax.legend(fontsize=8)
    _save(fig, out_dir / "exp_a_memory_scaling.png")


def plot_exp_b(results_dir: Path, out_dir: Path) -> None:
    rows = _load(results_dir, "exp_b")["rows"]
    has_reserved = all(r["reserved_is_meaningful"] for r in rows)
    labels = [r["stage"] for r in rows]
    x = list(range(len(rows)))
    width = 0.27 if has_reserved else 0.4
    fig, ax = plt.subplots(figsize=(10, 4.8))
    alloc_label = "allocated (live tensors)" if has_reserved else "process RSS (no allocator counters on CPU)"
    ax.bar([i - width for i in x] if has_reserved else [i - width / 2 for i in x],
           [r["allocated_mb"] for r in rows], width, color=COLORS["blue"], label=alloc_label)
    if has_reserved:  # on CPU "reserved" is the same RSS number, so drawing it would be misleading
        ax.bar(x, [r["reserved_mb"] for r in rows], width, color=COLORS["orange"], label="reserved (driver / cache)")
    ax.bar([i + (width if has_reserved else width / 2) for i in x], [r["peak_allocated_mb"] for r in rows],
           width, color=COLORS["grey"], label="peak allocated")
    ax.set_xticks(x, labels, rotation=40, ha="right")
    ax.set(ylabel="MB", title="Exp B: allocated vs reserved vs peak")
    ax.set_ylim(0, max(r["reserved_mb"] if has_reserved else r["allocated_mb"] for r in rows) * 1.3)
    ax.legend(loc="upper left", ncol=3)
    _save(fig, out_dir / "exp_b_allocated_vs_reserved.png")


def plot_exp_c(results_dir: Path, out_dir: Path) -> None:
    rows = _load(results_dir, "exp_c")["rows"]
    names = [r["allocator"].replace("contiguous-max", "contig\n(max)").replace("paged-", "paged\n") for r in rows]
    colors = [COLORS["orange"] if r["block_size"] is None else COLORS["blue"] for r in rows]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    panels = (
        ("mean_utilization", "KV pool utilization (tokens stored / capacity)", 100, "%"),
        ("unused_reserved_fraction", "Unused share of reserved space", 100, "%"),
        ("mean_running", "Mean concurrent requests", 1, ""),
        ("admission_failures", "Admission-blocked steps", 1, ""),
        ("preemptions", "Preemptions (evicted mid-decode)", 1, ""),
        ("throughput_tokens_per_step", "Useful tokens per step", 1, ""),
    )
    for ax, (key, title, scale, unit) in zip(axes.flat, panels):
        ax.bar(names, [r[key] * scale for r in rows], color=colors)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(unit)
        ax.tick_params(axis="x", labelsize=8)
    _save(fig, out_dir / "exp_c_paged_vs_contiguous.png")


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()
    plot_growth(args.results)
    plot_exp_a(args.results, args.results)
    plot_exp_b(args.results, args.results)
    plot_exp_c(args.results, args.results)


if __name__ == "__main__":
    _main()
