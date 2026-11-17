"""Figures for the matrix, ablation and multi-GPU model.

    .venv/bin/python plot_matrix.py --dir results/sim        # writes <dir>/figures/*.png

Simulated results carry a SIMULATED watermark. Colours are the Okabe-Ito colour-blind-safe set;
bars print their values.
"""
from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import matrix as mx
import results_io

COLORS = {"baseline": "#0072B2", "tuned": "#D55E00", "util-only": "#009E73", "tokens-only": "#CC79A7",
          "cap16": "#999999", "cap24": "#777777", "cap32": "#555555"}
LINK_COLORS = {"nvlink": "#0072B2", "pcie": "#E69F00", "network-100gbe": "#D55E00"}
DPI = 140
FIGURES_DIR = "figures"
MS_PER_S = 1000.0


def _mean(summary: Mapping[str, Any] | None, metric: str) -> float:
    """Mean over repeats, or NaN if the case or metric is missing/unmeasured (never drawn)."""
    stats = None if summary is None else summary.get(metric)
    return math.nan if stats is None else stats["mean"]


def _save(fig: plt.Figure, path: Path, simulated: bool | None) -> Path:
    """`simulated`: True -> SIMULATED watermark; None (metadata silent) -> PROVENANCE UNKNOWN; False -> none."""
    label = {True: "SIMULATED", None: "PROVENANCE UNKNOWN"}.get(simulated)
    if label:
        fig.text(0.99, 0.01, label, ha="right", va="bottom", color="#999999", fontsize=11, fontweight="bold")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def _cells(summaries: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [s for s in summaries if not s["is_control"]]


def plot_tradeoff(summaries, decisions, path: Path, simulated: bool) -> Path:
    """Throughput vs P95 latency per workload; the dashed line is the latency limit L."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    limits = {d["workload"]: d["threshold_ms"] for d in decisions}
    for ax, key in zip(axes, mx.WORKLOADS):
        for cfg in mx.CONFIGS:
            pts = sorted((s for s in _cells(summaries) if s["workload"] == key and s["config"] == cfg),
                         key=lambda s: s["concurrency"])
            pts = [s for s in pts if not math.isnan(_mean(s, "p95_e2e_ms") + _mean(s, "output_tok_s"))]
            xs = [_mean(s, "p95_e2e_ms") / MS_PER_S for s in pts]
            ys = [_mean(s, "output_tok_s") for s in pts]
            ax.plot(xs, ys, "o-", color=COLORS[cfg], label=cfg)
            for s, x, y in zip(pts, xs, ys):
                ax.annotate(f"c={s['concurrency']}", (x, y), textcoords="offset points",
                            xytext=(5, -11 if cfg == "tuned" else 5), fontsize=8, color=COLORS[cfg])
        if key in limits:
            ax.axvline(limits[key] / MS_PER_S, color="#444444", ls="--", lw=1)
            ax.text(limits[key] / MS_PER_S, ax.get_ylim()[0], " L", va="bottom", fontsize=9)
        ax.set_title(f"Workload {key}: {mx.WORKLOADS[key].label}")
        ax.set_xlabel("P95 end-to-end latency (s)")
        ax.set_ylabel("output tokens / s")
        ax.grid(alpha=0.3)
        ax.legend()
    return _save(fig, path, simulated)


def _bar_labels(ax, bars, digits: int = 0) -> None:
    for bar in bars:
        if math.isnan(bar.get_height()):
            continue
        ax.annotate(f"{bar.get_height():,.{digits}f}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    ha="center", va="bottom", fontsize=7, xytext=(0, 1), textcoords="offset points")


def plot_latency(summaries, path: Path, simulated: bool) -> Path:
    """Mean TTFT and TPOT per case, baseline vs tuned."""
    fig, axes = plt.subplots(2, 1, figsize=(11, 7))
    groups = [(w, c) for w in mx.WORKLOADS for c in mx.CONCURRENCY_LEVELS]
    for ax, metric, title, digits in ((axes[0], "mean_ttft_ms", "Mean TTFT (ms)", 0),
                                      (axes[1], "mean_tpot_ms", "Mean TPOT (ms)", 1)):
        width = 0.38
        for i, cfg in enumerate(mx.CONFIGS):
            by_id = {s["case_id"]: s for s in _cells(summaries)}
            vals = [_mean(by_id.get(f"{w}-c{c}-{cfg}"), metric) for w, c in groups]
            bars = ax.bar([x + (i - 0.5) * width for x in range(len(groups))], vals, width,
                          color=COLORS[cfg], label=cfg)
            _bar_labels(ax, bars, digits)
        ax.set_xticks(range(len(groups)), [f"{w} c={c}" for w, c in groups])
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
        ax.legend()
    return _save(fig, path, simulated)


def plot_kv(summaries, path: Path, simulated: bool) -> Path:
    """Peak KV-pool usage and preemptions: the capacity-pressure indicators."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    cells = sorted(_cells(summaries), key=lambda s: (s["workload"], s["concurrency"], s["config"]))
    names = [s["case_id"] for s in cells]
    colors = [COLORS[s["config"]] for s in cells]
    for ax, metric, title, digits, scale in ((axes[0], "kv_peak_frac", "Peak KV-pool usage (%)", 0, 100),
                                             (axes[1], "preemptions", "Preemptions per run", 0, 1)):
        vals = [_mean(s, metric) * scale for s in cells]
        _bar_labels(ax, ax.bar(range(len(cells)), vals, color=colors), digits)
        ax.set_xticks(range(len(cells)), names, rotation=60, ha="right", fontsize=7)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
    return _save(fig, path, simulated)


def plot_ablation(ablation, path: Path, simulated: bool, workload: str = "B",
                  concurrency: int = mx.KEY_CONCURRENCY) -> Path:
    """Each variant's change vs baseline (%), at the KV-pressured cell."""
    cells = [s for s in ablation if s["workload"] == workload and s["concurrency"] == concurrency]
    base = next((s for s in cells if s["config"] == "baseline"), None)
    if base is None:
        raise ValueError(f"ablation has no baseline cell for workload {workload} at concurrency {concurrency}")
    metrics = (("output_tok_s", "output tok/s"), ("mean_ttft_ms", "mean TTFT"),
               ("mean_tpot_ms", "mean TPOT"), ("p95_e2e_ms", "P95 e2e"))
    others = [s for s in cells if s["config"] != "baseline"]
    fig, ax = plt.subplots(figsize=(11, 4.6))
    width = 0.8 / len(others)
    for i, s in enumerate(others):
        deltas = [(_mean(s, m) / _mean(base, m) - 1) * 100 if _mean(base, m) else math.nan for m, _ in metrics]
        bars = ax.bar([x + (i - len(others) / 2 + 0.5) * width for x in range(len(metrics))], deltas, width,
                      color=COLORS.get(s["config"], "#999999"), label=s["config"])
        _bar_labels(ax, bars, 0)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(range(len(metrics)), [label for _, label in metrics])
    ax.set_ylabel(f"% change vs baseline ({workload} c={concurrency})")
    ax.set_title("Ablation: which knob did what")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(ncol=3, fontsize=8)
    return _save(fig, path, simulated)


def plot_multi_gpu(rows, path: Path, batch: int = 64) -> Path:
    """ASSUMED link constants: decode speedup and group KV capacity vs tensor-parallel degree."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for link, color in LINK_COLORS.items():
        pts = sorted((r for r in rows if r["link"] == link and r["batch_tokens"] == batch),
                     key=lambda r: r["tp"])
        axes[0].plot([r["tp"] for r in pts], [r["speedup"] for r in pts], "o-", color=color, label=link)
    axes[0].axhline(1.0, color="black", lw=0.8, ls=":")
    axes[0].set(title=f"Decode step speedup (batch {batch})", xlabel="TP degree", ylabel="speedup vs 1 GPU")
    pts = sorted((r for r in rows if r["link"] == "nvlink" and r["batch_tokens"] == batch), key=lambda r: r["tp"])
    axes[1].plot([r["tp"] for r in pts], [r["gain_vs_tp1"] for r in pts], "o-", color="#009E73", label="KV capacity")
    axes[1].plot([r["tp"] for r in pts], [r["tp"] for r in pts], "k:", label="linear (n x)")
    axes[1].set(title="Group KV capacity vs 1 GPU", xlabel="TP degree", ylabel="x capacity")
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xticks([r["tp"] for r in pts], [str(r["tp"]) for r in pts])
        ax.grid(alpha=0.3)
        ax.legend()
    fig.text(0.01, 0.01, "link constants ASSUMED, not measured", fontsize=8, color="#777777")
    return _save(fig, path, simulated=False)


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", type=Path, required=True)
    args = p.parse_args(argv)
    if not (args.dir / "analysis.json").is_file():
        print(f"{args.dir / 'analysis.json'} not found; run analyze.py first", file=sys.stderr)
        return 2
    analysis = results_io.read_results(args.dir / "analysis.json")
    simulated = analysis["meta"].get("simulated")  # None when the metadata is silent
    summary_path = args.dir / "matrix_summary.json"
    try:
        results_io.assert_fresh(summary_path, analysis["meta"].get("matrix_summary_sha256"))
    except ValueError as exc:
        print(f"cannot plot: {exc}", file=sys.stderr)
        return 2
    summaries = results_io.read_results(summary_path)["rows"]
    out = args.dir / FIGURES_DIR
    plot_tradeoff(summaries, analysis["rows"][0]["decisions"], out / "tradeoff.png", simulated)
    plot_latency(summaries, out / "latency.png", simulated)
    plot_kv(summaries, out / "kv_pressure.png", simulated)
    ablation_path = args.dir / "ablation_summary.json"
    if ablation_path.is_file():
        plot_ablation(results_io.read_results(ablation_path)["rows"], out / "ablation.png", simulated)
    plot_multi_gpu(results_io.read_results(args.dir / "multi_gpu.json")["rows"], out / "multi_gpu.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
