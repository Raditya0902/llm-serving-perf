"""Plots for Day 5, read from results/*.json (run run_experiments.py first).

    python plot_results.py            # full results -> results/*.png
    python plot_results.py --suffix _smoke --results DIR

Figures: tp_diagram.png (schematic, no data), exp_a_collectives.png, exp_b_speedup.png,
exp_b_comm_fraction.png, exp_c_model.png. Same palette as Day 3. Every measured figure is
Gloo-on-CPU; the GPU panel in exp_c_model.png is built from *assumed* hardware numbers and
says so in its title.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Rectangle  # noqa: E402
from matplotlib.ticker import FuncFormatter, NullFormatter  # noqa: E402

RESULTS_DIR = Path(__file__).parent / "results"
DPI = 150
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
COLLECTIVE_COLORS = {"all_reduce": BLUE, "all_gather": ORANGE, "broadcast": AQUA}
WORLD_STYLES = {2: "-", 4: "--"}
Rows = list[dict]


def _load(results: Path, name: str, suffix: str) -> Rows:
    return json.loads((results / f"{name}{suffix}.json").read_text())["rows"]


def _style(ax: Axes, title: str, xlabel: str = "", ylabel: str = "") -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    ax.set_xlabel(xlabel, fontsize=9, color=INK_2)
    ax.set_ylabel(ylabel, fontsize=9, color=INK_2)
    ax.grid(True, color=GRID, linewidth=0.7)
    ax.tick_params(colors=INK_2, labelsize=8)
    for spine in ax.spines.values():
        spine.set_visible(False)


def _plain_log_ticks(ax: Axes, which: str, ticks: list[float] | None = None) -> None:
    """Readable log axis: plain numbers on major ticks, no minor-tick labels."""
    axis = ax.xaxis if which == "x" else ax.yaxis
    axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    axis.set_minor_formatter(NullFormatter())
    if ticks is not None:
        (ax.set_xticks if which == "x" else ax.set_yticks)(ticks)


def _human_bytes(value: float, _pos: int | None = None) -> str:
    for unit, scale in (("GB", 1 << 30), ("MB", 1 << 20), ("KB", 1 << 10)):
        if value >= scale:
            return f"{value / scale:g}{unit}"
    return f"{value:g}B"


def _save(fig: plt.Figure, path: Path) -> Path:
    fig.patch.set_facecolor(SURFACE)
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return path


# ------------------------------------------------------------------ Experiment A
def plot_collectives(rows: Rows, out: Path) -> Path:
    fig, (ax_lat, ax_bw) = plt.subplots(1, 2, figsize=(11, 4))
    for world in sorted({r["world"] for r in rows}):
        for name, color in COLLECTIVE_COLORS.items():
            pts = sorted((r for r in rows if r["world"] == world and r["collective"] == name),
                         key=lambda r: r["size_bytes"])
            sizes = [r["size_bytes"] for r in pts]
            style = WORLD_STYLES.get(world, ":")
            ax_lat.plot(sizes, [r["latency_us"] for r in pts], style, color=color, marker="o", ms=3)
            ax_bw.plot(sizes, [r["busbw_GBps"] for r in pts], style, color=color, marker="o", ms=3)
    for ax, title, ylabel in (
        (ax_lat, "Latency vs per-rank message size (Gloo, CPU)", "latency (µs, median)"),
        (ax_bw, "Bus bandwidth vs message size", "bus bandwidth (GB/s)"),
    ):
        _style(ax, title, "per-rank message size", ylabel)
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_formatter(FuncFormatter(_human_bytes))
    ax_lat.set_yscale("log")
    handles = [Line2D([], [], color=c, label=n) for n, c in COLLECTIVE_COLORS.items()]
    handles += [Line2D([], [], color=INK_2, linestyle=s, label=f"{w} ranks") for w, s in WORLD_STYLES.items()]
    ax_lat.legend(handles=handles, fontsize=8, frameon=False)
    return _save(fig, out / "exp_a_collectives.png")


# ------------------------------------------------------------------ Experiment B
def plot_speedup(rows: Rows, out: Path) -> Path:
    worlds = sorted({r["world"] for r in rows})
    d_models = sorted({r["d_model"] for r in rows})
    palette = [BLUE, ORANGE, AQUA, YELLOW]
    fig, axes = plt.subplots(1, len(worlds), figsize=(5.5 * len(worlds), 4), squeeze=False)
    for ax, world in zip(axes[0], worlds):
        for color, d in zip(palette, d_models):
            pts = sorted((r for r in rows if r["world"] == world and r["d_model"] == d),
                         key=lambda r: r["tokens"])
            ax.plot([r["tokens"] for r in pts], [r["speedup"] for r in pts], color=color, marker="o",
                    ms=3, label=f"d_model={d}")
        ax.axhline(1.0, color=INK, linewidth=1)
        ax.axhline(world, color=INK_2, linewidth=0.8, linestyle=":")
        _style(ax, f"TP={world} MLP speed-up vs one process (CPU/Gloo)",
               f"tokens per forward   (solid black = break-even, dotted = ideal {world}x)", "speed-up (x)")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_ylim(0.02, world * 1.6)
        _plain_log_ticks(ax, "y", [0.03, 0.1, 0.3, 1, 3] if world == 2 else [0.03, 0.1, 0.3, 1, 3])
        ax.legend(fontsize=8, frameon=False, loc="lower right")
    return _save(fig, out / "exp_b_speedup.png")


def plot_comm_fraction(rows: Rows, out: Path) -> Path:
    worlds = sorted({r["world"] for r in rows})
    d_models = sorted({r["d_model"] for r in rows})
    tokens = sorted({r["tokens"] for r in rows})
    fig, axes = plt.subplots(1, len(worlds), figsize=(5 * len(worlds), 3.6), squeeze=False)
    for ax, world in zip(axes[0], worlds):
        grid = [[next(r["comm_fraction"] for r in rows
                      if r["world"] == world and r["d_model"] == d and r["tokens"] == t) for t in tokens]
                for d in d_models]
        image = ax.imshow(grid, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(tokens)), [str(t) for t in tokens])
        ax.set_yticks(range(len(d_models)), [str(d) for d in d_models])
        for i, row in enumerate(grid):
            for j, v in enumerate(row):
                ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=8,
                        color="white" if v > 0.55 else INK)
        _style(ax, f"Share of TP time spent in AllReduce, TP={world}", "tokens per forward", "d_model")
        ax.grid(False)
    fig.colorbar(image, ax=axes[0].tolist(), shrink=0.8, label="comm / total")
    return _save(fig, out / "exp_b_comm_fraction.png")


# ------------------------------------------------------------------ Experiment C
def plot_model(rows: Rows, out: Path) -> Path:
    fig, (ax_cpu, ax_gpu) = plt.subplots(1, 2, figsize=(11, 4))
    cpu = [r for r in rows if r["source"] == "measured_cpu_model"]
    for world, color in ((2, BLUE), (4, ORANGE)):
        pts = [r for r in cpu if r["world"] == world]
        ax_cpu.scatter([r["measured_comm_ms"] for r in pts], [r["predicted_comm_ms"] for r in pts],
                       color=color, s=18, label=f"{world} ranks")
    if cpu:
        lo = min(min(r["measured_comm_ms"], r["predicted_comm_ms"]) for r in cpu)
        hi = max(max(r["measured_comm_ms"], r["predicted_comm_ms"]) for r in cpu)
        ax_cpu.plot([lo, hi], [lo, hi], color=INK, linewidth=1)
    _style(ax_cpu, "AllReduce time in Exp B: alpha/beta model (A) vs measured",
           "measured (ms)", "predicted (ms)")
    ax_cpu.set_xscale("log")
    ax_cpu.set_yscale("log")
    _plain_log_ticks(ax_cpu, "x", [0.15, 0.3, 0.6, 1.2, 2.4, 4.8])
    _plain_log_ticks(ax_cpu, "y", [0.15, 0.3, 0.6, 1.2, 2.4, 4.8])
    ax_cpu.text(0.98, 0.04, "on the line = model right", transform=ax_cpu.transAxes, ha="right", fontsize=8, color=INK_2)
    ax_cpu.legend(fontsize=8, frameon=False, loc="upper left")

    gpu = [r for r in rows if r["source"] == "assumption"]
    for (hardware, world), color, style in (
        (("nvlink_class", 2), BLUE, "-"), (("nvlink_class", 4), BLUE, "--"),
        (("pcie_class", 2), ORANGE, "-"), (("pcie_class", 4), ORANGE, "--"),
    ):
        pts = sorted((r for r in gpu if r["hardware"] == hardware and r["world"] == world),
                     key=lambda r: r["tokens"])
        ax_gpu.plot([r["tokens"] for r in pts], [r["predicted_speedup"] for r in pts], style,
                    color=color, marker="o", ms=3, label=f"{hardware.split('_')[0]}, TP={world}")
    ax_gpu.axhline(1.0, color=INK, linewidth=1)
    _style(ax_gpu, "ASSUMED hardware (hypothesis, not measured): TP speed-up, 1.5B MLP",
           "tokens per forward", "predicted speed-up (x)")
    ax_gpu.set_xscale("log", base=2)
    ax_gpu.legend(fontsize=8, frameon=False)
    return _save(fig, out / "exp_c_model.png")


# --------------------------------------------------------------- schematic diagram
def _box(ax: Axes, x: float, y: float, w: float, h: float, label: str, color: str, alpha: float = 0.28) -> None:
    ax.add_patch(Rectangle((x, y), w, h, facecolor=color, alpha=alpha, edgecolor=color, linewidth=1.2))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=8, color=INK)


def _arrow(ax: Axes, x0: float, y0: float, x1: float, y1: float, color: str = INK_2) -> None:
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=9, color=color, lw=1.1))


def _panel(ax: Axes, title: str) -> None:
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=10, color=INK, loc="left")


def plot_diagram(out: Path) -> Path:
    """Schematic of Y = X W for two devices: column parallel, row parallel, and the fused MLP."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    dev = (BLUE, ORANGE)

    ax = axes[0]
    _panel(ax, "Column parallel: split W's columns")
    _box(ax, 0.2, 5.0, 2.2, 1.6, "X (full)\non both", AQUA)
    for i, y in enumerate((4.2, 1.4)):
        _box(ax, 3.2, y, 2.2, 1.8, f"W{i}\n[d_in, d_out/2]", dev[i])
        _arrow(ax, 2.4, 5.8, 3.2, y + 0.9)
        _box(ax, 6.0, y + 0.3, 1.0, 1.2, f"Y{i}", dev[i])
        _arrow(ax, 5.4, y + 0.9, 6.0, y + 0.9)
        _arrow(ax, 7.0, y + 0.9, 7.6, 3.6)
    _box(ax, 7.6, 2.8, 2.2, 1.6, "Y = [Y0 | Y1]", YELLOW)
    ax.text(5.0, 0.4, "AllGather (concat)\nonly if next op needs full Y", ha="center", fontsize=8, color=INK_2)

    ax = axes[1]
    _panel(ax, "Row parallel: split W's rows (and X's columns)")
    for i, y in enumerate((4.6, 1.8)):
        _box(ax, 0.2, y + 0.2, 1.3, 1.2, f"X{i}", dev[i])
        _box(ax, 2.0, y, 2.4, 1.6, f"W{i}\n[d_in/2, d_out]", dev[i])
        _arrow(ax, 1.5, y + 0.8, 2.0, y + 0.8)
        _box(ax, 5.1, y + 0.2, 1.4, 1.2, f"P{i}\n(partial)", dev[i])
        _arrow(ax, 4.4, y + 0.8, 5.1, y + 0.8)
        _arrow(ax, 6.5, y + 0.8, 7.2, 3.6)
    _box(ax, 7.2, 2.8, 2.6, 1.6, "Y = P0 + P1", YELLOW)
    ax.text(5.0, 0.5, "AllReduce (sum): partials are full-size,\nnot slices of Y", ha="center", fontsize=8, color=INK_2)

    ax = axes[2]
    _panel(ax, "MLP: column -> activation -> row = 1 AllReduce")
    _box(ax, 0.1, 3.4, 1.0, 1.2, "X", AQUA)
    for i, y in enumerate((5.0, 1.8)):
        _box(ax, 1.7, y, 1.4, 1.2, f"W1[:,{i}]\ncolumn", dev[i])
        _box(ax, 3.6, y, 1.2, 1.2, f"gelu\nh{i}", dev[i])
        _box(ax, 5.3, y, 1.4, 1.2, f"W2[{i},:]\nrow", dev[i])
        _arrow(ax, 1.1, 4.0, 1.7, y + 0.6)
        _arrow(ax, 3.1, y + 0.6, 3.6, y + 0.6)
        _arrow(ax, 4.8, y + 0.6, 5.3, y + 0.6)
        _arrow(ax, 6.7, y + 0.6, 7.3, 3.9)
    _box(ax, 7.3, 3.2, 2.5, 1.4, "AllReduce\n-> Y", YELLOW)
    ax.text(3.9, 0.6, "no communication between the two matmuls:\nactivation is elementwise, so it commutes with sharding",
            ha="center", fontsize=8, color=INK_2)

    fig.legend(handles=[Rectangle((0, 0), 1, 1, color=dev[0], alpha=0.5, label="device 0"),
                        Rectangle((0, 0), 1, 1, color=dev[1], alpha=0.5, label="device 1"),
                        Rectangle((0, 0), 1, 1, color=YELLOW, alpha=0.5, label="after collective")],
               loc="lower center", ncol=3, frameon=False, fontsize=9)
    return _save(fig, out / "tp_diagram.png")


def make_all(results: Path, suffix: str) -> list[Path]:
    paths = [plot_diagram(results)]
    a, b, c = (_load(results, name, suffix) for name in ("exp_a", "exp_b", "exp_c"))
    paths += [plot_collectives(a, results), plot_speedup(b, results),
              plot_comm_fraction(b, results), plot_model(c, results)]
    return paths


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, default=RESULTS_DIR)
    parser.add_argument("--suffix", default="", help="e.g. _smoke to plot smoke-run files")
    args = parser.parse_args(argv)
    for path in make_all(args.results, args.suffix):
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
