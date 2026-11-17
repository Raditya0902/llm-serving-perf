"""Plots for the day-3 experiments, read from results/*.json (run_experiments.py first).

Palette: reference categorical slots 1-4 (blue, orange, aqua, yellow), validated with the
dataviz validator. Slot 3 and 4 sit below 3:1 contrast on the light surface, so every bar
carries a printed value and the README repeats the numbers as tables. One y-scale per axis:
measures with different units get their own panel.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import FuncFormatter, NullFormatter  # noqa: E402

from simulator import Simulator  # noqa: E402
from timeline import demo_scenario, gantt_segments  # noqa: E402

RESULTS_DIR = Path(__file__).parent / "results"
DPI = 150
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
STRATEGY_COLORS = {"sequential": BLUE, "static": ORANGE, "continuous": AQUA, "continuous_chunked": YELLOW}
STRATEGY_LABELS = {"sequential": "sequential", "static": "static", "continuous": "continuous",
                   "continuous_chunked": "continuous +\nchunked prefill"}
Rows = list[dict]


def _load(results: Path, name: str) -> Rows:
    return json.loads((results / f"{name}.json").read_text())["rows"]


def _style(ax: Axes, title: str, ylabel: str = "", xlabel: str = "") -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    ax.set_ylabel(ylabel, fontsize=8, color=INK_2)
    ax.set_xlabel(xlabel, fontsize=8, color=INK_2)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=8, length=0)


def _log_y(ax: Axes) -> None:
    """Log y-axis with plain numeric tick labels (no 10^n mathtext)."""
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.yaxis.set_minor_formatter(NullFormatter())


def _figure(ncols: int, width: float, height: float = 3.6):
    fig, axes = plt.subplots(1, ncols, figsize=(width, height), facecolor=SURFACE, squeeze=False)
    return fig, list(axes[0])


def _save(fig, out: Path, name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out / name, dpi=DPI, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {out / name}")


def _bar_labels(ax: Axes, bars, fmt: str = "{:,.0f}") -> None:
    for b in bars:
        ax.annotate(fmt.format(b.get_height()), (b.get_x() + b.get_width() / 2, b.get_height()),
                    ha="center", va="bottom", fontsize=7, color=INK, xytext=(0, 2), textcoords="offset points")


# --------------------------------------------------------------------- plots
def plot_exp_a(results: Path, out: Path) -> None:
    rows = {r["strategy"]: r for r in _load(results, "exp_a_summary")}
    names = [n for n in STRATEGY_COLORS if n in rows]
    panels = [("throughput_tok_s", "Throughput (output tok / simulated s)", False, "{:,.0f}"),
              ("mean_latency_ms", "Mean latency (ms, log scale)", True, "{:,.0f}"),
              ("p95_latency_ms", "P95 latency (ms, log scale)", True, "{:,.0f}"),
              ("util_proxy", "Utilization PROXY (useful compute share)", False, "{:.2f}")]
    fig, axes = _figure(4, 15)
    for ax, (col, title, log, fmt) in zip(axes, panels):
        vals = [rows[n][col] for n in names]
        errs = [rows[n].get(f"{col}_std", 0.0) for n in names]
        bars = ax.bar(range(len(names)), vals, yerr=errs, color=[STRATEGY_COLORS[n] for n in names],
                      width=0.6, error_kw={"elinewidth": 0.8, "ecolor": INK_2, "capsize": 2})
        ax.set_xticks(range(len(names)), [STRATEGY_LABELS[n] for n in names])
        if log:
            _log_y(ax)
        _style(ax, title)
        _bar_labels(ax, bars, fmt)
    fig.legend([Patch(color=STRATEGY_COLORS[n]) for n in names], [n.replace("_", " ") for n in names],
               loc="lower center", ncol=4, fontsize=8, frameon=False)
    fig.suptitle("Experiment A: 100 requests, mean of 5 seeds (error bars = std)", fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "exp_a_strategies.png", dpi=DPI, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {out / 'exp_a_strategies.png'}")


def plot_exp_b(results: Path, out: Path) -> None:
    rows = sorted(_load(results, "exp_b_summary"), key=lambda r: r["max_running_seqs"])
    x = [r["max_running_seqs"] for r in rows]
    fig, axes = _figure(3, 12.5)
    specs = [("throughput_tok_s", "Throughput (tok/s)", False), ("mean_wait_ms", "Mean waiting time (ms, log)", True),
             ("peak_waiting", "Peak waiting-queue length", False)]
    for ax, (col, title, log) in zip(axes, specs):
        ax.plot(x, [r[col] for r in rows], color=BLUE, linewidth=2, marker="o", markersize=5)
        ax.set_xscale("log", base=2)
        ax.set_xticks(x, [str(v) for v in x])
        if log:
            _log_y(ax)
        _style(ax, title, xlabel="max_running_seqs")
    _save(fig, out, "exp_b_batch_size.png")


def plot_exp_c(results: Path, out: Path) -> None:
    rows = _load(results, "exp_c_summary")
    series = {"on_demand": ("on-demand growth + preemption", BLUE), "full_reserve": ("full reservation", ORANGE)}
    fig, axes = _figure(3, 12.5)
    specs = [("throughput_tok_s", "Throughput (tok/s)"), ("p95_latency_ms", "P95 latency (ms)"),
             ("preemptions", "Preemptions per run (recompute)")]
    for ax, (col, title) in zip(axes, specs):
        for mode, (label, color) in series.items():
            pts = sorted((r for r in rows if r["kv_mode"] == mode), key=lambda r: r["num_kv_blocks"])
            ax.plot([p["num_kv_blocks"] for p in pts], [p[col] for p in pts], color=color, linewidth=2,
                    marker="o", markersize=5, label=label)
        sizes = sorted({r["num_kv_blocks"] for r in rows}, reverse=True)
        ax.set_xscale("log", base=2)
        ax.set_xticks(sizes, [f"{int(v)}" for v in sizes], rotation=45)
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.invert_xaxis()
        _style(ax, title, xlabel="KV pool (blocks of 16 tokens), tighter ->")
    axes[0].legend(fontsize=8, frameon=False)
    _save(fig, out, "exp_c_kv_exhaustion.png")


def plot_exp_d(results: Path, out: Path) -> None:
    rows = _load(results, "exp_d_summary")
    labels = [("reference\n(no long prompt)" if "reference" in r["mode"] else
               "unchunked\n4096" if r["mode"] == "unchunked" else f"chunked\n{int(r['token_budget'])}") for r in rows]
    fig, axes = _figure(2, 11)
    specs = [("decoder_max_itl_ms", "Worst decode stall: max inter-token gap of running decoders (ms)"),
             ("long_mean_ttft_ms", "Time to first token of the 2048-token prompts (ms)")]
    for ax, (col, title) in zip(axes, specs):
        has_long = col != "long_mean_ttft_ms"  # the reference run has no long prompts
        idx = [i for i, r in enumerate(rows) if has_long or "reference" not in r["mode"]]
        colors = [INK_2 if "reference" in rows[i]["mode"] else BLUE for i in idx]
        bars = ax.bar(idx, [rows[i][col] for i in idx], color=colors, width=0.6)
        ax.set_xticks(idx, [labels[i] for i in idx])
        _style(ax, title)
        ax.title.set_fontsize(8.5)
        _bar_labels(ax, bars)
    _save(fig, out, "exp_d_long_prompt.png")


def plot_exp_e(results: Path, out: Path) -> None:
    rows = _load(results, "exp_e_summary")
    policies = ["fcfs", "priority", "priority+preempt"]
    fig, axes = _figure(2, 11)
    for ax, (col, title) in zip(axes, [("class_mean_wait_ms", "Mean waiting time (ms)"),
                                       ("class_mean_latency_ms", "Mean latency (ms)")]):
        for k, (prio, color, label) in enumerate([(0, BLUE, "priority 0 (high)"), (1, ORANGE, "priority 1 (normal)")]):
            vals = [next(r[col] for r in rows if r["policy"] == p and r["priority_class"] == prio) for p in policies]
            xs = [i + (k - 0.5) * 0.36 for i in range(len(policies))]
            bars = ax.bar(xs, vals, width=0.34, color=color, label=label)
            _bar_labels(ax, bars)
        ax.set_xticks(range(len(policies)), policies)
        _style(ax, title)
    axes[0].legend(fontsize=8, frameon=False)
    _save(fig, out, "exp_e_priorities.png")


def plot_exp_f(results: Path, out: Path) -> None:
    rows = _load(results, "exp_f_summary")
    families = {name: (name.replace("_", " "), STRATEGY_COLORS[name])  # same color per strategy as exp A
                for name in ("static", "continuous", "continuous_chunked")}
    fig, axes = _figure(2, 11)
    for ax, (col, title) in zip(axes, [("p95_latency_ms", "P95 end-to-end latency (ms, log)"),
                                       ("p95_token_gap_ms", "P95 token gap: time between consecutive tokens (ms, log)")]):
        for fam, (label, color) in families.items():
            pts = sorted((r for r in rows if r["family"] == fam), key=lambda r: r["max_running_seqs"])
            ax.plot([p["throughput_tok_s"] for p in pts], [p[col] for p in pts], color=color, linewidth=2,
                    marker="o", markersize=5, label=label)
            for p in pts:
                ax.annotate(str(p["max_running_seqs"]), (p["throughput_tok_s"], p[col]), fontsize=6.5,
                            color=INK_2, xytext=(3, 4), textcoords="offset points")
        _log_y(ax)
        _style(ax, title, xlabel="throughput (output tok/s); point labels = max_running_seqs")
    axes[0].legend(fontsize=8, frameon=False)
    _save(fig, out, "exp_f_throughput_vs_p95.png")


def plot_timeline(out: Path) -> None:
    requests, config = demo_scenario()
    result = Simulator(config).run(requests)
    segs = gantt_segments(result)
    style = {"W": ("#d9d8d3", 0.35), "R": ("#cfe0f5", 0.35), "P": (ORANGE, 0.7), "D": (BLUE, 0.7)}
    fig, ax = plt.subplots(figsize=(11, 3.8), facecolor=SURFACE)
    for rid, items in segs.items():
        for start, end, kind in sorted(items, key=lambda s: s[2] in "PD"):
            color, h = style[kind]
            ax.barh(rid, end - start, left=start, height=h, color=color, linewidth=0)
    for e in result.events:
        if e.kind == "PREEMPT":
            ax.plot(e.time_ms, e.request_id, marker="x", color=INK, markersize=8, markeredgewidth=2)
    ax.set_yticks(list(segs), [f"req {i}" for i in segs])
    ax.invert_yaxis()
    _style(ax, "Active-batch timeline (demo: 4 slots, chunked prefill, tight KV pool)", xlabel="simulated time (ms)")
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.legend([Patch(color=style[k][0]) for k in "WRPD"] + [plt.Line2D([], [], marker="x", color=INK, linestyle="")],
              ["waiting", "in batch (idle)", "prefill iteration", "decode iteration", "preempted (KV freed)"],
              ncol=5, fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    _save(fig, out, "timeline.png")


PLOTS = (plot_exp_a, plot_exp_b, plot_exp_c, plot_exp_d, plot_exp_e, plot_exp_f)


def _main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=RESULTS_DIR, help="where exp_*.json live")
    parser.add_argument("--out", type=Path, default=None, help="where PNGs go (default: --results)")
    args = parser.parse_args(argv)
    out = args.out or args.results
    for plot in PLOTS:
        plot(args.results, out)
    plot_timeline(out)


if __name__ == "__main__":
    _main()
