"""Plots from results/measured (real MPS runs). Nothing here reads simulated or synthetic data.

    python plot_results.py            # writes results/*.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

MEASURED = Path(__file__).parent / "results" / "measured"
OUT = Path(__file__).parent / "results"
# Okabe-Ito, colour-blind safe
BLUE, ORANGE, GREEN, GREY = "#0072B2", "#E69F00", "#009E73", "#999999"


def _rows(run: str, exp: str, root: Path) -> list[dict]:
    return json.loads((root / run / f"exp_{exp}.json").read_text())["rows"]


def plot_profile(root: Path, out: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, phase in zip(axes, ("decode", "prefill")):
        labels, parts = [], {"matmul": [], "attention core (est.)": [], "other": []}
        for impl, run in (("sdpa", "mps-baseline"), ("eager", "mps-eager")):
            attr = json.loads((root / run / "profile_summary.json").read_text())["device_attribution"][phase]
            labels.append(impl)
            parts["matmul"].append(attr["matmul_ms"])
            parts["attention core (est.)"].append(attr["attention_core_ms"])
            parts["other"].append(attr["other_ms"])
        bottom = [0.0] * len(labels)
        for (name, vals), colour in zip(parts.items(), (BLUE, ORANGE, GREY)):
            ax.bar(labels, vals, bottom=bottom, label=name, color=colour)
            bottom = [b + v for b, v in zip(bottom, vals)]
        ax.set_title(f"{phase}: ms per step (Qwen2.5-0.5B, MPS)")
        ax.set_ylabel("ms")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    path = out / "profile_breakdown.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_batching(root: Path, out: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for run, colour in (("mps", BLUE), ("mps-r2", ORANGE)):
        rows = _rows(run, "e1", root)
        x = [int(r["label"]) for r in rows]
        axes[0].plot(x, [r["output_tok_s"] for r in rows], "o-", color=colour, label=f"run {run}")
        axes[1].plot(x, [r["tpot_ms_p50"] for r in rows], "o-", color=colour, label=f"run {run}")
    axes[0].set(title="Output tokens/s vs batch size", xlabel="batch size (8 requests)", ylabel="tok/s")
    axes[1].set(title="Median TPOT vs batch size", xlabel="batch size", ylabel="ms/token")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    path = out / "e1_batching.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_scaling(root: Path, out: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for run, colour in (("mps", BLUE), ("mps-r2", ORANGE)):
        rows = _rows(run, "e2", root)
        axes[0].plot([int(r["label"]) for r in rows], [r["ttft_ms_p50"] for r in rows], "o-",
                     color=colour, label=f"run {run}")
    axes[0].set(title="TTFT vs prompt length", xlabel="prompt tokens", ylabel="ms (median)",
                xscale="log", yscale="log")
    axes[0].legend(frameon=False)
    rows = _rows("mps", "e3", root)
    axes[1].plot([int(r["label"]) for r in rows], [r["tpot_ms_p50"] for r in rows], "o-", color=GREEN)
    axes[1].set(title="TPOT vs output length", xlabel="output tokens", ylabel="ms/token (median)")
    axes[1].set_ylim(bottom=0)
    fig.tight_layout()
    path = out / "e2_e3_scaling.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_variants(root: Path, out: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=False)
    for ax, exp, title in ((axes[0], "o1", "O1: defer host syncs"), (axes[1], "o2", "O2: sdpa vs eager")):
        rows = _rows("mps", exp, root)
        labels = [r["label"] for r in rows]
        vals = [r["requests_per_s"] for r in rows]
        err = [[v - r["requests_per_s_min"] for v, r in zip(vals, rows)],
               [r["requests_per_s_max"] - v for v, r in zip(vals, rows)]]
        colours = [GREY if r["is_control"] else (BLUE if r["is_baseline"] else ORANGE) for r in rows]
        hatches = ["//" if r.get("tokens_match", 1.0) < 1.0 else "" for r in rows]  # hatched = outputs diverged
        ax.bar(labels, vals, yerr=err, color=colours, capsize=4)
        for bar, hatch in zip(ax.patches, hatches):
            bar.set_hatch(hatch)
        ax.set(title=title, ylabel="requests/s (bars: min..max of repeats)")
        ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    path = out / "o1_o2_variants.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measured", type=Path, default=MEASURED)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    for plot in (plot_profile, plot_batching, plot_scaling, plot_variants):
        print(plot(args.measured, args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
