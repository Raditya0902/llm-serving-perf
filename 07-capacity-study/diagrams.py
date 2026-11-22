"""Inference architecture diagram (report section 1), drawn with matplotlib.

    .venv/bin/python diagrams.py [--out results/architecture.png]

Process and thread layout follows `04-vllm-internals`'s SOURCE_TRACE.md section 0 (vLLM v0.34.0). Names and
file:line citations live there; this picture only shows who owns what and what crosses a
process boundary each step.
"""
from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

DEFAULT_OUT = Path(__file__).resolve().parent / "results" / "architecture.png"
DPI = 150
PANEL_FILL = {"api": "#E8F1FA", "engine": "#FDF1E3", "worker": "#E6F4EC"}
EDGE = "#333333"


def _panel(ax, x: float, w: float, title: str, key: str) -> None:
    ax.add_patch(FancyBboxPatch((x, 0.06), w, 0.84, boxstyle="round,pad=0.01,rounding_size=0.015",
                                fc=PANEL_FILL[key], ec=EDGE, lw=1.2))
    ax.text(x + w / 2, 0.87, title, ha="center", va="top", fontsize=10, fontweight="bold")


def _box(ax, x: float, y: float, w: float, h: float, text: str, fc: str = "white") -> None:
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.005,rounding_size=0.01",
                                fc=fc, ec=EDGE, lw=1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8)


def _arrow(ax, start: tuple[float, float], end: tuple[float, float], label: str = "", dy: float = 0.025) -> None:
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=12, lw=1.3, color=EDGE))
    if label:  # white backing keeps the label legible where it crosses a line
        ax.text((start[0] + end[0]) / 2, (start[1] + end[1]) / 2 + dy, label, ha="center", fontsize=7.5,
                style="italic", bbox={"fc": "white", "ec": "none", "alpha": 0.85, "pad": 1.5})


def draw(path: Path = DEFAULT_OUT) -> Path:
    fig, ax = plt.subplots(figsize=(13, 6.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    _panel(ax, 0.01, 0.22, "API server process", "api")
    _box(ax, 0.03, 0.70, 0.18, 0.09, "Clients\n(HTTP, streaming)")
    _box(ax, 0.03, 0.52, 0.18, 0.12, "AsyncLLM frontend\ntokenize prompt\nstream detokenized text")
    _box(ax, 0.03, 0.30, 0.18, 0.12, "benchmark client\nclosed loop:\nN requests in flight")

    _panel(ax, 0.30, 0.34, "EngineCore process (CPU only)", "engine")
    _box(ax, 0.32, 0.62, 0.30, 0.10, "input thread (ZMQ): preprocess,\nhash prompt blocks")
    _box(ax, 0.32, 0.37, 0.14, 0.20, "Scheduler\nwaiting / running\ntoken budget\nmax_num_seqs\npreempt = recompute", "#FFF8EC")
    _box(ax, 0.48, 0.37, 0.14, 0.20, "KVCacheManager\nBlockPool free list\nprefix-hash map\nallocate_slots()", "#FFF8EC")
    _box(ax, 0.32, 0.12, 0.30, 0.18, "per step:\n1 schedule()  2 execute_model\n3 sample_tokens  4 update_from_output\nnum_computed_tokens chases num_tokens")

    _panel(ax, 0.73, 0.26, "Worker process (one per GPU rank)", "worker")
    _box(ax, 0.75, 0.62, 0.22, 0.10, "GPUModelRunner\npersistent batch + block tables")
    _box(ax, 0.75, 0.42, 0.22, 0.14, "forward pass\nprefill chunk + decode tokens\npaged attention, CUDA graphs")
    _box(ax, 0.75, 0.24, 0.22, 0.12, "sampler -> token ids\nasync device-to-host copy")
    _box(ax, 0.75, 0.10, 0.22, 0.09, "HBM: weights + KV block pool\n(size = util x total - weights - act.)", "#DFF0E6")

    _arrow(ax, (0.12, 0.70), (0.12, 0.64))
    _arrow(ax, (0.21, 0.58), (0.32, 0.66), "ZMQ\nrequests", dy=0.045)
    _arrow(ax, (0.39, 0.62), (0.39, 0.57))
    _arrow(ax, (0.46, 0.47), (0.48, 0.47))
    _arrow(ax, (0.62, 0.30), (0.75, 0.66), "SchedulerOutput\ndiffs (shm ring)", dy=0.0)
    _arrow(ax, (0.86, 0.62), (0.86, 0.56))
    _arrow(ax, (0.86, 0.42), (0.86, 0.36))
    _arrow(ax, (0.75, 0.30), (0.62, 0.22), "sampled tokens", dy=-0.04)
    _arrow(ax, (0.32, 0.22), (0.21, 0.55), "EngineCore\nOutputs (ZMQ)", dy=0.0)
    ax.text(0.5, 0.015, "TP>1: every worker runs the same step; 2 AllReduces per layer over NVLink/NCCL "
            "(`05-tensor-parallel`).  Only block-id diffs and token ids cross process boundaries.",
            ha="center", fontsize=8.5, color="#444444")
    ax.set_title("LLM inference stack: prompt to streamed tokens (vLLM v0.34.0 process layout)",
                 fontsize=12, fontweight="bold")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    draw(p.parse_args(argv).out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
