"""Render results/exp_{a,b,c}.json into PNGs for the README."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RESULTS = Path(__file__).parent / "results"
COLORS = {"cached": "#2a78b5", "uncached": "#c4552d", "prefill": "#2a78b5", "decode": "#c4552d"}


def _rows(name: str) -> list[dict]:
    return json.loads((RESULTS / f"{name}.json").read_text())["rows"]


def plot_prompt_scaling() -> None:
    rows = _rows("exp_a")
    lens = [r["prompt_len"] for r in rows]
    fig, (left, right) = plt.subplots(1, 2, figsize=(9, 3.4))
    left.plot(lens, [r["prefill_ms"] for r in rows], "o-", color=COLORS["prefill"])
    left.set(title="Prefill latency (TTFT)", xlabel="prompt tokens", ylabel="ms")
    right.plot(lens, [r["decode_ms_per_token"] for r in rows], "o-", color=COLORS["decode"])
    right.set(title="Decode latency per token", xlabel="prompt tokens", ylabel="ms/token")
    right.set_ylim(bottom=0)
    for ax in (left, right):
        ax.set_xscale("log", base=2)
        ax.set_xticks(lens, [str(n) for n in lens])
    fig.suptitle(f"Cached, {rows[0]['new_tokens']} new tokens", fontsize=9)
    fig.tight_layout()
    fig.savefig(RESULTS / "exp_a_prompt_scaling.png", dpi=150)
    plt.close(fig)


def plot_output_scaling() -> None:
    rows = _rows("exp_b")
    outs = [r["new_tokens"] for r in rows]
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    ax.plot(outs, [r["decode_total_s"] for r in rows], "o-", label="decode", color=COLORS["decode"])
    ax.plot(outs, [r["total_s"] for r in rows], "o-", label="end-to-end", color=COLORS["prefill"])
    ax.set(
        title=f"Output-length scaling (prompt={rows[0]['prompt_len']}, cached)",
        xlabel="new tokens",
        ylabel="seconds",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "exp_b_output_scaling.png", dpi=150)
    plt.close(fig)


def plot_kv_cache() -> None:
    rows = _rows("exp_c")
    prompt_len = rows[0]["prompt_len"]
    fig, (left, right) = plt.subplots(1, 2, figsize=(9, 3.4))
    for mode in ("cached", "uncached"):
        subset = [r for r in rows if r["mode"] == mode]
        left.plot(
            [r["new_tokens"] for r in subset],
            [r["total_s"] for r in subset],
            "o-",
            label=mode,
            color=COLORS[mode],
        )
        longest = max(subset, key=lambda r: r["new_tokens"])
        steps = longest["step_times_ms"]
        right.plot(
            [prompt_len + i + 1 for i in range(len(steps))],
            steps,
            label=mode,
            color=COLORS[mode],
            linewidth=1,
        )
    left.set(title=f"Total time (prompt={prompt_len})", xlabel="new tokens", ylabel="seconds")
    right.set(title="Per-step latency vs context length", xlabel="context tokens", ylabel="ms")
    left.legend()
    right.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "exp_c_kv_cache.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    plot_prompt_scaling()
    plot_output_scaling()
    plot_kv_cache()
    print("wrote PNGs to", RESULTS)
