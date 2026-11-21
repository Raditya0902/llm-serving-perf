"""Plot one chart per swept knob from ``results/sweep_summary.json``.

Each chart shows, against the knob's values (baseline included and ringed):
  left : output token throughput (tok/s)
  right: p99 time-to-first-token and p99 inter-token latency (ms, log scale)
so the throughput-vs-latency trade-off of a scheduling parameter is visible at
a glance.  The x axis is categorical so ``request_rate = inf`` fits naturally.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")  # headless: must precede the pyplot import
import matplotlib.pyplot as plt  # noqa: E402

from sweep_config import SWEEP_KNOBS, format_value  # noqa: E402

MIN_POINTS = 2
INT_COLUMNS = ("max_num_seqs", "max_num_batched_tokens", "input_len", "output_len",
               "n_reps", "completed", "failed")
FLOAT_COLUMNS = ("gpu_memory_utilization", "request_rate")
TEXT_COLUMNS = ("run_id", "knob", "value", "model")
TTFT_COLOR, ITL_COLOR, THROUGHPUT_COLOR = "#D55E00", "#0072B2", "#009E73"  # Okabe-Ito
DPI = 120


def load_summary(path: Path) -> list[dict[str, Any]]:
    """Read ``sweep_summary.json``; restore numeric types and ``"inf"``."""
    data = json.loads(path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for raw in data["rows"]:
        row = dict(raw)
        for col in INT_COLUMNS:
            if col in row:
                row[col] = int(row[col])
        for col in FLOAT_COLUMNS:
            if col in row:
                row[col] = float(row[col])  # float("inf") accepts "inf"
        for col, value in row.items():
            # non-finite metrics are stored as the strings "nan"/"inf"/"-inf"
            if isinstance(value, str) and col not in TEXT_COLUMNS:
                row[col] = float(value)
        rows.append(row)
    return rows


def knob_series(rows: Sequence[Mapping[str, Any]], knob: str) -> list[Mapping[str, Any]]:
    """Rows for ``knob`` plus the baseline, ordered by the knob's value."""
    if knob not in SWEEP_KNOBS:
        raise ValueError(f"unknown knob {knob!r}; valid: {sorted(SWEEP_KNOBS)}")
    picked = [r for r in rows if r["knob"] in (knob, "baseline")]
    return sorted(picked, key=lambda r: r[knob])  # inf sorts last


def value_label(value: float | int) -> str:
    """Axis/table label for a knob value (``inf`` for unpaced request rate)."""
    return "inf" if isinstance(value, float) and math.isinf(value) else format_value(value)


def plot_knob(
    rows: Sequence[Mapping[str, Any]], knob: str, out_dir: Path, *, watermark: str | None = None,
) -> Path | None:
    """Write ``sweep_<knob>.png``; return None if there are too few points."""
    series = knob_series(rows, knob)
    if len(series) < MIN_POINTS:
        return None

    xs = list(range(len(series)))
    labels = [value_label(r[knob]) for r in series]
    base_x = [i for i, r in enumerate(series) if r["knob"] == "baseline"]

    fig, (ax_tput, ax_lat) = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    ax_tput.plot(xs, [r["output_throughput"] for r in series], marker="o",
                 color=THROUGHPUT_COLOR)
    ax_tput.set_ylabel("output throughput (tok/s)")
    ax_lat.plot(xs, [r["p99_ttft_ms"] for r in series], marker="o", color=TTFT_COLOR,
                label="p99 TTFT")
    ax_lat.plot(xs, [r["p99_itl_ms"] for r in series], marker="s", color=ITL_COLOR,
                label="p99 ITL")
    ax_lat.set_yscale("log")
    ax_lat.set_ylabel("latency (ms, log)")
    ax_lat.legend(frameon=False)

    for ax in (ax_tput, ax_lat):
        ax.set_xticks(xs, labels)
        ax.set_xlabel(knob)
        ax.grid(alpha=0.3)
        for i in base_x:  # ring the baseline configuration
            ax.axvline(i, color="grey", linestyle=":", linewidth=1)
    fig.suptitle(f"Sweep: {knob} (dotted line = baseline)")
    if watermark:
        fig.text(0.5, 0.5, watermark, ha="center", va="center", fontsize=40,
                 color="grey", alpha=0.25, rotation=25)

    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"sweep_{knob}.png"
        fig.savefig(path, dpi=DPI)
    finally:
        plt.close(fig)  # never leak a figure, even if the disk write fails
    return path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--summary", type=Path, default=Path("results/sweep_summary.json"))
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--watermark", default=None,
                        help="text stamped across charts (e.g. SYNTHETIC for fixture data)")
    args = parser.parse_args(argv)
    if not args.summary.exists():
        print(f"error: summary file {args.summary} not found; run parse_results.py first",
              file=sys.stderr)
        return 1
    rows = load_summary(args.summary)
    written = [plot_knob(rows, knob, args.out, watermark=args.watermark) for knob in SWEEP_KNOBS]
    for path in filter(None, written):
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
