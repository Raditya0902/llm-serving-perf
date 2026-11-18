"""Build the baseline-vs-modified table (and per-experiment tables) from result JSON files.

    python make_performance_report.py --raw results/raw/mps \
        --baseline "O1:day1" --modified "O1:deferred_sync" --out results/performance-tables.generated.md

`--baseline/--modified` are `<exp_id>:<label>`. The generated file never replaces the
hand-written performance-report.md. A metric a result file does not contain renders as
"not measured" (e.g. peak GPU memory on a Mac), never as a guess.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

MISSING = "not measured"
# (table label, row key, format, lower_is_better)
METRICS = (
    ("Requests/sec", "requests_per_s", "{:.3f}", False),
    ("Output tokens/sec", "output_tok_s", "{:.1f}", False),
    ("P95 TTFT (ms)", "ttft_ms_p95", "{:.1f}", True),
    ("P95 TPOT (ms)", "tpot_ms_p95", "{:.2f}", True),
    ("Peak memory (MB)", "peak_mem_mb", "{:.1f}", True),
)


def load_rows(raw_dir: Path, exp_id: str) -> list[dict]:
    path = raw_dir / f"exp_{exp_id.lower()}.json"
    if not path.is_file():
        raise FileNotFoundError(f"no results for {exp_id}: {path}")
    return json.loads(path.read_text())["rows"]


def find_row(raw_dir: Path, spec: str) -> dict:
    exp_id, _, label = spec.partition(":")
    for row in load_rows(raw_dir, exp_id):
        if row["label"] == label:
            return row
    raise KeyError(f"{spec}: label {label!r} not found")


def _fmt(row: dict, key: str, fmt: str) -> str:
    value = row.get(key)
    return MISSING if value is None else fmt.format(value)


def difference(base: dict, mod: dict, key: str) -> str:
    """Signed absolute and percent change, or MISSING when either side lacks the metric."""
    b, m = base.get(key), mod.get(key)
    if b is None or m is None:
        return MISSING
    if b == 0:
        return f"{m - b:+.4g} (n/a %)"
    return f"{m - b:+.4g} ({100.0 * (m / b - 1.0):+.1f}%)"


def comparison_table(base: dict, mod: dict, base_name: str, mod_name: str) -> str:
    lines = [
        f"| Metric | Baseline ({base_name}) | Modified ({mod_name}) | Difference |",
        "|---|---|---|---|",
    ]
    for name, key, fmt, _lower in METRICS:
        lines.append(
            f"| {name} | {_fmt(base, key, fmt)} | {_fmt(mod, key, fmt)} | {difference(base, mod, key)} |"
        )
    return "\n".join(lines)


def ranges_overlap(base: dict, mod: dict) -> bool:
    """Do the min..max requests/s ranges across repeats of the two chosen rows overlap?"""
    return (
        mod["requests_per_s_min"] <= base["requests_per_s_max"]
        and base["requests_per_s_min"] <= mod["requests_per_s_max"]
    )


def divergence_note(mod: dict) -> str:
    if mod.get("tokens_match", 1.0) < 1.0:
        return (f"WARNING: the modified row's outputs diverged from the baseline "
                f"(token match {mod['tokens_match']:.2f}); its speed is not comparable.")
    return ""


def noise_note(base: dict, mod: dict) -> str:
    if ranges_overlap(base, mod):
        return ("Requests/sec ranges across repeats overlap with the baseline: this difference is "
                "NOT clearly above run-to-run variation.")
    return "Requests/sec ranges across repeats do not overlap with the baseline."


def control_note(rows: list[dict]) -> str:
    """Report how far the A/A control sits from the baseline: the noise floor for this experiment."""
    controls = [r for r in rows if r.get("is_control")]
    if not controls:
        return ""
    c = controls[0]
    return (
        f"A/A control `{c['label']}` (identical work) differs from the baseline by "
        f"{c['requests_per_s_vs_baseline_pct']:+.1f}% requests/s and "
        f"{c['tpot_ms_p95_vs_baseline_pct']:+.1f}% P95 TPOT: that is the noise floor, so smaller "
        "differences between the baseline and the modified row are not evidence of an effect."
    )


UNSTABLE_SPREAD_FRACTION = 0.20


def stability_note(rows: list[dict]) -> str:
    """Warn when the wall-time spread across repeats is large: numbers from such rows are soft."""
    shaky = [r for r in rows if r.get("wall_spread_fraction", 0.0) > UNSTABLE_SPREAD_FRACTION]
    if not shaky:
        return ""
    detail = ", ".join(f"{r['label']} ({r['wall_spread_fraction']:.0%})" for r in shaky)
    return (f"WARNING: wall-time spread across repeats exceeds {UNSTABLE_SPREAD_FRACTION:.0%} for {detail}. "
            "Treat P95 values and small differences from these rows as unreliable; repeat the run.")


def experiment_table(rows: list[dict]) -> str:
    header = "| Value | req/s | out tok/s | P95 TTFT ms | P95 TPOT ms | peak MB | tokens match | req/s vs base |"
    lines = [header, "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        mark = " (baseline)" if r.get("is_baseline") else ""
        lines.append(
            f"| {r['label']}{mark} | {r['requests_per_s']:.3f} | {r['output_tok_s']:.1f} | "
            f"{r['ttft_ms_p95']:.1f} | {r['tpot_ms_p95']:.2f} | {r['peak_mem_mb']:.1f} | "
            f"{r['tokens_match']:.2f} | {r['requests_per_s_vs_baseline_pct']:+.1f}% |"
        )
    return "\n".join(lines)


def build_report(raw_dir: Path, baseline: str, modified: str) -> str:
    if baseline.partition(":")[0] != modified.partition(":")[0]:
        raise ValueError("baseline and modified must come from the same experiment")
    base, mod = find_row(raw_dir, baseline), find_row(raw_dir, modified)
    if not base.get("is_baseline"):
        raise ValueError(f"{baseline} is not the experiment's baseline row; `tokens_match` and deltas are relative to it")
    parts = [
        "# Generated tables (do not hand-edit; interpretation lives in performance-report.md)",
        "",
        "## Baseline vs modified",
        "",
        comparison_table(base, mod, baseline, modified),
        "",
        noise_note(base, mod),
        f"Token match vs baseline: {mod['tokens_match']:.2f}.",
    ]
    if divergence_note(mod):
        parts += ["", divergence_note(mod)]
    for exp_id in sorted({baseline.split(":")[0], modified.split(":")[0]}):
        rows = load_rows(raw_dir, exp_id)
        parts += ["", f"## {exp_id}", "", experiment_table(rows)]
        for note in (control_note(rows), stability_note(rows)):
            if note:
                parts += ["", note]
    return "\n".join(parts) + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--modified", required=True)
    parser.add_argument("--out", type=Path, default=Path("results/performance-tables.generated.md"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.out.name == "performance-report.md":
        raise SystemExit("refusing to overwrite the hand-written performance-report.md")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(build_report(args.raw, args.baseline, args.modified))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
