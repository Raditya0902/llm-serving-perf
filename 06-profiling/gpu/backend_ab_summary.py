"""Summarise backend A/B runs from gpu/run_backend_ab.sh / gpu/run_models_matrix.sh.

    python -m gpu.backend_ab_summary <root>      # <root>/<model>/s<seqs>/<BACKEND>_<in>x<out>_r<rep>.json

Backend tags are the --attention-backend name, with `-fa<N>` appended when the FlashAttention generation was forced
(`FLASH_ATTN:3` in BACKENDS becomes `FLASH_ATTN-fa3`). Reports per cell mean / spread / median ITL, the gain of every
backend over the base (FLASH_ATTN) and the change from the smallest to the largest max_num_seqs. Runs with failed
requests are listed and left out.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

NAME = re.compile(r"^([A-Z][A-Z_]*(?:-fa\d)?)_(\d+x\d+)_r(\d+)\.json$")
SEQS_DIR = re.compile(r"^s(\d+)$")
BASE_BACKEND = "FLASH_ATTN"


def parse_name(name: str) -> tuple[str, str, int]:
    match = NAME.match(name)
    if not match:
        raise ValueError(f"not a backend A/B result name: {name}")
    return match.group(1), match.group(2), int(match.group(3))


def gather(root: Path) -> tuple[dict[tuple[str, int, str, str], list[dict]], list[str]]:
    cells: dict[tuple[str, int, str, str], list[dict]] = defaultdict(list)
    rejected: list[str] = []
    root = Path(root)
    for path in sorted(root.rglob("*_r*.json")):
        seqs_match = SEQS_DIR.match(path.parent.name)
        if not seqs_match:
            continue
        try:
            backend, workload, _rep = parse_name(path.name)
        except ValueError:
            continue
        model = path.parent.parent.name if path.parent.parent != root else ""
        data = json.loads(path.read_text())
        if data.get("failed", 0) or not data.get("completed"):
            rejected.append(str(path.relative_to(root)))
            continue
        cells[(model, int(seqs_match.group(1)), workload, backend)].append(data)
    return cells, rejected


def summarize(root: Path, base_backend: str = BASE_BACKEND) -> dict:
    cells, rejected = gather(root)
    means = {key: float(np.mean([r["output_throughput"] for r in runs])) for key, runs in cells.items()}
    rows = []
    for key in sorted(cells):
        model, seqs, workload, backend = key
        tok = [r["output_throughput"] for r in cells[key]]
        rows.append({"model": model, "seqs": seqs, "workload": workload, "backend": backend, "n": len(tok), "mean_tok_s": means[key],
                     "min_tok_s": min(tok), "max_tok_s": max(tok), "spread_pct": (max(tok) / min(tok) - 1) * 100,
                     "median_itl_ms": float(np.mean([r["median_itl_ms"] for r in cells[key]]))})
    gains = [{"model": m, "seqs": s, "workload": w, "backend": b, "gain_pct_vs_base": (means[(m, s, w, b)] / means[(m, s, w, base_backend)] - 1) * 100}
             for (m, s, w, b) in sorted(cells) if b != base_backend and (m, s, w, base_backend) in means]
    losses = []
    for (m, w, b) in sorted({(m, w, b) for (m, _s, w, b) in cells}):
        seqs = sorted(s for (mm, s, ww, bb) in cells if (mm, ww, bb) == (m, w, b))
        if len(seqs) >= 2:
            lo, hi = seqs[0], seqs[-1]
            losses.append({"model": m, "workload": w, "backend": b, "low_seqs": lo, "high_seqs": hi,
                           "loss_pct": (means[(m, hi, w, b)] / means[(m, lo, w, b)] - 1) * 100})
    return {"cells": rows, "gains": gains, "seqs_loss": losses, "rejected": rejected}


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("root")
    parser.add_argument("--base", default=BASE_BACKEND, help="backend the gains are measured against (e.g. TRITON_ATTN on a Hopper pod)")
    args = parser.parse_args(argv)
    print(json.dumps(summarize(Path(args.root), args.base), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
