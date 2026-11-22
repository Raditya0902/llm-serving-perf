"""Make the `01-inference-loop` modules (generate, device, model_io, prompts, bench) importable.

`01-inference-loop` is a flat set of scripts, not a package. The directory is *appended* to
`sys.path` so `06-profiling` modules win on name clashes (both days have a `run_*experiments.py`).
`01-inference-loop` code is used as-is; nothing here modifies it.
"""

from __future__ import annotations

import sys
from pathlib import Path

DAY1_DIR = Path(__file__).resolve().parent.parent / "01-inference-loop"


def ensure_day1_on_path() -> Path:
    if not DAY1_DIR.is_dir():
        raise FileNotFoundError(f"`01-inference-loop` directory not found: {DAY1_DIR}")
    entry = str(DAY1_DIR)
    if entry not in sys.path:
        sys.path.append(entry)
    return DAY1_DIR


ensure_day1_on_path()
