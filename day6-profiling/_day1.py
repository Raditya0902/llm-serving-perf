"""Make the Day 1 modules (generate, device, model_io, prompts, bench) importable.

Day 1 is a flat set of scripts, not a package. The directory is *appended* to
`sys.path` so Day 6 modules win on name clashes (both days have a `run_*experiments.py`).
Day 1 code is used as-is; nothing here modifies it.
"""

from __future__ import annotations

import sys
from pathlib import Path

DAY1_DIR = Path(__file__).resolve().parent.parent / "day1-inference"


def ensure_day1_on_path() -> Path:
    if not DAY1_DIR.is_dir():
        raise FileNotFoundError(f"Day 1 directory not found: {DAY1_DIR}")
    entry = str(DAY1_DIR)
    if entry not in sys.path:
        sys.path.append(entry)
    return DAY1_DIR


ensure_day1_on_path()
