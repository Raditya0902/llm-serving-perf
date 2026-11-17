"""Make earlier days' flat modules importable (Day 2, 3, 4, 5, 6).

Directories are *appended* to ``sys.path`` so Day 7's own modules win on name
clashes (several days share names such as ``run_experiments`` or ``plot_results``).
Only import specific modules from the prior days; never ``import *``.
Nothing here modifies prior-day code.
"""

from __future__ import annotations

import sys
from pathlib import Path

LAB_DIR = Path(__file__).resolve().parent.parent
PRIOR_DIRS: tuple[Path, ...] = (
    LAB_DIR / "day2-gpu-memory",
    LAB_DIR / "day3-scheduling",
    LAB_DIR / "day4-vllm-internals",
    LAB_DIR / "day5-distributed",
)
# Day 6 is deliberately NOT importable from here: it lives on its own git branch
# (`day6-profiling`), so a Day 7 checkout cannot rely on its files being present.
# Its GPU-memory sampler is re-implemented in `gpu_mem.py`; its results are cited from git.
VLLM_SRC_DIR = LAB_DIR / "day4-vllm-internals" / "vllm_src"


def ensure_prior_days_on_path() -> tuple[Path, ...]:
    missing = [d for d in PRIOR_DIRS if not d.is_dir()]
    if missing:
        raise FileNotFoundError(f"prior-day directories not found: {missing}")
    for directory in PRIOR_DIRS:
        entry = str(directory)
        if entry not in sys.path:
            sys.path.append(entry)
    return PRIOR_DIRS


ensure_prior_days_on_path()
