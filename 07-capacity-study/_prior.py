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
    LAB_DIR / "02-kv-cache-memory",
    LAB_DIR / "03-batching-simulator",
    LAB_DIR / "04-vllm-internals",
    LAB_DIR / "05-tensor-parallel",
)
# Day 6 is deliberately NOT importable from here: it was developed on a separate branch
# (now merged into ../06-profiling). Its GPU-memory sampler is re-implemented in
# `gpu_mem.py`; its results are cited by path.
VLLM_SRC_DIR = LAB_DIR / "04-vllm-internals" / "vllm_src"


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
