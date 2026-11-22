"""Provenance block attached to every result file (so a number can be traced and reproduced)."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from pathlib import Path

import torch
import transformers

REPO_ROOT = Path(__file__).resolve().parent.parent
FINGERPRINT_LENGTH = 12


def config_fingerprint(config: dict) -> str:
    payload = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:FINGERPRINT_LENGTH]


def git_short_hash() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=5, check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def git_is_dirty() -> bool | None:
    """True if `06-profiling` or `01-inference-loop` has uncommitted/untracked changes (the hash then does not identify the code)."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain", "--", "06-profiling", "01-inference-loop"],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=5, check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return bool(result.stdout.strip())


def collect_meta(device: torch.device, model, config: dict, smoke: bool) -> dict:
    return {
        "device": str(device),
        "dtype": str(next(model.parameters()).dtype),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "machine": platform.platform(),
        "git": git_short_hash(),
        "git_dirty": git_is_dirty(),
        "smoke": smoke,
        "config": config,
        "config_id": config_fingerprint(config),
    }
