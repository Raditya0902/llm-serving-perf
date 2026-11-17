"""End-to-end: the scripts run under a real torchrun via launch.sh (the user-facing path)."""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

DAY_DIR = Path(__file__).resolve().parent.parent
LAUNCH = DAY_DIR / "launch.sh"
TIMEOUT_S = 120

pytestmark = pytest.mark.skipif(
    not (DAY_DIR / ".venv" / "bin" / "torchrun").exists(), reason="needs the day-5 venv's torchrun"
)


def _launch(nproc: int, script: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(LAUNCH), str(nproc), script],
        capture_output=True, text=True, timeout=TIMEOUT_S, cwd=DAY_DIR, check=False,
    )


def test_collectives_script_passes_on_two_processes_under_torchrun() -> None:
    result = _launch(2, "collectives.py")

    assert result.returncode == 0, result.stderr[-2000:]
    assert "collectives demo: all checks passed" in result.stdout
    for rank in (0, 1):
        assert f"[rank {rank}] identity: rank {rank} of 2" in result.stdout


def test_tp_demo_script_passes_on_two_processes_under_torchrun() -> None:
    result = _launch(2, "tp_demo.py")

    assert result.returncode == 0, result.stderr[-2000:]
    assert result.stdout.count("PASS") == 6 and "FAIL" not in result.stdout


def test_launcher_rejects_missing_arguments() -> None:
    result = subprocess.run([str(LAUNCH), "2"], capture_output=True, text=True, check=False)

    assert result.returncode == 2 and "usage" in result.stderr
