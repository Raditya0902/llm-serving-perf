import os
import subprocess
from pathlib import Path

import pytest

GPU = Path(__file__).resolve().parent.parent / "gpu"
SCRIPTS = ["run_busyfrac.sh", "run_round5_a6000.sh", "setup_lambda.sh", "run_ncu.sh", "run_round5_h100.sh", "run_models_matrix.sh"]


@pytest.mark.parametrize("name", SCRIPTS)
def test_script_parses(name):
    assert subprocess.run(["bash", "-n", str(GPU / name)], capture_output=True).returncode == 0


def test_busyfrac_refuses_without_nsys(tmp_path):
    env = {"PATH": "/usr/bin:/bin", "HOME": str(tmp_path)}
    result = subprocess.run(["bash", str(GPU / "run_busyfrac.sh"), str(tmp_path / "out")], capture_output=True, text=True, env=env)
    if subprocess.run(["bash", "-c", "command -v nsys"], capture_output=True, env=env).returncode == 0:
        pytest.skip("nsys is on the default PATH here")
    assert result.returncode == 2 and "nsys not found" in result.stderr


def test_ncu_script_requests_the_counters_the_summary_reads():
    text = (GPU / "run_ncu.sh").read_text()
    for metric in ("dram__bytes_read.sum", "launch__grid_size", "launch__waves_per_multiprocessor"):
        assert metric in text
