"""Run the round-4 shell drivers against stub `vllm`, `curl`, `setsid` and `sleep` so tag names, flags and file layout are checked without a GPU."""
from __future__ import annotations

import json
import os
import stat
import subprocess
from pathlib import Path

import pytest

from gpu import backend_ab_summary as bs
from gpu import e123_summary as e123

DAY6 = Path(__file__).resolve().parent.parent

FAKE_VLLM = r"""#!/usr/bin/env bash
echo "$@" >> "$STUB_LOG"
if [[ "$1" == bench ]]; then
  dir=""; file=""; prev=""
  for a in "$@"; do
    [[ "$prev" == --result-dir ]] && dir="$a"; [[ "$prev" == --result-filename ]] && file="$a"; prev="$a"
  done
  [[ -n "$dir" ]] && echo '{"output_throughput": 1000, "median_itl_ms": 5, "p99_itl_ms": 9, "median_tpot_ms": 5, "median_ttft_ms": 20, "failed": 0, "completed": 10}' > "$dir/$file"
fi
exit 0
"""


@pytest.fixture()
def stub_env(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    scripts = {"vllm": FAKE_VLLM, "curl": "#!/usr/bin/env bash\nexit 0\n", "sleep": "#!/usr/bin/env bash\nexit 0\n",
               "setsid": '#!/usr/bin/env bash\nexec "$@"\n'}
    for name, body in scripts.items():
        path = bin_dir / name
        path.write_text(body)
        path.chmod(path.stat().st_mode | stat.S_IEXEC)
    log = tmp_path / "stub.log"
    env = os.environ | {"PATH": f"{bin_dir}:{os.environ['PATH']}", "STUB_LOG": str(log)}
    return env, log, tmp_path


def _run(script, args, env, extra=None):
    return subprocess.run(["bash", str(DAY6 / "gpu" / script), *args], env=env | (extra or {}), capture_output=True, text=True, cwd=DAY6, timeout=60)


def test_backend_ab_tags_forced_fa_versions_and_passes_the_attention_config(stub_env):
    env, log, tmp = stub_env
    out = tmp / "ab"
    done = _run("run_backend_ab.sh", [str(out)], env, {"BACKENDS": "FLASH_ATTN:3 TRITON_ATTN", "REPS": "1", "WORKLOADS": "512x128", "NUM_PROMPTS": "4000", "BT": "2048", "SEQS": "64"})
    assert done.returncode == 0, done.stderr
    assert (out / "FLASH_ATTN-fa3_512x128_r0.json").is_file() and (out / "TRITON_ATTN_512x128_r0.json").is_file() and (out / "DONE").is_file()
    lines = log.read_text().splitlines()
    serve = [ln for ln in lines if ln.startswith("serve")]
    assert "--attention-backend FLASH_ATTN --attention-config.flash_attn_version 3" in serve[0]
    assert "--attention-config" not in serve[1] and "--attention-backend TRITON_ATTN" in serve[1]
    assert "--max-num-batched-tokens 2048" in serve[0] and "--max-num-seqs 64" in serve[0]
    assert all("--num-prompts 4000" in ln for ln in lines if ln.startswith("bench"))


def test_backend_ab_defaults_keep_the_round3_command_line(stub_env):
    env, log, tmp = stub_env
    _run("run_backend_ab.sh", [str(tmp / "ab")], env, {"REPS": "1", "WORKLOADS": "512x128", "BACKENDS": "FLASH_ATTN"})
    serve = [ln for ln in log.read_text().splitlines() if ln.startswith("serve")][0]
    assert "--max-model-len 4096" in serve and "--max-num-batched-tokens 8192" in serve and "--max-num-seqs 256" in serve
    assert (tmp / "ab" / "FLASH_ATTN_512x128_r0.json").is_file()


def test_models_matrix_writes_the_layout_the_summary_reads(stub_env):
    env, log, tmp = stub_env
    out = tmp / "matrix"
    done = _run("run_models_matrix.sh", [str(out)], env, {"REPS": "1", "ONLY": "qwen2.5-3b"})
    assert done.returncode == 0, done.stderr
    summary = bs.summarize(out)
    keys = {(c["model"], c["seqs"], c["workload"], c["backend"]) for c in summary["cells"]}
    assert ("qwen2.5-3b", 128, "3584x128", "TRITON_ATTN") in keys and ("qwen2.5-3b", 1024, "512x128", "FLASH_ATTN") in keys
    assert not any(c["model"] != "qwen2.5-3b" for c in summary["cells"]) and not summary["rejected"]


def test_models_matrix_runs_the_8b_at_seq_limits_its_kv_pool_can_hold(stub_env):
    env, log, tmp = stub_env
    _run("run_models_matrix.sh", [str(tmp / "m")], env, {"REPS": "1", "ONLY": "llama3.1-8b"})
    serves = [ln for ln in log.read_text().splitlines() if ln.startswith("serve")]
    assert {"--max-num-seqs 64", "--max-num-seqs 192"} <= {s for ln in serves for s in ("--max-num-seqs 64", "--max-num-seqs 192") if s in ln}
    assert all("--max-num-batched-tokens 2048" in ln for ln in serves) and not any("--max-num-seqs 1024" in ln for ln in serves)


def test_e123_runner_writes_every_file_the_summary_needs(stub_env):
    env, log, tmp = stub_env
    out = tmp / "e123"
    done = _run("run_e123_vllm.sh", [str(out)], env, {"MODELS": "tiny|Qwen/Tiny"})
    assert done.returncode == 0, done.stderr
    names = {p.name for p in (out / "tiny").glob("*.json")}
    assert {"e1_c1.json", "e1_c8.json", "e2_in512.json", "e2_in2048.json", "e3_out32.json", "e3_out256.json"} <= names
    result = e123.summarize(out)["tiny"]
    assert result["E1"]["confirmed"] is not None and result["E2"]["confirmed"] is not None and result["E3"]["confirmed"] is not None
    assert "--max-concurrency 1" in log.read_text()


def test_h100_script_covers_fa3_fa2_and_triton(stub_env):
    text = (DAY6 / "gpu" / "run_round4_h100.sh").read_text()
    assert "FLASH_ATTN:3 FLASH_ATTN:2 TRITON_ATTN" in text and "--fa-version" in text


def test_backend_ab_reruns_a_result_with_failed_requests_instead_of_skipping_it(stub_env):
    env, log, tmp = stub_env
    out = tmp / "ab"
    out.mkdir()
    (out / "FLASH_ATTN_512x128_r0.json").write_text(json.dumps({"output_throughput": 1, "failed": 2995, "completed": 1005}))
    _run("run_backend_ab.sh", [str(out)], env, {"REPS": "1", "WORKLOADS": "512x128", "BACKENDS": "FLASH_ATTN"})
    assert json.loads((out / "FLASH_ATTN_512x128_r0.json").read_text())["failed"] == 0
    assert "SKIP" not in (out / "driver.log").read_text()


def test_backend_ab_skips_a_clean_finished_result(stub_env):
    env, log, tmp = stub_env
    out = tmp / "ab"
    out.mkdir()
    (out / "FLASH_ATTN_512x128_r0.json").write_text(json.dumps({"output_throughput": 1, "failed": 0, "completed": 1000}))
    _run("run_backend_ab.sh", [str(out)], env, {"REPS": "1", "WORKLOADS": "512x128", "BACKENDS": "FLASH_ATTN"})
    assert "SKIP FLASH_ATTN_512x128_r0" in (out / "driver.log").read_text() and not log.exists()


def test_the_bench_drivers_raise_the_open_file_limit_before_sending_thousands_of_requests():
    for name in ("run_backend_ab.sh", "run_e123_vllm.sh", "run_seqs_isolation.sh"):
        assert "ulimit -n" in (DAY6 / "gpu" / name).read_text(), name
