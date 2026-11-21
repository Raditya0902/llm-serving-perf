import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "gpu"))

from conftest import SYNTHETIC_NOTE  # noqa: E402
from gpu_mem_sampler import parse_memory_used, sample_peak  # noqa: E402
from report_from_vllm import per_request_tpots, row_from_result  # noqa: E402

GPU_DIR = Path(__file__).resolve().parent.parent / "gpu"


def test_parse_memory_used_reads_one_value_per_gpu():
    assert parse_memory_used("1024\n2048\n") == [1024, 2048]


@pytest.mark.parametrize("text", ["", "   \n", "N/A\n"])
def test_parse_memory_used_rejects_unusable_output(text):
    with pytest.raises(ValueError):
        parse_memory_used(text)


def test_sample_peak_takes_the_max_and_samples_once_after_exit():
    readings = iter(["100\n", "300\n", "200\n", "150\n"])
    running = iter([True, True, True, False])

    peak = sample_peak(lambda: next(readings), lambda: next(running), sleep=lambda _s: None)

    assert (peak.peak_mib, peak.n_samples) == (300, 4)


def test_sample_peak_rejects_missing_gpu_index():
    with pytest.raises(ValueError, match="gpu_index"):
        sample_peak(lambda: "1\n", lambda: False, gpu_index=3, sleep=lambda _s: None)


def test_per_request_tpots_skips_requests_without_decode_tokens():
    assert per_request_tpots([[0.01, 0.03], [], [0.02]]) == pytest.approx([0.02, 0.02])


def _detailed(tmp_path) -> Path:
    path = tmp_path / "baseline__r0.json"
    path.write_text(json.dumps({
        "synthetic_note": SYNTHETIC_NOTE, "sweep_run_id": "baseline",
        "request_throughput": 2.0, "output_throughput": 250.0,
        "ttfts": [0.010, 0.020, 0.030, 0.040, 0.100],
        "itls": [[0.005, 0.007], [0.010], [0.006, 0.006], [0.008], [0.020]],
    }))
    return path


def test_row_from_result_prefers_vllms_own_p95_keys(tmp_path):
    path = tmp_path / "r.json"
    path.write_text(json.dumps({
        "synthetic_note": SYNTHETIC_NOTE, "request_throughput": 3.0, "output_throughput": 90.0,
        "p95_ttft_ms": 120.0, "p95_tpot_ms": 15.0, "ttfts": [9.0],
    }))

    row = row_from_result(path)

    assert (row.ttft_ms_p95, row.tpot_ms_p95) == (120.0, 15.0)


def test_row_from_result_falls_back_to_detailed_arrays(tmp_path):
    peak = tmp_path / "mem.json"
    peak.write_text(json.dumps({"peak_mib": 20480}))

    row = row_from_result(_detailed(tmp_path), peak)

    assert row.run_id == "baseline"
    assert row.ttft_ms_p95 == pytest.approx(88.0)  # linear interpolation over [10,20,30,40,100] ms
    assert row.tpot_ms_p95 == pytest.approx(18.0)  # per-request means [6,10,6,8,20] ms
    assert row.peak_mem_mb == 20480


def test_row_from_result_has_no_memory_when_not_measured(tmp_path):
    assert row_from_result(_detailed(tmp_path)).peak_mem_mb is None


def test_row_from_result_names_the_missing_flag_when_p95_is_unavailable(tmp_path):
    path = tmp_path / "day4_style.json"
    path.write_text(json.dumps({
        "request_throughput": 2.0, "output_throughput": 250.0, "p99_ttft_ms": 80.0, "p50_ttft_ms": 20.0,
    }))

    with pytest.raises(ValueError, match="metric-percentiles 50,95,99"):
        row_from_result(path)


def test_row_from_result_rejects_files_without_vllm_keys(tmp_path):
    bad = tmp_path / "x.json"
    bad.write_text("{}")
    with pytest.raises(ValueError, match="missing key"):
        row_from_result(bad)


@pytest.mark.skipif(sys.platform == "linux", reason="refusal path is for non-Linux hosts")
def test_run_nsys_refuses_to_run_off_linux():
    result = subprocess.run(["bash", str(GPU_DIR / "run_nsys.sh")], capture_output=True, text=True)

    assert result.returncode == 2
    assert "refusing" in result.stderr


def test_sampler_terminates_the_benchmark_when_sampling_fails(tmp_path, monkeypatch):
    import gpu_mem_sampler

    children = []
    real_popen = subprocess.Popen

    def spy(*args, **kwargs):
        children.append(real_popen(*args, **kwargs))
        return children[-1]

    def failing_smi() -> str:
        raise subprocess.TimeoutExpired("nvidia-smi", 10)

    monkeypatch.setattr(gpu_mem_sampler.subprocess, "Popen", spy)
    monkeypatch.setattr(gpu_mem_sampler, "_nvidia_smi", failing_smi)

    with pytest.raises(subprocess.TimeoutExpired):
        gpu_mem_sampler.main(["--out", str(tmp_path / "m.json"), "--",
                              sys.executable, "-c", "import time; time.sleep(60)"])

    assert children[0].poll() is not None  # the child was killed, not orphaned
    assert not (tmp_path / "m.json").exists()


def test_sampler_rejects_non_positive_interval(tmp_path):
    import gpu_mem_sampler

    with pytest.raises(SystemExit):
        gpu_mem_sampler.main(["--out", str(tmp_path / "m.json"), "--interval", "0", "--", "true"])


def test_row_from_result_rejects_memory_from_a_failed_benchmark(tmp_path):
    result = tmp_path / "r.json"
    result.write_text(json.dumps({"request_throughput": 1.0, "output_throughput": 1.0,
                                  "p95_ttft_ms": 1.0, "p95_tpot_ms": 1.0}))
    mem = tmp_path / "mem.json"
    mem.write_text(json.dumps({"peak_mib": 9000, "returncode": 137}))

    with pytest.raises(ValueError, match="exited with code 137"):
        row_from_result(result, mem)


@pytest.mark.parametrize("script", sorted(GPU_DIR.glob("*.sh")), ids=lambda p: p.name)
def test_every_gpu_shell_script_passes_a_bash_syntax_check(script):
    result = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_seqs_isolation_script_writes_the_file_names_the_summary_reads():
    import re

    from gpu import seqs_isolation

    text = (GPU_DIR / "run_seqs_isolation.sh").read_text()
    arms = re.search(r'ARMS="\$\{ARMS:-([^}]*)\}"', text).group(1).split()
    assert len(arms) == 9 and "1024:512x128" in arms and "512:512x128" in arms
    for arm in arms:
        seqs, workload = arm.split(":")
        assert seqs_isolation.parse_name(f"seqs{seqs}_{workload}_r0.json") == (int(seqs), workload, 0)
    assert 'tag="seqs${seqs}_${workload}_r${rep}"' in text
