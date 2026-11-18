"""summarize_tp_compare.py: aggregate run_tp_compare.sh output (synthetic files, no GPU)."""
from __future__ import annotations

import json
from pathlib import Path

import summarize_tp_compare as st


def _bench(path: Path, tput: float, tpot: float, ttft: float = 10.0, failed: int = 0) -> None:
    path.write_text(json.dumps({"output_throughput": tput, "mean_tpot_ms": tpot, "mean_ttft_ms": ttft,
                                "p99_tpot_ms": tpot * 2, "failed": failed}))


def _mem(path: Path, used_gpu0: int, used_gpu1: int) -> None:
    path.write_text("timestamp, index, memory.used [MiB], utilization.gpu [%]\n"
                    f"t, 0, {used_gpu0} MiB, 10 %\nt, 1, {used_gpu1} MiB, 10 %\nt, 0, 100 MiB, 0 %\n")


def test_groups_repeats_and_computes_tp2_vs_tp1_delta(tmp_path):
    for rep, (a, b) in enumerate([(100.0, 150.0), (102.0, 153.0)]):
        _bench(tmp_path / f"tp1_c8_r{rep}.json", a, 8.0)
        _bench(tmp_path / f"tp2_c8_r{rep}.json", b, 5.0)
    _mem(tmp_path / "mem_tp2_c8_r0.csv", 30000, 29000)
    rows = st.summarize(tmp_path)
    tp1, tp2 = rows["tp1_c8"], rows["tp2_c8"]
    assert tp1["n"] == 2 and round(tp1["tput"], 1) == 101.0
    assert round(st.delta(tp2["tput"], tp1["tput"]), 1) == 50.0
    assert tp2["mem_gpu0"] == 30000 and tp2["mem_gpu1"] == 29000


def test_data_parallel_throughput_is_the_sum_of_the_two_replicas(tmp_path):
    _bench(tmp_path / "dp2_c32_r0_a.json", 400.0, 7.0)
    _bench(tmp_path / "dp2_c32_r0_b.json", 410.0, 7.2)
    assert st.summarize(tmp_path)["dp2_c32"]["tput"] == 810.0


def test_failed_requests_are_surfaced_not_hidden(tmp_path):
    _bench(tmp_path / "tp1_c1_r0.json", 100.0, 4.0, failed=3)
    assert st.summarize(tmp_path)["tp1_c1"]["failed"] == 3
    assert "3" in st.render(st.summarize(tmp_path))
