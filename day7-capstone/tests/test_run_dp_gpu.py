"""Data-parallel baseline runner (two TP=1 replicas), driven with stand-in server/bench/nvidia-smi."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from helpers import make_bench_result
from run_sweep import Backend
from test_run_matrix_gpu import METRICS, FakeHandle

import matrix as mx
import run_dp_gpu as dp
import run_matrix_gpu as rg
import summarize_followup as sf

PORT = 18000


def _cases(total=64, cap=False, hardware="a100-40gb"):
    return dp.replica_cases("m", "llama3-8b", hardware, cap, total)


def test_replica_cases_split_concurrency_and_use_different_prompts():
    a, b = _cases(64)
    assert (a.case_id, b.case_id) == ("B-c32-dp2a", "B-c32-dp2b")
    assert a.concurrency == b.concurrency == 32 and a.num_prompts == b.num_prompts == 256
    assert a.seed_for(0) != b.seed_for(0)
    assert (a.config.gpu_memory_utilization, a.config.max_num_batched_tokens) == (0.85, 8192)
    assert a.config_id != b.config_id  # different config names and seeds: never mixed up on resume


@pytest.mark.parametrize("total", [1, 3, 0])
def test_replica_cases_reject_totals_that_cannot_be_split(total):
    with pytest.raises(ValueError, match="even"):
        _cases(total)


def test_default_seed_offset_keeps_existing_config_fingerprints():
    plain = {c.case_id: c.config_id for c in mx.build_matrix()}
    capped = {c.case_id: c.config_id for c in mx.build_matrix(cap_kv_pool=True)}
    assert (plain["B-c64-baseline"], plain["A-c4-tuned"]) == ("b18665b377b5", "8a0706e50e98")
    assert capped["B-c64-baseline"] == "31e24c797a8c"  # recorded before seed_offset existed
    assert mx.build_matrix()[0].seed_for(2) == mx.BASE_SEED + 2


def test_capped_replicas_each_get_the_single_gpu_pool():
    a, b = _cases(64, cap=True)
    single = next(c for c in mx.build_matrix(cap_kv_pool=True) if c.case_id == "B-c64-baseline")
    assert a.kv_cache_cap_bytes == b.kv_cache_cap_bytes == single.kv_cache_cap_bytes


def test_serve_command_pins_one_gpu_per_replica():
    a, b = _cases()
    cmd = dp.pinned_serve_command(b, PORT + 1, gpu=1)
    assert cmd[:2] == ["env", "CUDA_VISIBLE_DEVICES=1"] and cmd[2:] == b.serve_command(PORT + 1)
    assert "--tensor-parallel-size" not in cmd


@dataclass
class FakeDp:
    bench_rc: int = 0
    port_free: bool = True
    handles: list = field(default_factory=list)
    serve_cmds: list = field(default_factory=list)
    bench_cmds: list = field(default_factory=list)
    clock: float = 0.0

    def backend(self, cases) -> Backend:
        by_name = {f"{c.case_id}__r0.json": c for c in cases}

        def start_server(cmd, log_path):
            self.serve_cmds.append(cmd)
            self.handles.append(FakeHandle())
            return self.handles[-1]

        def run_bench(cmd):
            self.bench_cmds.append(cmd)
            name = cmd[cmd.index("--result-filename") + 1]
            out = Path(cmd[cmd.index("--result-dir") + 1])
            (out / name).write_text(json.dumps(make_bench_result(by_name[name], 0)))
            return self.bench_rc

        def monotonic():
            self.clock += 0.1
            return self.clock

        return Backend(start_server, run_bench, lambda url: True, lambda s: None, monotonic,
                       lambda url: METRICS, lambda port: self.port_free)


def _run(tmp_path, fake, cases):
    deps = rg.RunnerDeps(fake.backend(cases), gpu_query=lambda: "30000\n30000\n",
                         poll_interval_s=0.001, mem_interval_s=0.001)
    return dp.execute_dp_run(cases, 0, tmp_path, deps, port=PORT, startup_timeout_s=10, force=False)


def test_dp_run_starts_two_pinned_servers_and_benches_both_then_tears_down(tmp_path):
    cases = _cases()
    fake = FakeDp()
    outcome = _run(tmp_path, fake, cases)
    assert outcome.ok and not outcome.skipped
    assert [c[1] for c in fake.serve_cmds] == ["CUDA_VISIBLE_DEVICES=0", "CUDA_VISIBLE_DEVICES=1"]
    assert [c[c.index("--port") + 1] for c in fake.serve_cmds] == [str(PORT), str(PORT + 1)]
    assert sorted(c[c.index("--port") + 1] for c in fake.bench_cmds) == [str(PORT), str(PORT + 1)]
    for case in cases:
        assert (tmp_path / f"{case.case_id}__r0.json").exists()
        side = json.loads((tmp_path / "sidecar" / f"{case.case_id}__r0.json").read_text())
        assert side["poll"]["kv_peak_frac"] == 0.5 and side["gpu_mem"]["peak_mib"] == 30000
    assert all(h.terminated and h.closed for h in fake.handles)


def test_a_failed_bench_fails_the_run_and_leaves_no_results(tmp_path):
    cases = _cases()
    fake = FakeDp(bench_rc=1)
    outcome = _run(tmp_path, fake, cases)
    assert not outcome.ok and "exit code" in outcome.error
    assert not list(tmp_path.glob("*.json")) and not list((tmp_path / "sidecar").glob("*.json"))
    assert all(h.terminated for h in fake.handles)


def test_busy_port_is_refused_before_anything_starts(tmp_path):
    fake = FakeDp(port_free=False)
    outcome = _run(tmp_path, fake, _cases())
    assert not outcome.ok and "already in use" in outcome.error and not fake.serve_cmds


def test_complete_results_are_skipped_on_resume(tmp_path):
    cases = _cases()
    assert _run(tmp_path, FakeDp(), cases).ok
    again = FakeDp()
    outcome = _run(tmp_path, again, cases)
    assert outcome.ok and outcome.skipped and not again.serve_cmds


def test_combined_dp_row_divides_all_tokens_by_the_longer_duration(tmp_path):
    a, b = _cases()
    for case, tokens, dur in ((a, 20000, 40.0), (b, 18000, 36.0)):
        path = tmp_path / f"{case.case_id}__r0.json"
        path.write_text(json.dumps(make_bench_result(
            case, 0, total_output_tokens=tokens, duration=dur, output_throughput=tokens / dur)))
        side = tmp_path / "sidecar" / path.name
        side.parent.mkdir(exist_ok=True)
        side.write_text(json.dumps({"poll": {"kv_peak_frac": 0.9, "preemptions": 3.0, "waiting_peak": 4.0}}))
    row = sf.load_dp_runs(tmp_path, "B-c32-dp2a", "B-c32-dp2b")
    assert len(row) == 1
    assert row[0]["output_throughput"] == pytest.approx((20000 + 18000) / 40.0)
    assert row[0]["preempt"] == 6.0 and row[0]["kv_peak"] == 0.9


def test_summary_table_has_a_combined_dp_row_compared_with_tp1(tmp_path):
    a, b = _cases()
    tp1 = mx.MatrixCase(mx.WORKLOADS["B"], mx.ServingConfig("tp1", 0.85, 8192), 64)
    for case in (a, b, tp1):
        path = tmp_path / f"{case.case_id}__r0.json"
        path.write_text(json.dumps(make_bench_result(case, 0, total_output_tokens=20000, duration=40.0)))
        (tmp_path / "sidecar").mkdir(exist_ok=True)
        (tmp_path / "sidecar" / path.name).write_text(json.dumps(
            {"poll": {"kv_peak_frac": 0.5, "preemptions": 0.0, "waiting_peak": 1.0}}))
    table = sf.summarize([tmp_path])
    assert table["B-c64-dp2"]["output_throughput"] == pytest.approx(1000.0)  # 40000 tokens / 40 s
    assert "B-c64-dp2" in sf.render(table)


def test_dry_run_prints_pinned_commands_for_both_replicas_without_starting_anything(tmp_path, capsys):
    assert dp.main(["--out", str(tmp_path), "--dry-run", "--totals", "64", "--model", "m/x"]) == 0
    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("env ")]
    assert len(lines) == 2 and "CUDA_VISIBLE_DEVICES=0" in lines[0] and "--port 18001" in lines[1]
