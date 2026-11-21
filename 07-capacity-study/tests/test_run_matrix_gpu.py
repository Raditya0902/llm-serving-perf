"""The vLLM matrix runner, driven end to end with stand-in server/bench/nvidia-smi (no GPU, no vLLM)."""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from helpers import make_bench_result
from run_sweep import Backend

import matrix as mx
import run_matrix_gpu as rg

PORT = 8123
METRICS = ('vllm:kv_cache_usage_perc{e="0"} 0.5\nvllm:num_requests_waiting{e="0"} 2\n'
           'vllm:num_preemptions_total{e="0"} 1\n')


@dataclass
class FakeHandle:
    exit_code: int | None = None
    terminated: bool = False
    closed: bool = False

    def poll(self):
        return self.exit_code

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.terminated = True

    def wait(self, timeout=None):
        return 0

    def close(self):
        self.closed = True


@dataclass
class Fake:
    """Records every side effect; each behaviour can be broken per test."""
    server_exit_code: int | None = None
    bench_rc: int = 0
    write_result: bool = True
    port_free: bool = True
    healthy: bool = True
    handles: list = field(default_factory=list)
    serve_cmds: list = field(default_factory=list)
    bench_cmds: list = field(default_factory=list)
    clock: float = 0.0

    def backend(self, case: mx.MatrixCase, rep: int) -> Backend:
        def start_server(cmd, log_path):
            self.serve_cmds.append(cmd)
            handle = FakeHandle(exit_code=self.server_exit_code)
            self.handles.append(handle)
            return handle

        def run_bench(cmd):
            self.bench_cmds.append(cmd)
            if self.write_result:
                out = Path(cmd[cmd.index("--result-dir") + 1])
                (out / cmd[cmd.index("--result-filename") + 1]).write_text(
                    json.dumps(make_bench_result(case, rep)))
            return self.bench_rc

        def sleep(seconds):
            self.clock += seconds

        def monotonic():
            self.clock += 0.1
            return self.clock

        return Backend(start_server, run_bench, lambda url: self.healthy, sleep, monotonic,
                       lambda url: METRICS, lambda port: self.port_free)


CASE = next(c for c in mx.build_matrix() if c.case_id == "B-c16-tuned")
RUN = mx.RunSpec(CASE, 0)


def _deps(fake: Fake, case=CASE, rep=0, query=lambda: "30000\n"):
    return rg.RunnerDeps(fake.backend(case, rep), gpu_query=query, poll_interval_s=0.001, mem_interval_s=0.001)


def _execute(tmp_path, fake, **kw):
    return rg.execute_run(RUN, tmp_path, _deps(fake), port=PORT, startup_timeout_s=10, **kw)


def test_successful_run_writes_result_and_sidecar_and_stops_the_server(tmp_path):
    fake = Fake()
    outcome = _execute(tmp_path, fake)
    assert outcome.ok and not outcome.skipped and outcome.error is None
    assert (tmp_path / "B-c16-tuned__r0.json").exists()
    side = json.loads((tmp_path / "sidecar" / "B-c16-tuned__r0.json").read_text())
    assert side["poll"]["kv_peak_frac"] == 0.5 and side["poll"]["samples"] >= 1
    assert side["gpu_mem"]["peak_mib"] == 30000
    assert fake.handles[0].terminated and fake.handles[0].closed
    assert fake.serve_cmds[0] == CASE.serve_command(PORT)
    assert fake.bench_cmds[0] == CASE.bench_command(str(tmp_path), 0, PORT)


def test_server_that_dies_before_healthy_is_a_failure_and_still_torn_down(tmp_path):
    fake = Fake(server_exit_code=1, healthy=False)
    outcome = _execute(tmp_path, fake)
    assert not outcome.ok and "exited" in outcome.error
    assert fake.handles[0].terminated and not fake.bench_cmds


def test_server_that_never_becomes_healthy_times_out(tmp_path):
    fake = Fake(healthy=False)
    outcome = rg.execute_run(RUN, tmp_path, _deps(fake), port=PORT, startup_timeout_s=1.0)
    assert not outcome.ok and "not healthy" in outcome.error and fake.handles[0].terminated


def test_nonzero_bench_exit_is_a_failure(tmp_path):
    outcome = _execute(tmp_path, Fake(bench_rc=2))
    assert not outcome.ok and "exit code 2" in outcome.error
    assert not (tmp_path / "sidecar" / "B-c16-tuned__r0.json").exists()


def test_bench_timeout_is_a_failure_not_a_crash(tmp_path):
    fake = Fake()
    backend = fake.backend(CASE, 0)

    def hang(cmd):
        raise subprocess.TimeoutExpired(cmd, 1)

    deps = rg.RunnerDeps(Backend(**{**backend.__dict__, "run_bench": hang}),
                         gpu_query=lambda: "1\n", poll_interval_s=0.001, mem_interval_s=0.001)
    outcome = rg.execute_run(RUN, tmp_path, deps, port=PORT, startup_timeout_s=10)
    assert not outcome.ok and fake.handles[0].terminated


def test_missing_result_file_is_a_failure(tmp_path):
    outcome = _execute(tmp_path, Fake(write_result=False))
    assert not outcome.ok and "result file" in outcome.error


def test_busy_port_is_refused_without_starting_a_server(tmp_path):
    fake = Fake(port_free=False)
    outcome = _execute(tmp_path, fake)
    assert not outcome.ok and "in use" in outcome.error and not fake.handles


def test_existing_result_from_another_configuration_is_refused(tmp_path):
    stale = make_bench_result(CASE, 0, matrix_config_id="000000000000")
    (tmp_path / "B-c16-tuned__r0.json").write_text(json.dumps(stale))
    outcome = _execute(tmp_path, Fake())
    assert not outcome.ok and "different configuration" in outcome.error


def test_force_reruns_and_discards_the_stale_sidecar(tmp_path):
    (tmp_path / "B-c16-tuned__r0.json").write_text(json.dumps(make_bench_result(CASE, 0)))
    side = tmp_path / "sidecar"
    side.mkdir()
    (side / "B-c16-tuned__r0.json").write_text('{"poll": {"kv_peak_frac": 0.01}}')
    fake = Fake()
    assert _execute(tmp_path, fake, force=True).ok and fake.handles
    assert json.loads((side / "B-c16-tuned__r0.json").read_text())["poll"]["kv_peak_frac"] == 0.5


def test_unreadable_gpu_query_degrades_to_not_measured(tmp_path):
    def broken():
        raise OSError("no nvidia-smi")

    fake = Fake()
    outcome = rg.execute_run(RUN, tmp_path, _deps(fake, query=broken), port=PORT, startup_timeout_s=10)
    assert outcome.ok
    side = json.loads((tmp_path / "sidecar" / "B-c16-tuned__r0.json").read_text())
    assert side["gpu_mem"]["peak_mib"] is None and side["gpu_mem"]["errors"] >= 1


# ------------------------------------------------------------------ ordering and CLI
def test_ordered_runs_keeps_every_run_once_and_rotates_each_round():
    runs = mx.build_runs(reps=3, key_reps=3)
    ordered = rg.ordered_runs(runs)
    assert sorted(r.run_id for r in ordered) == sorted(r.run_id for r in runs)
    firsts = [next(r for r in ordered if r.rep == rep) for rep in range(3)]
    assert len({f.case.case_id for f in firsts}) == 3  # position bias: a different case leads each round


def test_ordered_runs_finishes_one_round_before_starting_the_next():
    ordered = rg.ordered_runs(mx.build_runs(reps=2, key_reps=2))
    reps = [r.rep for r in ordered]
    assert reps == sorted(reps)


def test_dry_run_prints_every_command_and_touches_nothing(tmp_path, capsys):
    assert rg.main(["--dry-run", "--reps", "1", "--key-reps", "1", "--out", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert out.count("vllm serve") == 13  # 12 cases + the A/A control
    assert out.count("vllm bench serve") == 13
    assert "--max-concurrency 64" in out and not list(tmp_path.iterdir())


def test_only_filter_selects_cases(tmp_path, capsys):
    assert rg.main(["--dry-run", "--only", "A-c4-baseline", "--reps", "1", "--key-reps", "1",
                    "--no-control", "--out", str(tmp_path)]) == 0
    assert capsys.readouterr().out.count("vllm serve") == 1


def test_only_filter_rejects_unknown_case_ids(tmp_path):
    with pytest.raises(SystemExit):
        rg.main(["--dry-run", "--only", "Z-c9-nope", "--out", str(tmp_path)])


def test_run_all_reports_failure_in_exit_code(tmp_path):
    fake = Fake(port_free=False)
    deps = _deps(fake)
    outcomes = rg.run_all([RUN], tmp_path, deps, port=PORT, startup_timeout_s=1, force=False)
    assert [o.ok for o in outcomes] == [False]
    assert rg.exit_code(outcomes) == rg.EXIT_RUN_FAILED
    assert rg.exit_code([rg.Outcome("x", True, False, None)]) == rg.EXIT_OK


# ---------------------------------------------------------------- review/security hardening
def test_runner_output_passes_ingest_end_to_end(tmp_path):
    import ingest_gpu as ig
    assert _execute(tmp_path, Fake()).ok
    row = ig.load_run(tmp_path / "B-c16-tuned__r0.json")
    assert row["case_id"] == "B-c16-tuned" and row["kv_peak_frac"] == 0.5
    assert row["peak_gpu_mem_gib"] == pytest.approx(30000 / 1024) and row["preemptions"] == 0.0


def _complete_run(tmp_path, **result_over):
    from helpers import write_bench_result, write_sidecar
    write_bench_result(tmp_path, CASE, 0, **result_over)
    write_sidecar(tmp_path, CASE, 0)


def test_complete_existing_run_is_skipped(tmp_path):
    _complete_run(tmp_path)
    fake = Fake()
    outcome = _execute(tmp_path, fake)
    assert outcome.ok and outcome.skipped and not fake.handles


@pytest.mark.parametrize("over", [{"failed": 2}, {"completed": 5}])
def test_failed_or_partial_existing_result_is_rerun_not_skipped(tmp_path, over):
    _complete_run(tmp_path, **over)
    fake = Fake()
    outcome = _execute(tmp_path, fake)
    assert outcome.ok and not outcome.skipped and fake.handles


def test_existing_result_without_a_sidecar_is_rerun(tmp_path):
    from helpers import write_bench_result
    write_bench_result(tmp_path, CASE, 0)
    fake = Fake()
    outcome = _execute(tmp_path, fake)
    assert outcome.ok and not outcome.skipped and (tmp_path / "sidecar" / "B-c16-tuned__r0.json").exists()


def test_failed_bench_does_not_leave_a_result_that_resume_would_trust(tmp_path):
    outcome = _execute(tmp_path, Fake(bench_rc=2))
    assert not outcome.ok and not (tmp_path / "B-c16-tuned__r0.json").exists()


def test_server_replaced_after_health_is_detected(tmp_path):
    fake = Fake()
    backend = fake.backend(CASE, 0)

    class Flaky(FakeHandle):
        calls = 0

        def poll(self):
            Flaky.calls += 1
            return None if Flaky.calls == 1 else 1  # alive at the health wait, gone right after

    handle = Flaky()
    deps = rg.RunnerDeps(Backend(**{**backend.__dict__, "start_server": lambda cmd, log: handle}),
                         gpu_query=lambda: "1\n", poll_interval_s=0.001, mem_interval_s=0.001)
    outcome = rg.execute_run(RUN, tmp_path, deps, port=PORT, startup_timeout_s=10)
    assert not outcome.ok and "exited" in outcome.error and not fake.bench_cmds


def test_ordering_is_shuffled_so_tuned_does_not_always_follow_baseline():
    ordered = rg.ordered_runs(mx.build_runs(reps=3, key_reps=3, include_control=False))
    assert sorted(r.run_id for r in ordered) == sorted(r.run_id for r in mx.build_runs(3, 3, include_control=False))
    tuned_first = both = 0
    for rep in range(3):
        position = {r.case.case_id: i for i, r in enumerate(x for x in ordered if x.rep == rep)}
        for w in "AB":
            for c in mx.CONCURRENCY_LEVELS:
                both += 1
                tuned_first += position[f"{w}-c{c}-tuned"] < position[f"{w}-c{c}-baseline"]
    assert 4 <= tuned_first <= both - 4


def test_ordering_is_deterministic():
    runs = mx.build_runs(reps=2, key_reps=2)
    assert [r.run_id for r in rg.ordered_runs(runs)] == [r.run_id for r in rg.ordered_runs(runs)]


def test_loopback_bypasses_proxies_without_dropping_existing_entries():
    env = {"NO_PROXY": "corp.internal", "HTTP_PROXY": "http://proxy:3128"}
    out = rg.bypass_proxy_for_loopback(env)
    assert out["HTTP_PROXY"] == "http://proxy:3128"  # the server may still need it to download weights
    for key in ("NO_PROXY", "no_proxy"):
        assert {"corp.internal", "127.0.0.1", "localhost", "::1"} <= set(out[key].split(","))
    assert env == {"NO_PROXY": "corp.internal", "HTTP_PROXY": "http://proxy:3128"}  # input untouched


@pytest.mark.parametrize("argv", [["--port", "70000"], ["--port", "0"], ["--reps", "100000"],
                                  ["--gpu-index", "-1"]])
def test_cli_rejects_out_of_range_numbers(tmp_path, argv):
    with pytest.raises(SystemExit):
        rg.main(["--dry-run", "--out", str(tmp_path), *argv])
