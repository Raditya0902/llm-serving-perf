from __future__ import annotations

import http.client
import json
import signal
import subprocess
from pathlib import Path

import pytest

from run_sweep import (
    Backend,
    RunOutcome,
    ServerStartError,
    _interrupt_on_signals,
    execute_run,
    main,
    run_sweep,
    stop_server,
    wait_until_healthy,
)
from sweep_config import build_sweep


class FakeServer:
    def __init__(self, exit_code=None, hang_on_terminate=False, never_dies=False):
        self.exit_code = exit_code
        self.hang_on_terminate = hang_on_terminate
        self.never_dies = never_dies
        self.closed = False
        self.calls: list[str] = []

    def close(self):
        self.closed = True

    def poll(self):
        return self.exit_code

    def terminate(self):
        self.calls.append("terminate")
        if not self.hang_on_terminate:
            self.exit_code = 0

    def kill(self):
        self.calls.append("kill")
        if not self.never_dies:
            self.exit_code = -9

    def wait(self, timeout=None):
        if self.exit_code is None:
            raise subprocess.TimeoutExpired("vllm", timeout)
        return self.exit_code


class Harness:
    """Records every backend interaction in order."""

    def __init__(self, *, healthy_after=0, server=None, bench_rc=0, bench_exc=None,
                 write_result=True, metrics_text="# no metrics\n", metrics_exc=None,
                 port_free=True):
        self.port_free = port_free
        self.metrics_text = metrics_text
        self.metrics_exc = metrics_exc
        self.events: list[str] = []
        self.servers: list[FakeServer] = []
        self.healthy_after = healthy_after
        self._server_proto = server
        self.bench_rc = bench_rc
        self.bench_exc = bench_exc
        self.write_result = write_result
        self.now = 0.0
        self.health_calls = 0

    def backend(self) -> Backend:
        return Backend(
            start_server=self._start, run_bench=self._bench, is_healthy=self._healthy,
            sleep=self._sleep, monotonic=lambda: self.now, fetch_text=self._fetch,
            is_port_free=lambda port: self.port_free,
        )

    def _fetch(self, url):
        self.events.append("metrics")
        if self.metrics_exc:
            raise self.metrics_exc
        return self.metrics_text

    def _start(self, cmd, log_path):
        self.events.append("start")
        self.servers.append(self._server_proto or FakeServer())
        return self.servers[-1]

    def _healthy(self, url):
        self.health_calls += 1
        self.events.append("health")
        return self.health_calls > self.healthy_after

    def _sleep(self, seconds):
        self.now += seconds

    def _bench(self, cmd):
        self.events.append("bench")
        if self.bench_exc:
            raise self.bench_exc
        if self.write_result and self.bench_rc == 0:
            out = Path(cmd[cmd.index("--result-dir") + 1])
            out.mkdir(parents=True, exist_ok=True)
            meta = dict(m.split("=", 1) for m in cmd[cmd.index("--metadata") + 1:])
            (out / cmd[cmd.index("--result-filename") + 1]).write_text(
                json.dumps({"sweep_config_id": meta["sweep_config_id"]}))
        return self.bench_rc


@pytest.fixture
def baseline():
    return build_sweep()[0]


def test_wait_until_healthy_polls_until_ready(baseline):
    h = Harness(healthy_after=3)

    wait_until_healthy(h.backend(), FakeServer(), baseline.base_url, timeout_s=60, interval_s=1)

    assert h.health_calls == 4
    assert h.now == 3  # slept between the three failed polls


def test_wait_until_healthy_times_out(baseline):
    h = Harness(healthy_after=10**6)

    with pytest.raises(TimeoutError, match="not healthy"):
        wait_until_healthy(h.backend(), FakeServer(), baseline.base_url, timeout_s=5, interval_s=1)


def test_wait_until_healthy_detects_dead_server(baseline):
    h = Harness(healthy_after=10**6)

    with pytest.raises(ServerStartError, match="exited with code 1"):
        wait_until_healthy(h.backend(), FakeServer(exit_code=1), baseline.base_url,
                           timeout_s=60, interval_s=1)


def test_stop_server_terminates_gracefully():
    server = FakeServer()

    stop_server(server, grace_s=1)

    assert server.calls == ["terminate"]


def test_stop_server_kills_when_terminate_hangs():
    server = FakeServer(hang_on_terminate=True)

    stop_server(server, grace_s=1)

    assert server.calls == ["terminate", "kill"]


def test_stop_server_still_signals_the_group_when_the_leader_already_exited():
    # vllm's engine/worker children can outlive the API-server leader and hold the GPU.
    server = FakeServer(exit_code=0)

    stop_server(server, grace_s=1)

    assert server.calls == ["terminate"]


def test_execute_run_happy_path_orders_start_health_bench_stop(baseline, tmp_path):
    h = Harness(healthy_after=1)

    outcome = execute_run(baseline, 0, tmp_path, h.backend(), startup_timeout_s=60)

    assert outcome.ok and outcome.error is None
    assert h.events == ["start", "health", "health", "bench", "metrics"]
    assert h.servers[0].calls == ["terminate"]
    assert outcome.result_path == tmp_path / "baseline__r0.json"
    assert outcome.result_path.exists()


def test_execute_run_saves_server_metrics_after_the_bench(baseline, tmp_path):
    text = 'vllm:num_preemptions_total{engine="0"} 7.0\n'
    h = Harness(metrics_text=text)

    execute_run(baseline, 0, tmp_path, h.backend(), startup_timeout_s=60)

    saved = tmp_path / "metrics" / "baseline__r0.metrics.txt"
    assert saved.read_text() == text


def test_metrics_failure_is_logged_but_does_not_fail_the_run(baseline, tmp_path, caplog):
    h = Harness(metrics_exc=OSError("connection refused"))

    with caplog.at_level("WARNING", logger="run_sweep"):
        outcome = execute_run(baseline, 0, tmp_path, h.backend(), startup_timeout_s=60)

    assert outcome.ok
    assert "connection refused" in caplog.text
    assert not (tmp_path / "metrics").exists()


def test_metrics_are_not_fetched_when_the_bench_fails(baseline, tmp_path):
    h = Harness(bench_rc=2)

    execute_run(baseline, 0, tmp_path, h.backend(), startup_timeout_s=60)

    assert "metrics" not in h.events


def test_execute_run_stops_server_when_bench_fails(baseline, tmp_path):
    h = Harness(bench_rc=2)

    outcome = execute_run(baseline, 0, tmp_path, h.backend(), startup_timeout_s=60)

    assert not outcome.ok
    assert "exit code 2" in outcome.error
    assert h.servers[0].calls == ["terminate"]


def test_execute_run_stops_server_when_bench_binary_is_missing(baseline, tmp_path):
    h = Harness(bench_exc=FileNotFoundError("vllm"))

    outcome = execute_run(baseline, 0, tmp_path, h.backend(), startup_timeout_s=60)

    assert not outcome.ok and "vllm" in outcome.error
    assert h.servers[0].calls == ["terminate"]


def test_execute_run_stops_server_on_keyboard_interrupt(baseline, tmp_path):
    h = Harness(bench_exc=KeyboardInterrupt())

    with pytest.raises(KeyboardInterrupt):
        execute_run(baseline, 0, tmp_path, h.backend(), startup_timeout_s=60)

    assert h.servers[0].calls == ["terminate"]


def test_execute_run_reports_startup_crash_without_running_bench(baseline, tmp_path):
    h = Harness(healthy_after=10**6, server=FakeServer(exit_code=3))

    outcome = execute_run(baseline, 0, tmp_path, h.backend(), startup_timeout_s=60)

    assert not outcome.ok and "exited with code 3" in outcome.error
    assert "bench" not in h.events


def test_execute_run_reports_startup_timeout(baseline, tmp_path):
    h = Harness(healthy_after=10**6)

    outcome = execute_run(baseline, 0, tmp_path, h.backend(), startup_timeout_s=3)

    assert not outcome.ok and "not healthy" in outcome.error
    assert h.servers[0].calls == ["terminate"]


def test_execute_run_flags_missing_result_file(baseline, tmp_path):
    h = Harness(write_result=False)

    outcome = execute_run(baseline, 0, tmp_path, h.backend(), startup_timeout_s=60)

    assert not outcome.ok and "result file" in outcome.error


def test_run_sweep_continues_after_a_failure_and_repeats(tmp_path):
    runs = build_sweep(only=["max_num_seqs"])[:2]
    h = Harness(bench_rc=1)

    outcomes = run_sweep(runs, tmp_path, h.backend(), repeat=2, startup_timeout_s=60)

    assert [o.ok for o in outcomes] == [False] * 4
    assert [(o.run_id, o.rep) for o in outcomes] == [
        (runs[0].run_id, 0), (runs[0].run_id, 1), (runs[1].run_id, 0), (runs[1].run_id, 1)]


def test_run_sweep_skips_existing_results_unless_forced(baseline, tmp_path):
    h = Harness()
    run_sweep([baseline], tmp_path, h.backend(), repeat=1, startup_timeout_s=60)
    h.events.clear()

    again = run_sweep([baseline], tmp_path, h.backend(), repeat=1, startup_timeout_s=60)
    forced = run_sweep([baseline], tmp_path, h.backend(), repeat=1, startup_timeout_s=60,
                       force=True)

    assert again[0].skipped and again[0].ok
    assert not forced[0].skipped
    assert h.events.count("start") == 1  # only the forced run started a server


def test_run_sweep_rejects_bad_repeat(baseline, tmp_path):
    with pytest.raises(ValueError, match="repeat"):
        run_sweep([baseline], tmp_path, Harness().backend(), repeat=0, startup_timeout_s=1)


def test_main_dry_run_prints_commands_and_never_touches_backend(capsys):
    def boom():
        raise AssertionError("backend must not be built for --dry-run")

    code = main(["--dry-run", "--only", "max_num_seqs"], backend_factory=boom)

    out = capsys.readouterr().out
    assert code == 0
    assert "vllm serve Qwen/Qwen2.5-0.5B-Instruct" in out
    assert "vllm bench serve" in out
    assert "--max-num-seqs 16" in out
    assert "baseline" in out


def test_main_returns_one_when_any_run_fails(tmp_path):
    h = Harness(bench_rc=1)

    code = main(["--out", str(tmp_path), "--only", "max_num_seqs"],
                backend_factory=h.backend)

    assert code == 1


def test_main_returns_zero_on_success(tmp_path):
    h = Harness()

    code = main(["--out", str(tmp_path), "--only", "gpu_memory_utilization"],
                backend_factory=h.backend)

    assert code == 0
    assert (tmp_path / "baseline__r0.json").exists()


def test_main_rejects_unknown_knob(capsys):
    code = main(["--dry-run", "--only", "bogus"])

    assert code == 2
    assert "unknown knob" in capsys.readouterr().err


def test_run_outcome_is_immutable():
    outcome = RunOutcome("a", 0, True, None, None, False)
    with pytest.raises(AttributeError):
        outcome.ok = False  # type: ignore[misc]


# --- review fixes -------------------------------------------------------------------------
def test_stop_server_always_closes_the_handle():
    for server in (FakeServer(), FakeServer(exit_code=0), FakeServer(hang_on_terminate=True)):
        stop_server(server, grace_s=1)
        assert server.closed


def test_stop_server_reports_but_survives_a_server_that_cannot_be_killed(caplog):
    server = FakeServer(hang_on_terminate=True, never_dies=True)

    with caplog.at_level("ERROR", logger="run_sweep"):
        stop_server(server, grace_s=0.01)

    assert server.calls == ["terminate", "kill"]
    assert server.closed
    assert "may still hold GPU memory" in caplog.text


def test_busy_port_is_refused_before_any_server_is_started(baseline, tmp_path):
    h = Harness(port_free=False)

    outcome = execute_run(baseline, 0, tmp_path, h.backend(), startup_timeout_s=60)

    assert not outcome.ok and "already in use" in outcome.error
    assert h.events == []  # no server started, no bench run against a foreign server


def test_stale_result_cannot_satisfy_a_run_whose_bench_wrote_nothing(baseline, tmp_path):
    (tmp_path / "baseline__r0.json").write_text('{"sweep_config_id": "old"}')
    h = Harness(write_result=False)

    outcome = execute_run(baseline, 0, tmp_path, h.backend(), startup_timeout_s=60)

    assert not outcome.ok and "was not written" in outcome.error
    assert not (tmp_path / "baseline__r0.json").exists()


def test_stale_metrics_are_not_attributed_to_a_run_whose_scrape_failed(baseline, tmp_path):
    stale = tmp_path / "metrics" / "baseline__r0.metrics.txt"
    stale.parent.mkdir()
    stale.write_text("vllm:num_preemptions_total 99\n")
    h = Harness(metrics_exc=OSError("refused"))

    outcome = execute_run(baseline, 0, tmp_path, h.backend(), startup_timeout_s=60)

    assert outcome.ok
    assert not stale.exists()


def test_http_protocol_errors_while_scraping_do_not_abort_the_sweep(baseline, tmp_path, caplog):
    h = Harness(metrics_exc=http.client.IncompleteRead(b"partial"))

    with caplog.at_level("WARNING", logger="run_sweep"):
        outcome = execute_run(baseline, 0, tmp_path, h.backend(), startup_timeout_s=60)

    assert outcome.ok and "could not fetch /metrics" in caplog.text


def test_resume_refuses_a_result_from_a_different_configuration(tmp_path):
    brief = build_sweep(load="brief")[0]
    saturated = build_sweep(load="saturated")[0]
    h = Harness()
    run_sweep([brief], tmp_path, h.backend(), repeat=1, startup_timeout_s=60)
    h.events.clear()

    outcomes = run_sweep([saturated], tmp_path, h.backend(), repeat=1, startup_timeout_s=60)

    assert not outcomes[0].ok and not outcomes[0].skipped
    assert "different configuration" in outcomes[0].error
    assert h.events == []  # nothing ran, nothing overwritten
    assert json.loads((tmp_path / "baseline__r0.json").read_text())["sweep_config_id"] == brief.config_id


def test_resume_refuses_unreadable_or_untagged_existing_results(baseline, tmp_path):
    path = tmp_path / "baseline__r0.json"
    for content, needle in (("{not json", "unreadable"), ("{}", "different configuration")):
        path.write_text(content)

        outcome = run_sweep([baseline], tmp_path, Harness().backend(), repeat=1,
                            startup_timeout_s=60)[0]

        assert not outcome.ok and needle in outcome.error


def test_force_overwrites_a_result_from_a_different_configuration(tmp_path):
    h = Harness()
    run_sweep([build_sweep(load="brief")[0]], tmp_path, h.backend(), repeat=1, startup_timeout_s=60)
    saturated = build_sweep(load="saturated")[0]

    outcome = run_sweep([saturated], tmp_path, h.backend(), repeat=1, startup_timeout_s=60,
                        force=True)[0]

    assert outcome.ok and not outcome.skipped


@pytest.mark.parametrize("bad", ["0", "-1", "abc"])
def test_main_rejects_a_non_positive_repeat_with_a_usage_error(bad):
    with pytest.raises(SystemExit) as info:
        main(["--dry-run", "--repeat", bad])

    assert info.value.code == 2


def test_sigterm_and_sighup_become_keyboard_interrupt_and_handlers_are_restored():
    before = (signal.getsignal(signal.SIGTERM), signal.getsignal(signal.SIGHUP))

    with _interrupt_on_signals():
        for sig in (signal.SIGTERM, signal.SIGHUP):
            with pytest.raises(KeyboardInterrupt):
                signal.raise_signal(sig)

    assert (signal.getsignal(signal.SIGTERM), signal.getsignal(signal.SIGHUP)) == before


def test_main_returns_130_when_interrupted_and_still_tears_down(tmp_path):
    h = Harness(bench_exc=KeyboardInterrupt())

    code = main(["--out", str(tmp_path), "--only", "max_num_seqs"], backend_factory=h.backend)

    assert code == 130
    assert h.servers[0].calls == ["terminate"]
