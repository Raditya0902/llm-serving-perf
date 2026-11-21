"""Smoke tests for the real subprocess/HTTP backend, using harmless stand-ins for vLLM."""
from __future__ import annotations

import http.server
import pathlib
import socket
import subprocess
import sys
import threading
import time

import pytest

import run_sweep as run_sweep_module
from run_sweep import (
    _fetch_text,
    _is_healthy,
    _is_port_free,
    _run_bench,
    _start_server,
    default_backend,
    stop_server,
)

SLEEPER = [sys.executable, "-c", "import time; time.sleep(60)"]
SIGTERM_IGNORER = [
    sys.executable, "-c",
    "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
    "print('ready', flush=True); time.sleep(60)",
]
STARTUP_WAIT_S = 10.0
# Leader that spawns a long-lived grandchild in the same process group, prints its pid,
# then either stays alive (GROUP_LEADER) or exits at once, orphaning the grandchild.
_GRANDCHILD = "c = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); "
GROUP_LEADER = [sys.executable, "-c",
                "import subprocess, sys, time; " + _GRANDCHILD + "print(c.pid, flush=True); time.sleep(60)"]
ORPHANING_LEADER = [sys.executable, "-c",
                    "import subprocess, sys; " + _GRANDCHILD + "print(c.pid, flush=True)"]


def _read_pid(log, timeout_s=STARTUP_WAIT_S):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        text = log.read_text().strip()
        if text.isdigit():
            return int(text)
        time.sleep(0.05)
    raise AssertionError("child never reported its grandchild's pid")


def _is_gone(pid, timeout_s=10.0):
    """True once ``pid`` no longer runs (a zombie awaiting reaping counts as gone)."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        state = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)],
                               capture_output=True, text=True, check=False).stdout.strip()
        if state == "" or state.startswith("Z"):
            return True
        time.sleep(0.1)
    return False


class _Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802 - stdlib naming
        body = b"ok" if self.path == "/health" else b'vllm:num_preemptions_total{} 1.0\n'
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):  # silence test output
        pass


@pytest.fixture
def local_server():
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


@pytest.fixture
def closed_port_url():
    server = http.server.HTTPServer(("127.0.0.1", 0), _Handler)
    port = server.server_address[1]
    server.server_close()  # nothing listens here any more
    return f"http://127.0.0.1:{port}"


def test_server_process_is_started_and_terminated_via_its_process_group(tmp_path):
    log = tmp_path / "logs" / "s.log"

    handle = _start_server(SLEEPER, log)

    assert handle.poll() is None
    assert log.exists()
    stop_server(handle, grace_s=10)
    assert handle.poll() is not None


def test_server_that_ignores_sigterm_is_killed_after_the_grace_period(tmp_path):
    log = tmp_path / "s.log"
    handle = _start_server(SIGTERM_IGNORER, log)
    deadline = time.monotonic() + STARTUP_WAIT_S
    while "ready" not in log.read_text() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert "ready" in log.read_text(), "child never installed its SIGTERM handler"

    stop_server(handle, grace_s=0.3)

    assert handle.poll() is not None


def test_missing_server_binary_raises_and_closes_the_log_handle(tmp_path, monkeypatch):
    opened = []
    real_open = pathlib.Path.open

    def recording_open(self, *args, **kwargs):
        handle = real_open(self, *args, **kwargs)
        opened.append(handle)
        return handle

    monkeypatch.setattr(pathlib.Path, "open", recording_open)

    with pytest.raises(FileNotFoundError):
        _start_server(["definitely-not-a-real-binary-xyz"], tmp_path / "s.log")

    assert len(opened) == 1 and opened[0].closed


def test_stop_server_kills_grandchildren_through_the_process_group(tmp_path):
    log = tmp_path / "s.log"
    handle = _start_server(GROUP_LEADER, log)
    grandchild = _read_pid(log)

    stop_server(handle, grace_s=10)

    assert _is_gone(grandchild), "engine/worker-style child survived server teardown"


def test_stop_server_kills_children_that_outlive_an_already_exited_leader(tmp_path):
    # The scenario that leaves a GPU occupied: API server gone, worker still running.
    log = tmp_path / "s.log"
    handle = _start_server(ORPHANING_LEADER, log)
    grandchild = _read_pid(log)
    deadline = time.monotonic() + STARTUP_WAIT_S
    while handle.poll() is None and time.monotonic() < deadline:
        time.sleep(0.05)
    assert handle.poll() is not None, "leader should have exited"
    assert not _is_gone(grandchild, timeout_s=0.3), "orphan should still be running"

    stop_server(handle, grace_s=10)

    assert _is_gone(grandchild)


def test_handle_wait_times_out_while_a_group_member_survives(tmp_path):
    log = tmp_path / "s.log"
    handle = _start_server(ORPHANING_LEADER, log)
    grandchild = _read_pid(log)
    deadline = time.monotonic() + STARTUP_WAIT_S
    while handle.poll() is None and time.monotonic() < deadline:
        time.sleep(0.05)

    with pytest.raises(subprocess.TimeoutExpired):
        handle.wait(timeout=0.3)

    stop_server(handle, grace_s=10)
    assert _is_gone(grandchild)


def test_handle_close_is_idempotent(tmp_path):
    handle = _start_server([sys.executable, "-c", "pass"], tmp_path / "s.log")
    handle.wait(timeout=10)

    handle.close()
    handle.close()


def test_run_bench_times_out_instead_of_hanging(monkeypatch):
    monkeypatch.setattr(run_sweep_module, "BENCH_TIMEOUT_S", 0.3)

    with pytest.raises(subprocess.TimeoutExpired):
        _run_bench(SLEEPER)


def test_port_probe_sees_a_listener_and_a_free_port():
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    busy_port = listener.getsockname()[1]
    try:
        assert _is_port_free(busy_port) is False
    finally:
        listener.close()
    assert _is_port_free(busy_port) is True


def test_proxy_environment_variables_do_not_intercept_local_requests(local_server, monkeypatch):
    for name in ("http_proxy", "HTTP_PROXY", "all_proxy", "ALL_PROXY"):
        monkeypatch.setenv(name, "http://127.0.0.1:9")  # nothing listens on the discard port
    monkeypatch.delenv("no_proxy", raising=False)
    monkeypatch.delenv("NO_PROXY", raising=False)

    assert _is_healthy(local_server) is True
    assert "num_preemptions" in _fetch_text(f"{local_server}/metrics")


def test_run_bench_returns_the_exit_code():
    assert _run_bench([sys.executable, "-c", "raise SystemExit(3)"]) == 3
    assert _run_bench([sys.executable, "-c", "pass"]) == 0


def test_is_healthy_true_for_a_live_server(local_server):
    assert _is_healthy(local_server) is True


def test_is_healthy_false_when_nothing_listens(closed_port_url):
    assert _is_healthy(closed_port_url) is False


def test_fetch_text_returns_the_body(local_server):
    assert "num_preemptions" in _fetch_text(f"{local_server}/metrics")


def test_fetch_text_raises_oserror_when_unreachable(closed_port_url):
    with pytest.raises(OSError):
        _fetch_text(f"{closed_port_url}/metrics")


def test_default_backend_wires_all_real_callables():
    backend = default_backend()

    assert backend.start_server is _start_server
    assert backend.fetch_text is _fetch_text
    assert backend.is_port_free is _is_port_free
    assert callable(backend.sleep) and callable(backend.monotonic)
