"""Run the one-factor-at-a-time sweep: ``vllm serve`` + ``vllm bench serve``.

For every run in ``sweep_config.build_sweep`` this starts a fresh server,
waits for ``/health``, runs the benchmark (which saves its own JSON via
``--save-result``), and *always* tears the server down -- including on
failure or Ctrl-C -- so a crashed run cannot leave a GPU-holding process
behind for the next one.

This must be executed on a Linux CUDA host with vLLM installed (see
RUNBOOK.md).  All process / HTTP / clock access goes through ``Backend`` so the
orchestration logic is unit-tested on any machine; ``--dry-run`` prints the
exact commands without needing vLLM at all.
"""
from __future__ import annotations

import argparse
import contextlib
import http.client
import json
import logging
import os
import shlex
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Callable, Iterator, Protocol, Sequence

from sweep_config import DEFAULT_MODEL, LOAD_PROFILES, LOOPBACK, SweepRun, build_sweep

logger = logging.getLogger("run_sweep")

DEFAULT_STARTUP_TIMEOUT_S = 600.0  # first start downloads weights + compiles
HEALTH_POLL_INTERVAL_S = 2.0
SHUTDOWN_GRACE_S = 30.0
HEALTH_REQUEST_TIMEOUT_S = 2.0
METRICS_REQUEST_TIMEOUT_S = 10.0
METRICS_SUBDIR = "metrics"
BENCH_TIMEOUT_S = 3600.0  # a hung bench must not hang the whole sweep
KILL_WAIT_S = 10.0
GROUP_POLL_S = 0.1
PORT_PROBE_TIMEOUT_S = 0.5
EXIT_OK, EXIT_RUN_FAILED, EXIT_USAGE, EXIT_INTERRUPTED = 0, 1, 2, 130


class ServerStartError(RuntimeError):
    """The server process died before it became healthy."""


# Everything that can go wrong around one run without being a bug in this script.
_RUN_ERRORS = (
    ServerStartError, TimeoutError, OSError, subprocess.SubprocessError,
    http.client.HTTPException,
)


class ServerHandle(Protocol):
    def poll(self) -> int | None: ...
    def terminate(self) -> None: ...
    def kill(self) -> None: ...
    def wait(self, timeout: float | None = None) -> int: ...
    def close(self) -> None: ...


@dataclass(frozen=True)
class Backend:
    """Everything with side effects, injectable for tests."""

    start_server: Callable[[list[str], Path], ServerHandle]
    run_bench: Callable[[list[str]], int]
    is_healthy: Callable[[str], bool]
    sleep: Callable[[float], None]
    monotonic: Callable[[], float]
    fetch_text: Callable[[str], str]  # GET a URL; raises OSError on failure
    is_port_free: Callable[[int], bool]


@dataclass(frozen=True)
class RunOutcome:
    run_id: str
    rep: int
    ok: bool
    error: str | None
    result_path: Path | None
    skipped: bool


# --------------------------------------------------------------------------- #
# Orchestration (pure given a Backend)
# --------------------------------------------------------------------------- #
def wait_until_healthy(
    backend: Backend, handle: ServerHandle, base_url: str, *,
    timeout_s: float, interval_s: float = HEALTH_POLL_INTERVAL_S,
) -> None:
    """Block until ``<base_url>/health`` answers; raise if the server dies or times out."""
    started = backend.monotonic()
    while True:
        code = handle.poll()
        if code is not None:
            raise ServerStartError(f"server exited with code {code} before becoming healthy")
        if backend.is_healthy(base_url):
            return
        if backend.monotonic() - started >= timeout_s:
            raise TimeoutError(f"server at {base_url} not healthy after {timeout_s:.0f}s")
        backend.sleep(interval_s)


def stop_server(handle: ServerHandle, grace_s: float = SHUTDOWN_GRACE_S) -> None:
    """SIGTERM the whole server, SIGKILL after ``grace_s``, and always close the handle.

    The group is signalled even when the leader has already exited: vLLM's engine
    and worker children can outlive the API-server process and keep the GPU busy.
    """
    try:
        handle.terminate()
        try:
            handle.wait(timeout=grace_s)
        except subprocess.TimeoutExpired:
            logger.warning("server ignored terminate for %.0fs; killing", grace_s)
            handle.kill()
            try:
                handle.wait(timeout=KILL_WAIT_S)
            except subprocess.TimeoutExpired:
                logger.error("server did not exit after SIGKILL; it may still hold GPU memory")
    finally:
        handle.close()


def result_path_for(run: SweepRun, rep: int, out_dir: Path) -> Path:
    return out_dir / f"{run.run_id}__r{rep}.json"


def metrics_path_for(run: SweepRun, rep: int, out_dir: Path) -> Path:
    return out_dir / METRICS_SUBDIR / f"{run.run_id}__r{rep}.metrics.txt"


def _save_server_metrics(run: SweepRun, rep: int, out_dir: Path, backend: Backend) -> None:
    """Keep the server's Prometheus text (preemption counts etc.) next to the result.

    The bench client cannot see scheduler-internal counters such as preemptions,
    so they are scraped from the server before it is torn down.  A failure here
    is logged, not fatal: the benchmark result itself is already saved.
    """
    try:
        text = backend.fetch_text(f"{run.base_url}/metrics")
    except (OSError, http.client.HTTPException) as exc:
        logger.warning("%s rep %d: could not fetch /metrics: %s", run.run_id, rep, exc)
        return
    path = metrics_path_for(run, rep, out_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _discard_stale_outputs(run: SweepRun, rep: int, out_dir: Path) -> None:
    """Remove leftovers of an earlier attempt so they can never pass for this run's data."""
    result_path_for(run, rep, out_dir).unlink(missing_ok=True)
    metrics_path_for(run, rep, out_dir).unlink(missing_ok=True)


def _existing_result_problem(run: SweepRun, path: Path) -> str | None:
    """Why ``path`` must not be reused for ``run`` (None if it is safe to skip the run)."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return f"existing result {path.name} is unreadable ({exc}); use --force to redo it"
    found = data.get("sweep_config_id") if isinstance(data, dict) else None
    if found != run.config_id:
        return (
            f"existing result {path.name} was produced by a different configuration "
            f"(config_id {found!r}, expected {run.config_id!r}); use --force to overwrite "
            "it or choose another --out directory"
        )
    return None


def execute_run(
    run: SweepRun, rep: int, out_dir: Path, backend: Backend, *, startup_timeout_s: float,
) -> RunOutcome:
    """Serve, benchmark, tear down.  Failures are returned, never swallowed."""
    result_path = result_path_for(run, rep, out_dir)
    log_path = out_dir / "logs" / f"{run.run_id}__r{rep}.server.log"
    handle: ServerHandle | None = None

    def failed(message: str) -> RunOutcome:
        logger.error("%s rep %d FAILED: %s", run.run_id, rep, message)
        return RunOutcome(run.run_id, rep, False, message, None, False)

    if not backend.is_port_free(run.serve.port):
        return failed(
            f"port {run.serve.port} is already in use (stale or foreign server?); refusing "
            "to benchmark a server this run did not start"
        )
    _discard_stale_outputs(run, rep, out_dir)
    try:
        handle = backend.start_server(run.serve_command(), log_path)
        wait_until_healthy(backend, handle, run.base_url, timeout_s=startup_timeout_s)
        returncode = backend.run_bench(run.bench_command(str(out_dir), rep))
        if returncode != 0:
            return failed(f"vllm bench serve failed with exit code {returncode}")
        if not result_path.exists():
            return failed(f"bench finished but result file {result_path} was not written")
        _save_server_metrics(run, rep, out_dir, backend)
        return RunOutcome(run.run_id, rep, True, None, result_path, False)
    except _RUN_ERRORS as exc:
        return failed(str(exc))
    finally:
        if handle is not None:
            stop_server(handle)


def _resume_outcome(run: SweepRun, rep: int, existing: Path) -> RunOutcome:
    """Skip a finished run, but only if it provably came from the same configuration."""
    problem = _existing_result_problem(run, existing)
    if problem is not None:
        logger.error("%s rep %d: %s", run.run_id, rep, problem)
        return RunOutcome(run.run_id, rep, False, problem, None, False)
    logger.info("%s rep %d: result exists, skipping (use --force)", run.run_id, rep)
    return RunOutcome(run.run_id, rep, True, None, existing, True)


def run_sweep(
    runs: Sequence[SweepRun], out_dir: Path, backend: Backend, *,
    repeat: int, startup_timeout_s: float, force: bool = False,
) -> list[RunOutcome]:
    """Execute every run ``repeat`` times, resuming past results unless ``force``."""
    if repeat < 1:
        raise ValueError("repeat must be >= 1")
    outcomes: list[RunOutcome] = []
    for run in runs:
        for rep in range(repeat):
            existing = result_path_for(run, rep, out_dir)
            if existing.exists() and not force:
                outcomes.append(_resume_outcome(run, rep, existing))
                continue
            logger.info("%s rep %d: starting", run.run_id, rep)
            outcomes.append(
                execute_run(run, rep, out_dir, backend, startup_timeout_s=startup_timeout_s)
            )
    return outcomes


# --------------------------------------------------------------------------- #
# Real backend (not exercised by unit tests; needs vLLM on a Linux GPU host)
# --------------------------------------------------------------------------- #
class _PopenHandle:
    """A server started in its own session, so its pgid equals the leader's pid.

    ``vllm serve`` spawns engine-core and worker subprocesses; signalling the whole
    group is what actually releases GPU memory.  The pgid is recorded at spawn time
    and the group is signalled even after the leader has exited and been reaped.
    (A recycled pid could in theory alias the group; the window is the few seconds
    of one teardown.)
    """

    def __init__(self, proc: subprocess.Popen[bytes], log_file: IO[bytes]) -> None:
        self._proc = proc
        self._log = log_file
        self._pgid = proc.pid

    def poll(self) -> int | None:
        return self._proc.poll()

    def _signal_group(self, sig: int) -> None:
        try:
            os.killpg(self._pgid, sig)
        except ProcessLookupError:
            pass  # nothing left in the group

    def _group_alive(self) -> bool:
        try:
            os.killpg(self._pgid, 0)
        except ProcessLookupError:
            return False
        return True

    def terminate(self) -> None:
        self._signal_group(signal.SIGTERM)

    def kill(self) -> None:
        self._signal_group(signal.SIGKILL)

    def wait(self, timeout: float | None = None) -> int:
        """Wait for the leader *and* every process in its group to exit."""
        deadline = None if timeout is None else time.monotonic() + timeout
        code = self._proc.wait(timeout=timeout)
        while self._group_alive():
            if deadline is not None and time.monotonic() >= deadline:
                raise subprocess.TimeoutExpired("vllm server process group", timeout)
            time.sleep(GROUP_POLL_S)
        return code

    def close(self) -> None:
        if not self._log.closed:
            self._log.close()


def _start_server(cmd: list[str], log_path: Path) -> ServerHandle:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("ab")
    try:
        proc = subprocess.Popen(  # noqa: S603 - argv list built by sweep_config
            cmd, stdout=log_file, stderr=subprocess.STDOUT, start_new_session=True,
        )
    except OSError:
        log_file.close()
        raise
    return _PopenHandle(proc, log_file)


def _run_bench(cmd: list[str]) -> int:
    return subprocess.run(cmd, check=False, timeout=BENCH_TIMEOUT_S).returncode  # noqa: S603


# Requests go to 127.0.0.1 only; an http(s)_proxy in the environment must not intercept them.
_LOCAL_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _is_healthy(base_url: str) -> bool:
    # base_url is always http://127.0.0.1:<port> (see SweepRun.base_url).
    try:
        with _LOCAL_OPENER.open(  # nosec B310
            f"{base_url}/health", timeout=HEALTH_REQUEST_TIMEOUT_S
        ) as response:
            return response.status == 200
    except (urllib.error.URLError, OSError, http.client.HTTPException):
        return False


def _fetch_text(url: str) -> str:
    # url is always http://127.0.0.1:<port>/metrics (see SweepRun.base_url).
    with _LOCAL_OPENER.open(url, timeout=METRICS_REQUEST_TIMEOUT_S) as response:  # nosec B310
        return response.read().decode("utf-8", errors="replace")


def _is_port_free(port: int) -> bool:
    """True if nothing is listening on ``LOOPBACK:port`` and it can be bound."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(PORT_PROBE_TIMEOUT_S)
        if probe.connect_ex((LOOPBACK, port)) == 0:
            return False  # something is accepting connections
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as binder:
        binder.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            binder.bind((LOOPBACK, port))
        except OSError:
            return False
    return True


def default_backend() -> Backend:
    return Backend(_start_server, _run_bench, _is_healthy, time.sleep, time.monotonic,
                   _fetch_text, _is_port_free)


@contextlib.contextmanager
def _interrupt_on_signals() -> Iterator[None]:
    """Turn SIGTERM/SIGHUP (kill, SSH disconnect) into KeyboardInterrupt.

    The server runs in its own session, so without this a plain ``kill`` of the
    sweep would skip the ``finally`` blocks and leave the GPU occupied.
    """
    def handler(signum: int, _frame: object) -> None:
        raise KeyboardInterrupt(f"received signal {signum}")

    signals = (signal.SIGTERM, signal.SIGHUP)
    previous = {sig: signal.signal(sig, handler) for sig in signals}
    try:
        yield
    finally:
        for sig, old in previous.items():
            signal.signal(sig, old)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _positive_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not an integer") from None
    if value < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return value


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", type=Path, default=Path("results/raw"),
                   help="directory for raw bench JSON + server logs")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--load", choices=LOAD_PROFILES, default="brief",
                   help="brief: paced 2 req/s (the task); saturated: rate=inf")
    p.add_argument("--only", nargs="+", metavar="KNOB", help="sweep only these knobs")
    p.add_argument("--repeat", type=_positive_int, default=1, help="repetitions per run")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-prompts", type=int, default=100)
    p.add_argument("--no-ignore-eos", action="store_true",
                   help="let the model stop early (output length then varies)")
    p.add_argument("--startup-timeout", type=float, default=DEFAULT_STARTUP_TIMEOUT_S)
    p.add_argument("--force", action="store_true", help="re-run even if a result exists")
    p.add_argument("--dry-run", action="store_true", help="print commands and exit")
    return p


def main(
    argv: Sequence[str] | None = None,
    backend_factory: Callable[[], Backend] = default_backend,
) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        runs = build_sweep(
            model=args.model, load=args.load, only=args.only, port=args.port,
            seed=args.seed, num_prompts=args.num_prompts,
            ignore_eos=not args.no_ignore_eos,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE

    if args.dry_run:
        for run in runs:
            print(f"# {run.run_id}")
            print(shlex.join(run.serve_command()))
            print(shlex.join(run.bench_command(str(args.out), 0)))
            print()
        return EXIT_OK

    try:
        with _interrupt_on_signals():
            outcomes = run_sweep(
                runs, args.out, backend_factory(), repeat=args.repeat,
                startup_timeout_s=args.startup_timeout, force=args.force,
            )
    except KeyboardInterrupt:
        print("interrupted: server torn down; finished runs are kept (re-run to resume)",
              file=sys.stderr)
        return EXIT_INTERRUPTED
    return _report(outcomes)


def _report(outcomes: Sequence[RunOutcome]) -> int:
    failures = [o for o in outcomes if not o.ok]
    for o in failures:
        print(f"FAILED {o.run_id} rep {o.rep}: {o.error}", file=sys.stderr)
    done = sum(o.ok and not o.skipped for o in outcomes)
    skipped = sum(o.skipped for o in outcomes)
    print(f"{done} run(s) completed, {skipped} skipped, {len(failures)} failed")
    return EXIT_RUN_FAILED if failures else EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
