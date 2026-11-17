"""A tiny background thread that calls ``tick()`` every ``interval_s`` until stopped.

Used by the /metrics poller and the GPU-memory sampler so both share one start/stop and
error-handling story. One final ``tick()`` runs after the stop request, so the state at the
end of the measured window is always observed.
"""
from __future__ import annotations

import threading
from collections.abc import Callable

STOP_JOIN_TIMEOUT_S = 30.0


class PeriodicThread:
    def __init__(self, tick: Callable[[], None], interval_s: float, name: str) -> None:
        if interval_s <= 0:
            raise ValueError("interval must be > 0")
        self._tick, self._interval_s = tick, interval_s
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)
        self.error: BaseException | None = None

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                self._tick()
                self._stop.wait(self._interval_s)
            self._tick()  # the end-of-window state
        except Exception as exc:  # noqa: BLE001 - surfaced via .error, never silently lost
            self.error = exc

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(STOP_JOIN_TIMEOUT_S)

    def __enter__(self) -> PeriodicThread:
        self.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.stop()
