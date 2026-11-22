"""Prometheus text parsing and a background poller for vLLM's ``/metrics``.

`04-vllm-internals` only scraped the final preemption counter after a run. The matrix needs the *peak* KV
usage and queue depth during the run, so this polls while the benchmark executes. Names are
the v0.34.0 ones (``vllm/v1/metrics/loggers.py``):

* ``vllm:kv_cache_usage_perc``   gauge, fraction 0..1 (the name says perc; the value is not x100)
* ``vllm:num_requests_waiting``  gauge
* ``vllm:num_preemptions``       counter (exposed as ``..._total``)

A metric the server did not expose is ``None`` in the summary ("not measured"), never zero.
"""
from __future__ import annotations

import http.client
import math
import re
import threading
from collections.abc import Callable
from urllib.parse import urlparse

from periodic import PeriodicThread

KV_USAGE = "vllm:kv_cache_usage_perc"
WAITING = "vllm:num_requests_waiting"
PREEMPTIONS = "vllm:num_preemptions"
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
DEFAULT_INTERVAL_S = 0.5

# name{labels} value   (timestamps are not used by vLLM's exposition)
_SAMPLE_RE = re.compile(r"^([A-Za-z_:][A-Za-z0-9_:]*)(?:\{[^}]*\})?\s+(\S+)\s*$")


def metric_values(text: str, name: str) -> list[float]:
    """Every finite sample of `name` (one per label set). Counters match ``name_total`` too;
    ``name_created`` timestamp series never match."""
    wanted = {name, f"{name}_total"}
    values = []
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        match = _SAMPLE_RE.match(line)
        if not match or match.group(1) not in wanted:
            continue
        try:
            value = float(match.group(2))
        except ValueError:
            continue
        if math.isfinite(value):
            values.append(value)
    return values


class MetricsPoller:
    """Poll ``url`` every `interval_s`; track peak KV usage, peak queue depth, preemption delta."""

    def __init__(
        self, url: str, fetch: Callable[[str], str], interval_s: float = DEFAULT_INTERVAL_S,
    ) -> None:
        parsed = urlparse(url)
        if parsed.scheme != "http":
            raise ValueError(f"metrics URL must use plain http, got {url!r}")
        if parsed.hostname not in LOOPBACK_HOSTS:
            raise ValueError(f"metrics URL must be loopback, got {url!r}")
        self._url, self._fetch = url, fetch
        self._thread = PeriodicThread(self.poll_once, interval_s, "metrics-poller")
        self._lock = threading.Lock()
        self._samples = self._errors = 0
        self._kv_peak: float | None = None
        self._waiting_peak: float | None = None
        self._pre_first: float | None = None
        self._pre_last: float | None = None
        self._pre_samples = 0

    def poll_once(self) -> None:
        try:
            text = self._fetch(self._url)
        except (OSError, ValueError, http.client.HTTPException):
            with self._lock:
                self._errors += 1
            return
        kv, waiting = metric_values(text, KV_USAGE), metric_values(text, WAITING)
        pre = metric_values(text, PREEMPTIONS)
        with self._lock:
            self._samples += 1
            if kv:
                self._kv_peak = max(max(kv), self._kv_peak or 0.0)  # busiest engine
            if waiting:
                self._waiting_peak = max(sum(waiting), self._waiting_peak or 0.0)
            if pre:
                self._pre_first = self._pre_first if self._pre_first is not None else sum(pre)
                self._pre_last = sum(pre)
                self._pre_samples += 1

    def summary(self) -> dict[str, float | int | str | None]:
        """Peaks and counts. If the polling thread died, every server-side value is None: a partial
        peak taken before the failure must not be mistaken for the run's peak."""
        with self._lock:
            common = {"samples": self._samples, "errors": self._errors}
            if self._thread.error is not None:
                return {"kv_peak_frac": None, "waiting_peak": None, "preemptions": None, **common,
                        "thread_error": f"{type(self._thread.error).__name__}: {self._thread.error}"}
            delta = None
            if self._pre_samples >= 2 and self._pre_first is not None and self._pre_last is not None:
                if self._pre_last >= self._pre_first:  # a decrease means a server restart: unmeasured
                    delta = self._pre_last - self._pre_first
            return {"kv_peak_frac": self._kv_peak, "waiting_peak": self._waiting_peak,
                    "preemptions": delta, **common, "thread_error": None}

    def __enter__(self) -> MetricsPoller:
        self._thread.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self._thread.stop()
