"""Prometheus text parsing, the /metrics poller, and the nvidia-smi peak-memory sampler."""
from __future__ import annotations

import threading

import pytest

import gpu_mem
import prom_metrics as pm

TEXT = """\
# HELP vllm:kv_cache_usage_perc KV-cache usage. 1 means 100 percent usage.
# TYPE vllm:kv_cache_usage_perc gauge
vllm:kv_cache_usage_perc{engine="0",model_name="m"} 0.25
vllm:kv_cache_usage_perc{engine="1",model_name="m"} 0.75
# TYPE vllm:num_requests_waiting gauge
vllm:num_requests_waiting{engine="0",model_name="m"} 7.0
# TYPE vllm:num_preemptions_total counter
vllm:num_preemptions_total{engine="0",model_name="m"} 12.0
vllm:num_preemptions_created{engine="0",model_name="m"} 1.7e9
"""


def test_metric_values_collects_every_label_set():
    assert pm.metric_values(TEXT, "vllm:kv_cache_usage_perc") == [0.25, 0.75]


def test_counter_matches_total_suffix_but_not_created_series():
    assert pm.metric_values(TEXT, "vllm:num_preemptions") == [12.0]


def test_missing_metric_gives_empty_list():
    assert pm.metric_values(TEXT, "vllm:nope") == []


def test_unlabelled_and_exponent_values_parse():
    assert pm.metric_values("vllm:x 1.5e3\n", "vllm:x") == [1500.0]


def test_non_finite_samples_are_dropped():
    assert pm.metric_values("vllm:x NaN\nvllm:x +Inf\nvllm:x 2\n", "vllm:x") == [2.0]


def _poller(texts):
    it = iter(texts)

    def fetch(_url):
        value = next(it)
        if isinstance(value, Exception):
            raise value
        return value

    return pm.MetricsPoller("http://127.0.0.1:8000/metrics", fetch, interval_s=0.001)


def _text(kv, waiting, pre):
    return (f'vllm:kv_cache_usage_perc{{e="0"}} {kv}\nvllm:num_requests_waiting{{e="0"}} {waiting}\n'
            f'vllm:num_preemptions_total{{e="0"}} {pre}\n')


def test_poller_tracks_peaks_and_preemption_delta():
    p = _poller([_text(0.1, 0, 3), _text(0.9, 5, 5), _text(0.4, 2, 9)])
    for _ in range(3):
        p.poll_once()
    s = p.summary()
    assert s["kv_peak_frac"] == 0.9 and s["waiting_peak"] == 5
    assert s["preemptions"] == 6.0  # 9 - 3: only what happened during the run
    assert s["samples"] == 3 and s["errors"] == 0


def test_poller_takes_the_busiest_engine_for_kv_usage():
    p = _poller([TEXT])
    p.poll_once()
    assert p.summary()["kv_peak_frac"] == 0.75


def test_poller_counts_errors_and_keeps_going():
    p = _poller([OSError("down"), _text(0.5, 1, 0)])
    p.poll_once()
    p.poll_once()
    s = p.summary()
    assert s["errors"] == 1 and s["samples"] == 1 and s["kv_peak_frac"] == 0.5


def test_poller_with_no_successful_sample_reports_not_measured():
    p = _poller([OSError("down")])
    p.poll_once()
    s = p.summary()
    assert s["kv_peak_frac"] is None and s["preemptions"] is None and s["waiting_peak"] is None


def test_poller_without_the_metric_reports_none_not_zero():
    p = _poller(["# nothing useful\n"])
    p.poll_once()
    assert p.summary()["kv_peak_frac"] is None


def test_poller_thread_starts_stops_and_samples():
    seen = threading.Event()

    def fetch(_url):
        seen.set()
        return _text(0.3, 0, 0)

    p = pm.MetricsPoller("http://127.0.0.1:1/metrics", fetch, interval_s=0.001)
    with p:
        assert seen.wait(timeout=2)
    assert p.summary()["samples"] >= 1


def test_poller_refuses_non_loopback_urls():
    with pytest.raises(ValueError, match="loopback"):
        pm.MetricsPoller("http://example.com/metrics", lambda u: "", interval_s=1)


def test_poller_rejects_non_positive_interval():
    with pytest.raises(ValueError, match="interval"):
        pm.MetricsPoller("http://127.0.0.1:1/metrics", lambda u: "", interval_s=0)


# ---------------------------------------------------------------- GPU memory
def test_parse_memory_used_one_value_per_gpu():
    assert gpu_mem.parse_memory_used("1024\n2048\n") == [1024, 2048]


@pytest.mark.parametrize("bad", ["", "abc", "12.5 MiB"])
def test_parse_memory_used_rejects_garbage(bad):
    with pytest.raises(ValueError):
        gpu_mem.parse_memory_used(bad)


def test_sampler_tracks_the_peak_of_the_selected_gpu():
    readings = iter(["100\n900\n", "300\n700\n", "200\n800\n"])
    s = gpu_mem.PeakMemorySampler(lambda: next(readings), gpu_index=1, interval_s=0.001)
    for _ in range(3):
        s.sample_once()
    assert s.result().peak_mib == 900 and s.result().n_samples == 3


def test_sampler_with_no_samples_reports_none():
    s = gpu_mem.PeakMemorySampler(lambda: "1\n", gpu_index=0, interval_s=0.001)
    assert s.result().peak_mib is None


def test_sampler_bad_gpu_index_is_an_error():
    s = gpu_mem.PeakMemorySampler(lambda: "1\n", gpu_index=3, interval_s=0.001)
    with pytest.raises(ValueError, match="gpu_index"):
        s.sample_once()


def test_sampler_survives_a_failing_query_and_counts_it():
    calls = iter([OSError("smi died"), "500\n"])

    def query():
        v = next(calls)
        if isinstance(v, Exception):
            raise v
        return v

    s = gpu_mem.PeakMemorySampler(query, gpu_index=0, interval_s=0.001)
    s.sample_once()
    s.sample_once()
    assert s.result().peak_mib == 500 and s.result().errors == 1


def test_sampler_thread_runs_as_context_manager():
    s = gpu_mem.PeakMemorySampler(lambda: "42\n", gpu_index=0, interval_s=0.001)
    with s:
        threading.Event().wait(0.02)
    assert s.result().peak_mib == 42


# ---------------------------------------------------------------- review/security hardening
def test_http_protocol_errors_count_as_poll_errors_not_thread_death():
    import http.client
    p = _poller([http.client.IncompleteRead(b"x"), _text(0.5, 1, 0)])
    p.poll_once()
    p.poll_once()
    s = p.summary()
    assert s["errors"] == 1 and s["samples"] == 1 and s["thread_error"] is None


def test_unexpected_failure_in_the_poller_thread_makes_everything_unmeasured():
    def boom(_url):
        raise RuntimeError("bug")

    p = pm.MetricsPoller("http://127.0.0.1:1/metrics", boom, interval_s=0.001)
    with p:
        threading.Event().wait(0.05)
    s = p.summary()
    assert s["thread_error"] and s["kv_peak_frac"] is None and s["preemptions"] is None


def test_counter_reset_is_not_reported_as_zero_preemptions():
    p = _poller([_text(0.1, 0, 9), _text(0.1, 0, 2)])
    p.poll_once()
    p.poll_once()
    assert p.summary()["preemptions"] is None


@pytest.mark.parametrize("url", ["file://127.0.0.1/etc/passwd", "ftp://localhost/x", "gopher://127.0.0.1/"])
def test_poller_requires_plain_http(url):
    with pytest.raises(ValueError, match="http"):
        pm.MetricsPoller(url, lambda u: "", interval_s=1)


def test_sampler_counts_unparseable_nvidia_smi_output_as_an_error():
    s = gpu_mem.PeakMemorySampler(lambda: "[N/A]\n", gpu_index=0, interval_s=0.001)
    s.sample_once()
    r = s.result()
    assert r.peak_mib is None and r.errors == 1


def test_sampler_thread_death_is_reported_not_raised():
    def boom():
        raise RuntimeError("bug")

    s = gpu_mem.PeakMemorySampler(boom, gpu_index=0, interval_s=0.001)
    with s:
        threading.Event().wait(0.05)
    r = s.result()
    assert r.peak_mib is None and r.thread_error


def test_negative_gpu_index_is_rejected():
    with pytest.raises(ValueError, match="gpu_index"):
        gpu_mem.PeakMemorySampler(lambda: "1\n", gpu_index=-1, interval_s=1)
