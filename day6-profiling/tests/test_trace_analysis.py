import pytest

from conftest import SYNTHETIC_NOTE
from trace_analysis import analyze_gaps, summarize_alloc_events

assert SYNTHETIC_NOTE  # fixtures below are hand-made, not measurements


def kernel(ts: float, dur: float, name: str = "k") -> dict:
    return {"ph": "X", "cat": "kernel", "name": name, "ts": ts, "dur": dur}


def test_gaps_report_is_unavailable_without_device_events():
    trace = {"traceEvents": [{"ph": "X", "cat": "cpu_op", "name": "aten::mm", "ts": 0, "dur": 5}]}

    report = analyze_gaps(trace)

    assert report.available is False
    assert "no device kernel events" in report.reason
    assert report.busy_fraction is None


def test_gaps_busy_fraction_and_gap_statistics():
    # busy [0,10] [20,30] [30,40] (last two touch) [100,110]; window 0..110
    trace = {"traceEvents": [kernel(0, 10), kernel(20, 10), kernel(30, 10), kernel(100, 10)]}

    report = analyze_gaps(trace)

    assert report.available is True
    assert report.n_kernels == 4
    assert report.window_us == pytest.approx(110)
    assert report.busy_fraction == pytest.approx(40 / 110)
    assert report.gap_max_us == pytest.approx(60)
    assert report.gaps_us == (10.0, 60.0)


def test_overlapping_kernels_are_merged_not_double_counted():
    trace = {"traceEvents": [kernel(0, 10), kernel(5, 10)]}

    report = analyze_gaps(trace)

    assert report.busy_fraction == pytest.approx(1.0)
    assert report.gaps_us == ()


def test_gap_percentiles_interpolate_linearly_like_measure_percentile():
    # gaps of 1..21 us: p50 = 11, p95 = 1 + 0.95*20 = 20
    events, t = [], 0.0
    for gap in range(1, 22):
        events.append(kernel(t, 10))
        t += 10 + gap
    events.append(kernel(t, 10))

    report = analyze_gaps({"traceEvents": events})

    assert report.gap_p50_us == pytest.approx(11)
    assert report.gap_p95_us == pytest.approx(20)


def test_gaps_use_only_the_busiest_device_so_other_gpus_do_not_hide_idle_time():
    dev0 = [{**kernel(0, 10), "args": {"device": 0}}, {**kernel(100, 10), "args": {"device": 0}},
            {**kernel(200, 10), "args": {"device": 0}}]
    dev1 = [{**kernel(0, 300), "args": {"device": 1}}]

    report = analyze_gaps({"traceEvents": dev0 + dev1})

    assert report.n_kernels == 3
    assert report.busy_fraction == pytest.approx(30 / 210)


def test_n_kernels_excludes_memcpy_events_but_they_still_count_as_busy_time():
    memcpy = {"ph": "X", "cat": "gpu_memcpy", "name": "copy", "ts": 10, "dur": 10}

    report = analyze_gaps({"traceEvents": [kernel(0, 10), memcpy]})

    assert report.n_kernels == 1
    assert report.busy_fraction == pytest.approx(1.0)


def test_alloc_summary_counts_cuda_device_allocations_separately():
    host = {"ph": "i", "name": "[memory]", "args": {"Bytes": 10, "Device Type": 0}}
    cuda = {"ph": "i", "name": "[memory]", "args": {"Bytes": 20, "Device Type": 1}}

    summary = summarize_alloc_events({"traceEvents": [host, cuda]})

    assert (summary.n_allocs, summary.n_device_allocs) == (2, 1)


def test_alloc_event_summary_counts_allocations_frees_and_bytes():
    mem = lambda b: {"ph": "i", "name": "[memory]", "args": {"Bytes": b, "Device Type": 0}}
    trace = {"traceEvents": [mem(100), mem(200), mem(-100), {"ph": "X", "name": "aten::mm"}]}

    summary = summarize_alloc_events(trace)

    assert summary.n_allocs == 2
    assert summary.n_frees == 1
    assert summary.bytes_allocated == 300
    assert summary.bytes_freed == 100


def test_alloc_event_summary_is_empty_for_trace_without_memory_events():
    summary = summarize_alloc_events({"traceEvents": []})
    assert (summary.n_allocs, summary.n_frees) == (0, 0)
