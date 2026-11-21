"""All-reduce share from an nsys SQLite export, using a synthetic database with the real table names."""
from __future__ import annotations

import sqlite3

import pytest

import analyze_nsys as an

NCCL = "ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevComm*, unsigned long, ncclWork*)"
CUSTOM = "void vllm::cross_device_reduce_1stage<__nv_bfloat16, 2>(vllm::RankData*, vllm::RankSignals, int, int)"
GEMM = "ampere_bf16_s16816gemm_bf16_128x64_ldg8_f2f_stages_64x3_tn"


def _db(path, rows):
    """rows: (device, start_ns, end_ns, kernel name)."""
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT)")
    con.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start INTEGER, end INTEGER, deviceId INTEGER, demangledName INTEGER)")
    names = {}
    for dev, start, end, name in rows:
        sid = names.setdefault(name, len(names) + 1)
        con.execute("INSERT OR IGNORE INTO StringIds VALUES (?, ?)", (sid, name))
        con.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, ?, ?)", (start, end, dev, sid))
    con.commit()
    con.close()


@pytest.mark.parametrize("name,expected", [
    (NCCL, "allreduce"), (CUSTOM, "allreduce"), ("void vllm::cross_device_reduce_2stage<half, 4>(...)", "allreduce"),
    ("ncclDevKernel_AllGather_RING_LL(...)", "nccl_other"), (GEMM, "other"),
    ("void cutlass::Kernel2<cutlass_80_tensorop_s16816gemm_relu_bf16>(...)", "other")])
def test_classify_kernel_names(name, expected):
    assert an.classify(name) == expected


def test_share_is_allreduce_time_over_busy_time_per_device_and_overall(tmp_path):
    db = tmp_path / "t.sqlite"
    _db(db, [(0, 0, 600, GEMM), (0, 600, 800, NCCL), (1, 0, 700, GEMM), (1, 700, 800, CUSTOM)])
    out = an.summarize(db)
    assert out["per_device"]["0"]["allreduce_share_of_busy"] == pytest.approx(200 / 800)
    assert out["per_device"]["1"]["allreduce_share_of_busy"] == pytest.approx(100 / 800)
    assert out["allreduce_share_of_busy"] == pytest.approx(300 / 1600)
    assert out["per_device"]["0"]["kernel_instances"] == 2


def test_busy_fraction_of_window_exposes_idle_gaps(tmp_path):
    db = tmp_path / "t.sqlite"
    _db(db, [(0, 0, 100, GEMM), (0, 900, 1000, GEMM)])  # busy 200 ns of a 1000 ns window
    assert an.summarize(db)["per_device"]["0"]["busy_fraction_of_window"] == pytest.approx(0.2)


def test_overlapping_kernels_are_not_double_counted_in_busy_time(tmp_path):
    db = tmp_path / "t.sqlite"
    _db(db, [(0, 0, 100, GEMM), (0, 50, 150, NCCL)])
    dev = an.summarize(db)["per_device"]["0"]
    assert dev["busy_ns"] == 150 and dev["allreduce_ns"] == 100  # raw kernel time kept for the share numerator


def test_top_kernels_are_ranked_by_total_time(tmp_path):
    db = tmp_path / "t.sqlite"
    _db(db, [(0, 0, 10, GEMM), (0, 10, 40, NCCL), (0, 40, 45, GEMM)])
    top = an.summarize(db)["top_kernels"]
    assert top[0]["name"] == NCCL and top[0]["total_ns"] == 30 and top[1]["instances"] == 2


def test_empty_capture_is_reported_not_divided_by_zero(tmp_path):
    db = tmp_path / "t.sqlite"
    _db(db, [])
    out = an.summarize(db)
    assert out["allreduce_share_of_busy"] is None and out["per_device"] == {}


def _db_with_graphs(path, kernels, graph_trace_rows):
    """Like `_db` but kernels carry a graphId (0 = launched outside a graph) and a GRAPH_TRACE table exists."""
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT)")
    con.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start INTEGER, end INTEGER, deviceId INTEGER, demangledName INTEGER, graphId INTEGER)")
    con.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_GRAPH_TRACE (start INTEGER, end INTEGER)")
    names = {}
    for dev, start, end, name, graph in kernels:
        sid = names.setdefault(name, len(names) + 1)
        con.execute("INSERT OR IGNORE INTO StringIds VALUES (?, ?)", (sid, name))
        con.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, ?, ?, ?)", (start, end, dev, sid, graph))
    for i in range(graph_trace_rows):
        con.execute("INSERT INTO CUPTI_ACTIVITY_KIND_GRAPH_TRACE VALUES (?, ?)", (i, i + 1))
    con.commit()
    con.close()


def test_graph_launches_without_visible_graph_kernels_are_flagged(tmp_path):
    db = tmp_path / "g.sqlite"
    _db_with_graphs(db, [(0, 0, 10, GEMM, 0)], graph_trace_rows=50)  # default nsys: graph contents hidden
    out = an.summarize(db)
    assert out["graph_launches"] == 50 and out["kernels_from_graph_nodes"] == 0
    assert out["graph_contents_visible"] is False
    assert "cuda-graph-trace=node" in out["warning"]


def test_node_level_graph_tracing_is_reported_as_visible(tmp_path):
    db = tmp_path / "g.sqlite"
    _db_with_graphs(db, [(0, 0, 10, GEMM, 7), (0, 10, 20, NCCL, 7)], graph_trace_rows=1)
    out = an.summarize(db)
    assert out["kernels_from_graph_nodes"] == 2 and out["graph_contents_visible"] is True and "warning" not in out


def test_runs_without_any_graphs_need_no_warning(tmp_path):
    db = tmp_path / "t.sqlite"  # the plain schema from `_db` has neither graphId nor a graph table
    _db(db, [(0, 0, 10, GEMM)])
    out = an.summarize(db)
    assert out["graph_launches"] == 0 and out["graph_contents_visible"] is True and "warning" not in out


def test_sampler_launches_count_decode_steps_per_device(tmp_path):
    db = tmp_path / "t.sqlite"
    sampler = "void flashinfer::sampling::TopPSamplingFromProbKernel<(unsigned int)1024>(T6 *)"
    _db(db, [(0, 0, 1, sampler), (0, 2, 3, sampler), (0, 4, 5, NCCL), (1, 0, 1, sampler)])
    per = an.summarize(db)["per_device"]
    assert per["0"]["sampler_launches"] == 2 and per["1"]["sampler_launches"] == 1
    assert per["0"]["allreduce_instances"] == 1
