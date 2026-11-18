"""Turning vLLM bench JSON (+ sidecar) into the unified row schema. Inputs are SYNTHETIC."""
from __future__ import annotations

import json

import pytest
from helpers import write_bench_result, write_sidecar

import ingest_gpu as ig
import matrix as mx
import results_io as rio

CASE = next(c for c in mx.build_matrix() if c.case_id == "B-c16-tuned")


def test_row_maps_vllm_keys_to_the_shared_schema(tmp_path):
    path = write_bench_result(tmp_path, CASE, rep=1)
    write_sidecar(tmp_path, CASE, rep=1)
    row = ig.load_run(path)
    assert row["case_id"] == "B-c16-tuned" and row["rep"] == 1 and row["config_id"] == CASE.config_id
    assert row["source"] == "vllm-gpu" and row["simulated"] is False
    assert row["mean_ttft_ms"] == 300.0 and row["p95_e2e_ms"] == 15000.0 and row["mean_e2e_ms"] == 9000.0
    assert row["req_per_s"] == 2.5 and row["output_tok_s"] == 800.0
    assert row["peak_gpu_mem_gib"] == pytest.approx(36864 / 1024)
    assert row["kv_peak_frac"] == 0.8 and row["preemptions"] == 2.0 and row["peak_waiting"] == 3.0


def test_missing_sidecar_leaves_server_side_metrics_not_measured(tmp_path):
    row = ig.load_run(write_bench_result(tmp_path, CASE))
    assert row["peak_gpu_mem_gib"] is None and row["kv_peak_frac"] is None and row["preemptions"] is None


def test_unexposed_metrics_inside_the_sidecar_stay_none(tmp_path):
    path = write_bench_result(tmp_path, CASE)
    write_sidecar(tmp_path, CASE, poll={"kv_peak_frac": None, "waiting_peak": None, "preemptions": None,
                                        "samples": 0, "errors": 4},
                  gpu_mem={"peak_mib": None, "n_samples": 0, "errors": 3, "gpu_index": 0})
    row = ig.load_run(path)
    assert row["kv_peak_frac"] is None and row["peak_gpu_mem_gib"] is None


def test_missing_required_key_is_a_format_error(tmp_path):
    path = write_bench_result(tmp_path, CASE)
    data = json.loads(path.read_text())
    del data["p95_e2el_ms"]
    path.write_text(json.dumps(data))
    with pytest.raises(ig.ResultFormatError, match="p95_e2el_ms"):
        ig.load_run(path)


@pytest.mark.parametrize("override", [{"failed": 3}, {"completed": 99}])
def test_failed_or_partial_runs_are_rejected(tmp_path, override):
    with pytest.raises(ig.ResultFormatError, match="incomplete|failed"):
        ig.load_run(write_bench_result(tmp_path, CASE, **override))


def test_nan_or_non_numeric_metric_is_rejected(tmp_path):
    path = write_bench_result(tmp_path, CASE)
    path.write_text(path.read_text().replace('"mean_ttft_ms": 300.0', '"mean_ttft_ms": NaN'))
    with pytest.raises(ig.ResultFormatError, match="mean_ttft_ms"):
        ig.load_run(path)


def test_observed_concurrency_above_the_target_is_rejected(tmp_path):
    with pytest.raises(ig.ResultFormatError, match="concurrency"):
        ig.load_run(write_bench_result(tmp_path, CASE, max_concurrent_requests=CASE.concurrency + 5))


def test_result_from_a_changed_configuration_is_rejected(tmp_path):
    with pytest.raises(ig.ResultFormatError, match="config_id"):
        ig.load_run(write_bench_result(tmp_path, CASE, matrix_config_id="deadbeef0000"))


def test_unknown_case_in_metadata_is_rejected(tmp_path):
    with pytest.raises(ig.ResultFormatError, match="case"):
        ig.load_run(write_bench_result(tmp_path, CASE, matrix_case_id="Z-c1-nope"))


def test_control_case_round_trips(tmp_path):
    control = next(r.case for r in mx.build_runs(3, 5) if r.case.is_control)
    row = ig.load_run(write_bench_result(tmp_path, control, rep=2))
    assert row["is_control"] is True and row["case_id"] == "B-c16-baseline-aa"


def test_load_dir_returns_good_rows_and_lists_problems(tmp_path):
    other = next(c for c in mx.build_matrix() if c.case_id == "A-c4-baseline")
    write_bench_result(tmp_path, CASE)
    write_bench_result(tmp_path, other, failed=1)
    (tmp_path / "notes.json").write_text("{}")  # not a result file name: ignored
    rows, problems = ig.load_dir(tmp_path)
    assert [r["case_id"] for r in rows] == ["B-c16-tuned"]
    assert len(problems) == 1 and "A-c4-baseline" in problems[0]


def test_cli_writes_the_same_files_as_the_simulator(tmp_path):
    raw, out = tmp_path / "raw", tmp_path / "out"
    for rep in range(2):
        write_bench_result(raw, CASE, rep=rep)
    assert ig.main(["--raw", str(raw), "--out", str(out)]) == 0
    data = rio.read_results(out / "matrix.json")
    assert data["meta"]["simulated"] is False and len(data["rows"]) == 2
    assert (out / "matrix_summary.json").exists()


def test_cli_exits_nonzero_when_any_run_is_bad(tmp_path):
    raw = tmp_path / "raw"
    write_bench_result(raw, CASE)
    write_bench_result(raw, CASE, rep=1, failed=2)
    assert ig.main(["--raw", str(raw), "--out", str(tmp_path / "out")]) == 1


# ---------------------------------------------------------------- hostile / malformed input
def _write_raw(tmp_path, text, name="B-c16-tuned__r0.json"):
    path = tmp_path / name
    path.write_text(text)
    return path


@pytest.mark.parametrize("text", ["[]", '"x"', "null", "{", "[" * 200_000])
def test_non_object_or_unparseable_json_is_a_format_error(tmp_path, text):
    with pytest.raises(ig.ResultFormatError):
        ig.load_run(_write_raw(tmp_path, text))


@pytest.mark.parametrize("rep", ["abc", [1], -5, True, 1.5, None])
def test_bad_matrix_rep_is_rejected(tmp_path, rep):
    with pytest.raises(ig.ResultFormatError, match="rep"):
        ig.load_run(write_bench_result(tmp_path, CASE, matrix_rep=rep))


def test_unhashable_case_id_is_rejected(tmp_path):
    with pytest.raises(ig.ResultFormatError, match="case"):
        ig.load_run(write_bench_result(tmp_path, CASE, matrix_case_id=["x"]))


def test_metadata_must_agree_with_the_filename(tmp_path):
    other = next(c for c in mx.build_matrix() if c.case_id == "A-c4-baseline")
    forged = write_bench_result(tmp_path, other)  # contents claim A-c4-baseline ...
    renamed = forged.rename(tmp_path / "B-c64-tuned__r0.json")  # ... under another case's name
    with pytest.raises(ig.ResultFormatError, match="filename"):
        ig.load_run(renamed)


def test_rep_in_metadata_must_match_the_filename(tmp_path):
    with pytest.raises(ig.ResultFormatError, match="filename"):
        ig.load_run(write_bench_result(tmp_path, CASE, rep=0, matrix_rep=3))


def test_non_canonical_filenames_are_reported_not_double_counted(tmp_path):
    write_bench_result(tmp_path, CASE, rep=0)
    (tmp_path / "B-c16-tuned__r00.json").write_text((tmp_path / "B-c16-tuned__r0.json").read_text())
    rows, problems = ig.load_dir(tmp_path)
    assert len(rows) == 1 and len(problems) == 1 and "r00" in problems[0]


def test_negative_metric_is_rejected(tmp_path):
    with pytest.raises(ig.ResultFormatError, match="mean_ttft_ms"):
        ig.load_run(write_bench_result(tmp_path, CASE, mean_ttft_ms=-1.0))


def test_bucketed_bench_concurrency_is_not_trusted_when_exact_intervals_are_available(tmp_path):
    # vLLM reports 68 (1-second buckets) but the intervals never overlap more than 2 at a time.
    starts, lats = [0.0, 0.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]
    path = write_bench_result(tmp_path, CASE, max_concurrent_requests=68, start_times=starts, latencies=lats)
    assert ig.load_run(path)["case_id"] == CASE.case_id


def test_exact_intervals_still_reject_real_over_concurrency(tmp_path):
    n = CASE.concurrency + 1
    path = write_bench_result(tmp_path, CASE, max_concurrent_requests=n, start_times=[0.0] * n, latencies=[1.0] * n)
    with pytest.raises(ig.ResultFormatError, match="concurrency"):
        ig.load_run(path)


def test_missing_observed_concurrency_is_rejected(tmp_path):
    path = write_bench_result(tmp_path, CASE)
    data = json.loads(path.read_text())
    del data["max_concurrent_requests"]
    path.write_text(json.dumps(data))
    with pytest.raises(ig.ResultFormatError, match="max_concurrent_requests"):
        ig.load_run(path)


@pytest.mark.parametrize("side", [
    {"poll": "x"}, {"gpu_mem": [1]}, {"poll": {"kv_peak_frac": "0.5"}}, {"poll": {"kv_peak_frac": 1.5}},
    {"poll": {"kv_peak_frac": -0.1}}, {"poll": {"preemptions": -2}}, {"poll": {"waiting_peak": True}},
    {"gpu_mem": {"peak_mib": "9"}}, {"gpu_mem": {"peak_mib": -4}},
    {"gpu_mem": {"peak_mib": float("nan")}},
])
def test_invalid_sidecar_values_are_rejected(tmp_path, side):
    path = write_bench_result(tmp_path, CASE)
    write_sidecar(tmp_path, CASE, **side)
    with pytest.raises(ig.ResultFormatError, match="sidecar"):
        ig.load_run(path)


def test_sidecar_that_is_not_an_object_is_rejected(tmp_path):
    path = write_bench_result(tmp_path, CASE)
    (tmp_path / "sidecar").mkdir()
    (tmp_path / "sidecar" / path.name).write_text("[1, 2]")
    with pytest.raises(ig.ResultFormatError, match="sidecar"):
        ig.load_run(path)


def test_sidecar_from_another_run_is_rejected(tmp_path):
    path = write_bench_result(tmp_path, CASE)
    write_sidecar(tmp_path, CASE, run_id="A-c4-baseline__r0")
    with pytest.raises(ig.ResultFormatError, match="sidecar.*run_id|run_id"):
        ig.load_run(path)
    write_sidecar(tmp_path, CASE, matrix_config_id="deadbeef0000")
    with pytest.raises(ig.ResultFormatError, match="config_id"):
        ig.load_run(path)


def test_oversized_result_file_is_rejected_without_reading_it(tmp_path, monkeypatch):
    path = write_bench_result(tmp_path, CASE)
    monkeypatch.setattr(ig, "MAX_RESULT_BYTES", 10)
    with pytest.raises(ig.ResultFormatError, match="too large"):
        ig.load_run(path)


def test_oversized_sidecar_is_rejected(tmp_path, monkeypatch):
    path = write_bench_result(tmp_path, CASE)
    write_sidecar(tmp_path, CASE)
    monkeypatch.setattr(ig, "MAX_SIDECAR_BYTES", 10)
    with pytest.raises(ig.ResultFormatError, match="too large"):
        ig.load_run(path)


def test_symlinked_result_is_rejected(tmp_path):
    real = write_bench_result(tmp_path / "elsewhere", CASE)
    link = tmp_path / "B-c16-tuned__r0.json"
    link.symlink_to(real)
    with pytest.raises(ig.ResultFormatError, match="regular file"):
        ig.load_run(link)


def test_one_hostile_file_does_not_stop_the_rest(tmp_path):
    write_bench_result(tmp_path, CASE, rep=0)
    _write_raw(tmp_path, "[" * 200_000, name="B-c16-tuned__r1.json")
    rows, problems = ig.load_dir(tmp_path)
    assert len(rows) == 1 and len(problems) == 1


# ---------------------------------------------------------------- telemetry health gating
def _poll(**over):
    return {"kv_peak_frac": 0.8, "waiting_peak": 3.0, "preemptions": 2.0, "samples": 10, "errors": 0, **over}


@pytest.mark.parametrize("poll", [_poll(samples=1), _poll(samples=4, errors=6), _poll(thread_error="boom")])
def test_unhealthy_metrics_polling_leaves_server_metrics_unmeasured(tmp_path, poll):
    path = write_bench_result(tmp_path, CASE)
    write_sidecar(tmp_path, CASE, poll=poll)
    row = ig.load_run(path)
    assert row["kv_peak_frac"] is None and row["preemptions"] is None and row["peak_waiting"] is None
    assert row["output_tok_s"] == 800.0  # the client-side result is still good


def test_gpu_sampler_thread_error_leaves_memory_unmeasured(tmp_path):
    path = write_bench_result(tmp_path, CASE)
    write_sidecar(tmp_path, CASE, gpu_mem={"peak_mib": 999, "n_samples": 3, "errors": 0, "gpu_index": 0,
                                           "thread_error": "boom"})
    assert ig.load_run(path)["peak_gpu_mem_gib"] is None


def test_poll_sample_counts_must_be_numbers(tmp_path):
    path = write_bench_result(tmp_path, CASE)
    write_sidecar(tmp_path, CASE, poll=_poll(samples="many"))
    with pytest.raises(ig.ResultFormatError, match="sidecar"):
        ig.load_run(path)
