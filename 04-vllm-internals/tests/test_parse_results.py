from __future__ import annotations

import csv
import json
import math

import pytest

from helpers import make_result, write_result
from parse_results import (
    REQUIRED_METRICS,
    OPTIONAL_METRICS,
    ResultFormatError,
    aggregate,
    load_dir,
    load_result,
    main,
    parse_preemptions,
    write_summary,
)


def test_load_result_extracts_params_and_metrics(tmp_path):
    path = write_result(tmp_path, make_result("max_num_seqs-16", "max_num_seqs", "16",
                                              max_num_seqs=16, output_throughput=123.0))

    rec = load_result(path)

    assert rec.run_id == "max_num_seqs-16"
    assert rec.knob == "max_num_seqs"
    assert rec.rep == 0
    assert rec.params["max_num_seqs"] == 16
    assert rec.params["request_rate"] == 2.0
    assert rec.metrics["output_throughput"] == 123.0
    assert set(rec.metrics) == set(REQUIRED_METRICS) | set(OPTIONAL_METRICS)  # no per-request arrays


def test_inf_request_rate_is_parsed_to_float_inf(tmp_path):
    path = write_result(tmp_path, make_result("request_rate-inf", "request_rate", "inf",
                                              request_rate="inf"))

    assert math.isinf(load_result(path).params["request_rate"])


def test_rep_is_read_from_filename(tmp_path):
    path = write_result(tmp_path, make_result(), rep=3)

    assert load_result(path).rep == 3


def test_missing_required_metric_names_the_key_and_file(tmp_path):
    data = make_result()
    del data["p99_ttft_ms"]
    path = write_result(tmp_path, data)

    with pytest.raises(ResultFormatError, match=r"baseline__r0\.json.*p99_ttft_ms"):
        load_result(path)


def test_missing_sweep_metadata_is_an_error(tmp_path):
    data = make_result()
    del data["sweep_run_id"]
    path = tmp_path / "x__r0.json"
    path.write_text(json.dumps(data))

    with pytest.raises(ResultFormatError, match="sweep_run_id"):
        load_result(path)


def test_invalid_json_is_an_error(tmp_path):
    path = tmp_path / "bad__r0.json"
    path.write_text("{not json")

    with pytest.raises(ResultFormatError, match="not valid JSON"):
        load_result(path)


def test_optional_metrics_may_be_absent(tmp_path):
    data = make_result()
    del data["p99_e2el_ms"]
    path = write_result(tmp_path, data)

    assert load_result(path).metrics["p99_e2el_ms"] is None


METRICS = """\
# HELP vllm:num_preemptions_total Cumulative number of preemption
# TYPE vllm:num_preemptions_total counter
vllm:num_preemptions_total{engine="0",model_name="m"} 3.0
vllm:num_preemptions_total{engine="1",model_name="m"} 2.0
vllm:num_preemptions_created{engine="0",model_name="m"} 1.7e9
vllm:other_total{engine="0"} 99.0
"""


def test_parse_preemptions_sums_label_sets_and_ignores_other_metrics():
    assert parse_preemptions(METRICS) == 5.0


def test_parse_preemptions_returns_none_when_metric_absent():
    assert parse_preemptions("# nothing here\nvllm:other_total 1\n") is None


def test_preemptions_come_from_the_sibling_metrics_file(tmp_path):
    path = write_result(tmp_path, make_result())
    (tmp_path / "metrics").mkdir()
    (tmp_path / "metrics" / "baseline__r0.metrics.txt").write_text(METRICS)

    assert load_result(path).metrics["num_preemptions"] == 5.0


def test_preemptions_are_none_without_a_metrics_file(tmp_path):
    path = write_result(tmp_path, make_result())

    assert load_result(path).metrics["num_preemptions"] is None


def test_load_dir_sorts_and_rejects_empty(tmp_path):
    write_result(tmp_path, make_result("b-2", "b", "2"))
    write_result(tmp_path, make_result("a-1", "a", "1"))

    assert [r.run_id for r in load_dir(tmp_path)] == ["a-1", "b-2"]
    with pytest.raises(ResultFormatError, match="no result"):
        load_dir(tmp_path / "empty")


def test_aggregate_means_over_repetitions(tmp_path):
    write_result(tmp_path, make_result(output_throughput=100.0), rep=0)
    write_result(tmp_path, make_result(output_throughput=300.0), rep=1)
    write_result(tmp_path, make_result("max_num_seqs-4", "max_num_seqs", "4",
                                       max_num_seqs=4, output_throughput=50.0))

    rows = aggregate(load_dir(tmp_path))

    base = next(r for r in rows if r["run_id"] == "baseline")
    assert base["n_reps"] == 2
    assert base["output_throughput"] == pytest.approx(200.0)
    assert base["output_throughput_std"] == pytest.approx(141.42, abs=0.01)  # sample std (ddof=1)
    single = next(r for r in rows if r["run_id"] == "max_num_seqs-4")
    assert single["n_reps"] == 1
    assert single["output_throughput_std"] is None  # one sample: variance unknown, not zero


def test_aggregate_sums_failed_requests(tmp_path):
    write_result(tmp_path, make_result(failed=2), rep=0)
    write_result(tmp_path, make_result(failed=3), rep=1)

    assert aggregate(load_dir(tmp_path))[0]["failed"] == 5


def test_write_summary_outputs_valid_json_and_csv_with_inf(tmp_path):
    write_result(tmp_path / "raw", make_result("request_rate-inf", "request_rate", "inf",
                                               request_rate="inf"))
    rows = aggregate(load_dir(tmp_path / "raw"))

    json_path, csv_path = write_summary(rows, tmp_path / "out")

    data = json.loads(json_path.read_text())  # strict JSON: no bare Infinity
    assert data["rows"][0]["request_rate"] == "inf"
    with csv_path.open() as fh:
        assert next(csv.DictReader(fh))["request_rate"] == "inf"


def test_main_writes_summary_files(tmp_path, capsys):
    write_result(tmp_path / "raw", make_result())

    code = main(["--raw", str(tmp_path / "raw"), "--out", str(tmp_path / "out")])

    assert code == 0
    assert (tmp_path / "out" / "sweep_summary.json").exists()
    assert "baseline" in capsys.readouterr().out


def test_main_returns_nonzero_on_bad_input(tmp_path, capsys):
    code = main(["--raw", str(tmp_path / "nope"), "--out", str(tmp_path / "out")])

    assert code == 1
    assert "no result" in capsys.readouterr().err


# --- review fixes -------------------------------------------------------------------------
def test_missing_config_id_is_an_error(tmp_path):
    data = make_result()
    del data["sweep_config_id"]
    path = write_result(tmp_path, data)

    with pytest.raises(ResultFormatError, match="sweep_config_id"):
        load_result(path)


def test_reps_from_different_configurations_are_never_averaged(tmp_path):
    write_result(tmp_path, make_result(config_id="brief"), rep=0)
    write_result(tmp_path, make_result(config_id="saturated"), rep=1)

    with pytest.raises(ResultFormatError, match="different configurations"):
        aggregate(load_dir(tmp_path))


def test_filename_must_agree_with_the_run_id_inside(tmp_path):
    data = make_result("max_num_seqs-4", "max_num_seqs", "4", max_num_seqs=4)
    path = tmp_path / "baseline__r0.json"
    path.write_text(json.dumps(data))

    with pytest.raises(ResultFormatError, match="does not match"):
        load_result(path)


def test_filename_without_rep_suffix_is_rejected(tmp_path):
    path = tmp_path / "baseline (1).json"
    path.write_text(json.dumps(make_result()))

    with pytest.raises(ResultFormatError, match="does not match"):
        load_result(path)


def test_bad_optional_metric_value_names_the_key(tmp_path):
    path = write_result(tmp_path, make_result(p50_ttft_ms="fast"))

    with pytest.raises(ResultFormatError, match="p50_ttft_ms"):
        load_result(path)


def test_optional_metric_present_in_only_some_reps_is_not_averaged(tmp_path):
    write_result(tmp_path, make_result(), rep=0)
    write_result(tmp_path, make_result(), rep=1)
    (tmp_path / "metrics").mkdir()
    (tmp_path / "metrics" / "baseline__r0.metrics.txt").write_text(METRICS)

    row = aggregate(load_dir(tmp_path))[0]

    assert row["num_preemptions"] is None  # 1 of 2 reps: unknown, not "5"


def test_nan_is_written_as_nan_not_as_negative_infinity(tmp_path):
    rows = [{"run_id": "a", "x": float("nan"), "y": float("inf"), "z": float("-inf")}]

    json_path, _ = write_summary(rows, tmp_path)

    saved = json.loads(json_path.read_text())["rows"][0]
    assert (saved["x"], saved["y"], saved["z"]) == ("nan", "inf", "-inf")
