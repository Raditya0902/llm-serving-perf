from __future__ import annotations

import math

import pytest

from helpers import make_result, write_result
from parse_results import aggregate, load_dir, write_summary
from plot_results import knob_series, load_summary, main, plot_knob

PNG_MAGIC = b"\x89PNG"


@pytest.fixture
def summary_path(tmp_path):
    raw = tmp_path / "raw"
    write_result(raw, make_result(output_throughput=250.0, p99_ttft_ms=80.0))
    for n, tput in ((4, 90.0), (16, 180.0)):
        write_result(raw, make_result(f"max_num_seqs-{n}", "max_num_seqs", str(n),
                                      max_num_seqs=n, output_throughput=tput))
    write_result(raw, make_result("request_rate-inf", "request_rate", "inf",
                                  request_rate="inf", output_throughput=900.0,
                                  p99_ttft_ms=4000.0))
    json_path, _ = write_summary(aggregate(load_dir(raw)), tmp_path)
    return json_path


def test_load_summary_restores_inf_and_numeric_types(summary_path):
    rows = load_summary(summary_path)

    inf_row = next(r for r in rows if r["run_id"] == "request_rate-inf")
    assert math.isinf(inf_row["request_rate"])
    assert isinstance(next(r for r in rows if r["run_id"] == "baseline")["max_num_seqs"], int)


def test_knob_series_includes_baseline_sorted_by_value(summary_path):
    series = knob_series(load_summary(summary_path), "max_num_seqs")

    assert [r["max_num_seqs"] for r in series] == [4, 16, 256]
    assert series[-1]["run_id"] == "baseline"


def test_knob_series_puts_inf_last(summary_path):
    series = knob_series(load_summary(summary_path), "request_rate")

    assert [r["request_rate"] for r in series] == [2.0, math.inf]


def test_plot_knob_writes_a_png(summary_path, tmp_path):
    path = plot_knob(load_summary(summary_path), "max_num_seqs", tmp_path / "plots")

    assert path is not None and path.name == "sweep_max_num_seqs.png"
    assert path.read_bytes()[:4] == PNG_MAGIC


def test_plot_knob_handles_inf_category(summary_path, tmp_path):
    path = plot_knob(load_summary(summary_path), "request_rate", tmp_path, watermark="SYNTHETIC")

    assert path is not None and path.stat().st_size > 0


def test_plot_knob_skips_knob_with_too_few_points(summary_path, tmp_path):
    assert plot_knob(load_summary(summary_path), "input_len", tmp_path) is None


def test_plot_knob_rejects_unknown_knob(summary_path, tmp_path):
    with pytest.raises(ValueError, match="unknown knob"):
        plot_knob(load_summary(summary_path), "tensor_parallel", tmp_path)


def test_main_plots_every_knob_with_data(summary_path, tmp_path, capsys):
    out = tmp_path / "plots"

    code = main(["--summary", str(summary_path), "--out", str(out)])

    assert code == 0
    assert sorted(p.name for p in out.glob("*.png")) == [
        "sweep_max_num_seqs.png", "sweep_request_rate.png"]
    assert "sweep_max_num_seqs.png" in capsys.readouterr().out


def test_main_errors_on_missing_summary(tmp_path, capsys):
    code = main(["--summary", str(tmp_path / "nope.json"), "--out", str(tmp_path)])

    assert code == 1
    assert "summary" in capsys.readouterr().err


# --- review fixes -------------------------------------------------------------------------
def test_load_summary_converts_non_finite_metric_strings_back_to_floats(tmp_path):
    import json
    row = {"run_id": "baseline", "knob": "baseline", "value": "baseline", "model": "m",
           "n_reps": 1, "max_num_seqs": 256, "max_num_batched_tokens": 8192,
           "gpu_memory_utilization": 0.85, "request_rate": "2.0", "input_len": 512,
           "output_len": 128, "failed": 0, "completed": 1,
           "output_throughput": "nan", "p99_ttft_ms": "inf", "p99_itl_ms": 1.5}
    path = tmp_path / "s.json"
    path.write_text(json.dumps({"rows": [row]}))

    loaded = load_summary(path)[0]

    assert math.isnan(loaded["output_throughput"])
    assert math.isinf(loaded["p99_ttft_ms"])
    assert loaded["p99_itl_ms"] == 1.5
    assert loaded["run_id"] == "baseline" and loaded["model"] == "m"  # text stays text


def test_figure_is_closed_even_when_saving_fails(summary_path, tmp_path, monkeypatch):
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

    def boom(self, *args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(Figure, "savefig", boom)
    plt.close("all")

    with pytest.raises(OSError, match="disk full"):
        plot_knob(load_summary(summary_path), "max_num_seqs", tmp_path)

    assert plt.get_fignums() == []
