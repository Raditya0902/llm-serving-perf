from __future__ import annotations

import csv
import json

import pytest

import run_experiments as rx


def test_aggregate_means_and_stds_over_seeds() -> None:
    rows = [
        {"strategy": "a", "seed": 0, "x": 1.0, "n": 2},
        {"strategy": "a", "seed": 1, "x": 3.0, "n": 4},
        {"strategy": "b", "seed": 0, "x": 5.0, "n": 6},
    ]
    out = {r["strategy"]: r for r in rx.aggregate(rows, ("strategy",))}
    assert out["a"]["x"] == 2.0 and out["a"]["x_std"] == pytest.approx(2 ** 0.5) and out["a"]["n"] == 3.0  # sample std
    assert out["a"]["n_seeds"] == 2 and out["b"]["x_std"] == 0.0  # single seed -> 0
    assert "seed" not in out["a"]


def _f_row(family, n, seed, tput, p95, itl):
    return {"family": family, "max_running_seqs": n, "seed": seed, "throughput_tok_s": tput,
            "p95_latency_ms": p95, "p95_token_gap_ms": itl, "worst_itl_p95_ms": itl}


def test_tradeoff_pairs_counts_seeds_where_throughput_up_and_tail_worse() -> None:
    rows = []
    for seed, (p95_hi, itl_hi) in enumerate([(110.0, 50.0), (90.0, 50.0)]):
        rows += [_f_row("f", 8, seed, 100.0, 100.0, 10.0), _f_row("f", 16, seed, 150.0, p95_hi, itl_hi)]
    (pair,) = rx.tradeoff_pairs(rows)
    assert pair["throughput_change_pct"] == pytest.approx(50.0)
    assert pair["seeds_tput_up_and_p95_latency_worse"] == 1
    assert pair["seeds_tput_up_and_tail_gap_worse"] == 2
    assert pair["seeds"] == 2


def test_tradeoff_pairs_ignores_noise_level_changes() -> None:
    rows = [_f_row("f", 8, 0, 1000.0, 100.0, 10.0), _f_row("f", 16, 0, 1004.0, 104.0, 10.1)]  # +0.4% / +4% / +1%
    (pair,) = rx.tradeoff_pairs(rows)
    assert pair["seeds_tput_up_and_p95_latency_worse"] == 0  # throughput gain below the 2% threshold
    assert pair["seeds_tput_up_and_tail_gap_worse"] == 0


def test_tradeoff_pairs_requires_material_tail_worsening() -> None:
    rows = [_f_row("f", 8, 0, 1000.0, 100.0, 10.0), _f_row("f", 16, 0, 1200.0, 100.5, 10.0)]  # tail +0.5%
    (pair,) = rx.tradeoff_pairs(rows)
    assert pair["seeds_tput_up_and_p95_latency_worse"] == 0


def test_write_results_creates_json_and_csv(tmp_path) -> None:
    rx.write_results("demo", [{"a": 1, "b": 2.5}, {"a": 3, "c": "x"}], {"k": "v"}, tmp_path)
    data = json.loads((tmp_path / "demo.json").read_text())
    assert data["meta"] == {"k": "v"} and len(data["rows"]) == 2
    with (tmp_path / "demo.csv").open() as fh:
        header = next(csv.reader(fh))
    assert header == ["a", "b", "c"]


def test_write_results_with_no_rows_skips_csv(tmp_path) -> None:
    rx.write_results("empty", [], {}, tmp_path)
    assert (tmp_path / "empty.json").exists() and not (tmp_path / "empty.csv").exists()


def test_smoke_run_of_every_experiment_writes_outputs(tmp_path, capsys) -> None:
    rx._main(["--exp", "all", "--smoke", "--out", str(tmp_path)])
    names = {p.name for p in tmp_path.iterdir()}
    for exp in "abcdef":
        assert f"exp_{exp}_smoke.json" in names and f"exp_{exp}_summary_smoke.csv" in names
    assert "exp_f_pairs_smoke.json" in names and "timeline_smoke.txt" in names
    summary = json.loads((tmp_path / "exp_a_summary_smoke.json").read_text())["rows"]
    assert [r["strategy"] for r in summary] == list(rx.STRATEGY_NAMES)
    assert {"throughput_tok_s", "mean_latency_ms", "p95_latency_ms", "util_proxy"} <= set(summary[0])
    assert "Experiment A" in capsys.readouterr().out


def test_seed_flag_shifts_all_workload_seeds(tmp_path) -> None:
    rx._main(["--exp", "A", "--smoke", "--seed", "10", "--out", str(tmp_path)])
    rows = json.loads((tmp_path / "exp_a_smoke.json").read_text())["rows"]
    assert {r["seed"] for r in rows} == {10, 11}


def test_unknown_experiment_is_rejected() -> None:
    with pytest.raises(SystemExit):
        rx._main(["--exp", "Z"])
