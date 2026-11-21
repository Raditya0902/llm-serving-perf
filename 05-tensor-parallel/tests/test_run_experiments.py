from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
from torch.multiprocessing.spawn import ProcessExitedException

import run_experiments as rx


@pytest.fixture(autouse=True)
def _isolated_retry_log():
    rx.SPAWN_RETRY_LOG.clear()
    yield
    rx.SPAWN_RETRY_LOG.clear()


# --- row builders on synthetic data ------------------------------------------------------
def test_collective_rows_compute_latency_and_bandwidths() -> None:
    merged = [
        {"collective": "all_reduce", "size_bytes": 1000, "median_s": 1e-3, "min_s": 9e-4},
        {"collective": "all_gather", "size_bytes": 1000, "median_s": 1e-3, "min_s": 9e-4},
    ]

    ar, ag = rx.collective_rows(2, merged)

    assert ar["world"] == 2 and ar["latency_us"] == pytest.approx(1000.0)
    assert ar["algbw_GBps"] == pytest.approx(0.001)
    assert ar["busbw_GBps"] == pytest.approx(0.001)  # factor 2(n-1)/n = 1 at n=2
    assert ag["transfer_bytes"] == 2000  # all_gather moves world x shard
    assert ag["algbw_GBps"] == pytest.approx(0.002)
    assert ag["busbw_GBps"] == pytest.approx(0.001)  # factor (n-1)/n = 0.5


def test_fit_rows_recover_alpha_and_beta_per_world_and_collective() -> None:
    rows = [
        {"world": 2, "collective": "all_reduce", "transfer_bytes": b, "latency_us": 20.0 + 0.002 * b}
        for b in (1_000, 100_000, 1_000_000)
    ]

    (fit,) = rx.fit_rows(rows)

    assert fit["alpha_us"] == pytest.approx(20.0, rel=1e-4)
    assert fit["beta_ns_per_byte"] == pytest.approx(2.0, rel=1e-4)
    assert fit["implied_bw_GBps"] == pytest.approx(0.5, rel=1e-4)
    assert fit["max_rel_err"] == pytest.approx(0.0, abs=1e-6)


def test_mlp_rows_derive_speedup_comm_fraction_and_memory() -> None:
    merged = [{
        "d_model": 32, "tokens": 4, "ref_s": 2e-3, "tp_total_s": 1e-3, "tp_compute_s": 0.8e-3,
        "tp_comm_s": 0.2e-3, "max_err": 1e-6, "shard_weight_bytes": 50, "full_weight_bytes": 100,
    }]

    (row,) = rx.mlp_rows(2, merged)

    assert row["speedup"] == pytest.approx(2.0)
    assert row["compute_speedup"] == pytest.approx(2.5)
    assert row["comm_fraction"] == pytest.approx(0.2)
    assert row["weight_fraction"] == pytest.approx(0.5)


def test_cpu_model_rows_predict_comm_from_alpha_beta_and_compare_with_measurement() -> None:
    fits = [{"world": 2, "collective": "all_reduce", "alpha_us": 20.0, "beta_ns_per_byte": 2.0}]
    measured = [{
        "world": 2, "d_model": 32, "tokens": 4, "ref_ms": 1.0, "tp_compute_ms": 0.4,
        "tp_comm_ms": 0.03, "speedup": 2.0,
    }]

    (row,) = rx.cpu_model_rows(fits, measured)

    message_bytes = 4 * 32 * 4  # tokens x d_model x float32
    expected_comm_ms = (20.0 + 2.0e-3 * message_bytes) / 1e3
    assert row["source"] == "measured_cpu_model"
    assert row["predicted_comm_ms"] == pytest.approx(expected_comm_ms)
    assert row["measured_comm_ms"] == 0.03
    assert row["comm_model_ratio"] == pytest.approx(expected_comm_ms / 0.03)
    assert row["predicted_speedup"] == pytest.approx(1.0 / (0.4 + expected_comm_ms))
    assert row["measured_speedup"] == 2.0


def test_cpu_model_rows_skip_worlds_without_a_fit() -> None:
    measured = [{"world": 4, "d_model": 32, "tokens": 4, "ref_ms": 1.0, "tp_compute_ms": 0.4,
                 "tp_comm_ms": 0.03, "speedup": 2.0}]

    assert rx.cpu_model_rows([], measured) == []


def test_gpu_assumption_rows_are_labelled_and_nvlink_beats_pcie() -> None:
    rows = rx.gpu_assumption_rows()

    assert rows and all(r["source"] == "assumption" for r in rows)
    by = {(r["hardware"], r["tokens"], r["world"]): r["predicted_speedup"] for r in rows}
    for (hw, tokens, world), speedup in by.items():
        if hw == "nvlink_class":
            assert speedup >= by[("pcie_class", tokens, world)]


# --- output files and CLI ----------------------------------------------------------------
def test_write_results_creates_json_with_meta_and_csv(tmp_path: Path) -> None:
    rx.write_results("exp_x", [{"a": 1, "b": 2.5}], {"smoke": True}, tmp_path)

    data = json.loads((tmp_path / "exp_x.json").read_text())
    assert data == {"meta": {"smoke": True}, "rows": [{"a": 1, "b": 2.5}]}
    assert list(csv.DictReader((tmp_path / "exp_x.csv").open()))[0] == {"a": "1", "b": "2.5"}


def test_write_results_skips_csv_for_empty_rows(tmp_path: Path) -> None:
    rx.write_results("exp_empty", [], {}, tmp_path)

    assert (tmp_path / "exp_empty.json").exists()
    assert not (tmp_path / "exp_empty.csv").exists()


def test_cli_rejects_unknown_experiment() -> None:
    with pytest.raises(SystemExit):
        rx.main(["--exp", "Z"])


def test_smoke_run_of_all_experiments_writes_smoke_suffixed_files(tmp_path: Path) -> None:
    rx.main(["--exp", "all", "--smoke", "--out", str(tmp_path)])

    names = {p.name for p in tmp_path.iterdir()}
    for stem in ("exp_a", "exp_a_fits", "exp_b", "exp_c"):
        assert f"{stem}_smoke.json" in names, names
    meta = json.loads((tmp_path / "exp_b_smoke.json").read_text())["meta"]
    assert meta["smoke"] is True and meta["backend"] == "gloo" and meta["threads_per_rank"] == 1
    assert meta["config"]["repeats"] == 1
    a_rows = json.loads((tmp_path / "exp_a_smoke.json").read_text())["rows"]
    assert all("latency_spread" in r for r in a_rows)
    assert all(r["max_err"] < 1e-4 for r in json.loads((tmp_path / "exp_b_smoke.json").read_text())["rows"])


def test_experiment_c_alone_runs_its_own_inputs(tmp_path: Path) -> None:
    rx.main(["--exp", "C", "--smoke", "--out", str(tmp_path)])

    rows = json.loads((tmp_path / "exp_c_smoke.json").read_text())["rows"]
    assert {r["source"] for r in rows} == {"measured_cpu_model", "assumption"}


# --- bounded retry for Gloo's intermittent process deaths ---------------------------------
def _died() -> ProcessExitedException:
    return ProcessExitedException("died", 1, 123, exit_code=-6, signal_name="SIGABRT")


def test_spawn_retrying_retries_once_after_a_process_death(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []

    def flaky(fn, world, *args):
        calls.append(1)
        if len(calls) == 1:
            raise _died()
        return ["ok"]

    monkeypatch.setattr(rx, "spawn_group", flaky)

    assert rx._spawn_retrying(print, 2) == ["ok"]
    assert len(calls) == 2


def test_spawn_retrying_gives_up_after_the_attempt_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    def always(fn, world, *args):
        raise _died()

    monkeypatch.setattr(rx, "spawn_group", always)

    with pytest.raises(ProcessExitedException):
        rx._spawn_retrying(print, 2)


def test_spawn_retrying_does_not_hide_python_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []

    def buggy(fn, world, *args):
        calls.append(1)
        raise ValueError("a real bug")

    monkeypatch.setattr(rx, "spawn_group", buggy)

    with pytest.raises(ValueError, match="real bug"):
        rx._spawn_retrying(print, 2)
    assert len(calls) == 1


def test_metadata_records_core_split_and_retries_and_timing_caveat() -> None:
    meta = rx.run_metadata(rx.SMOKE, smoke=True)

    assert {"performance_cores", "efficiency_cores", "spawn_retries"} <= meta.keys()
    assert meta["spawn_retries"] == []
    assert "barrier" in meta["note"]


def test_a_retry_is_recorded_in_the_run_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []

    def flaky(fn, world, *args):
        calls.append(1)
        if len(calls) == 1:
            raise _died()
        return ["ok"]

    monkeypatch.setattr(rx, "spawn_group", flaky)
    rx.SPAWN_RETRY_LOG.clear()

    rx._spawn_retrying(print, 2)

    assert len(rx.run_metadata(rx.SMOKE, smoke=True)["spawn_retries"]) == 1
    rx.SPAWN_RETRY_LOG.clear()


def test_core_split_is_none_when_the_platform_does_not_report_it() -> None:
    assert rx.core_split(lambda name: None) == (None, None)
    assert rx.core_split(lambda name: {"hw.perflevel0.logicalcpu": 4, "hw.perflevel1.logicalcpu": 6}[name]) == (4, 6)
