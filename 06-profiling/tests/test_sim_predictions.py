import json

import pytest

from sim_predictions import (
    BASELINE_BUDGET,
    BASELINE_SEQS,
    SIM_NOTE,
    SMOKE,
    case_configs,
    main,
    run_sim_experiment,
)


def test_case_configs_pin_one_knob_at_a_time_around_the_day4_baseline():
    e4 = {label: cfg for label, cfg, _ in case_configs("E4", SMOKE)}
    e5 = {label: cfg for label, cfg, _ in case_configs("E5", SMOKE)}

    assert all(c.max_running_seqs == BASELINE_SEQS and c.chunked_prefill for c in e4.values())
    assert {c.token_budget for c in e5.values()} == {BASELINE_BUDGET}
    assert [lbl for lbl, _, base in case_configs("E4", SMOKE) if base] == [str(BASELINE_BUDGET)]


def test_case_configs_rejects_experiments_without_a_simulator_mapping():
    with pytest.raises(KeyError):
        case_configs("E1", SMOKE)


@pytest.mark.parametrize("exp_id", ["E4", "E5", "E6"])
def test_rows_are_labelled_simulated_and_have_exactly_one_baseline(exp_id):
    rows = run_sim_experiment(exp_id, SMOKE)

    assert all(r["simulated"] is True and r["note"] == SIM_NOTE for r in rows)
    assert sum(r["is_baseline"] for r in rows) == 1
    assert all(r["n_seeds"] == 2 for r in rows)


def test_small_kv_pool_causes_more_preemptions_than_an_ample_one():
    rows = {r["label"]: r for r in run_sim_experiment("E6", SMOKE)}

    assert rows["512"]["preemptions"] >= rows["8192"]["preemptions"]
    assert rows["8192"]["kv_peak_frac"] <= 1.0


def test_main_smoke_writes_files_with_simulated_marker(tmp_path):
    assert main(["--exp", "E5", "--smoke", "--out", str(tmp_path)]) == 0

    payload = json.loads((tmp_path / "sim_e5.json").read_text())
    assert payload["meta"]["simulated"] is True
    assert (tmp_path / "sim_e5.csv").is_file()
