import json

import pytest

from run_local_experiments import (
    BUILDERS,
    SMOKE,
    main,
    parse_args,
    run_experiment,
    selected_experiments,
    smoke_context,
    values_and_baseline,
)
from experiments import get_experiment

import torch


@pytest.fixture(scope="module")
def ctx():
    return smoke_context(torch.device("cpu"), SMOKE)


def test_every_local_experiment_has_a_builder():
    local = {e.exp_id for e in selected_experiments(["all"])}
    assert local == set(BUILDERS) == {"E1", "E2", "E3", "O1", "O2"}


def test_gpu_experiments_are_rejected_with_a_pointer_to_the_runbook():
    with pytest.raises(SystemExit, match="GPU"):
        selected_experiments(["E4"])


def test_smoke_overrides_keep_baseline_inside_values():
    for exp_id in SMOKE.overrides:
        values, baseline = values_and_baseline(get_experiment(exp_id), SMOKE)
        assert baseline in values


def test_e1_rows_cover_each_batch_size_and_match_single_request_tokens(ctx):
    rows = run_experiment(get_experiment("E1"), ctx)

    assert [r["label"] for r in rows] == ["1", "2", "4"]
    assert all(r["tokens_match"] == 1.0 for r in rows)
    assert all(r["n_requests"] == 4 for r in rows)
    assert [r["is_baseline"] for r in rows] == [True, False, False]
    # queueing: with B=1 the last request waits for three others, so P95 TTFT is highest
    assert rows[0]["ttft_ms_p95"] > rows[2]["ttft_ms_p95"]


def test_e2_and_e3_are_deterministic(ctx):
    for exp_id in ("E2", "E3"):
        rows = run_experiment(get_experiment(exp_id), ctx)
        assert all(r["tokens_match"] == 1.0 for r in rows), exp_id


def test_o1_deferred_sync_matches_day1_tokens(ctx):
    rows = {r["label"]: r for r in run_experiment(get_experiment("O1"), ctx)}

    assert rows["deferred_sync"]["tokens_match"] == 1.0
    assert rows["day1"]["is_baseline"] is True
    assert rows["day1_control"]["is_control"] is True
    assert rows["day1_control"]["tokens_match"] == 1.0
    assert rows["deferred_sync"]["is_control"] is False
    assert all(r["valid"] for r in rows.values())
    assert rows["day1"]["n_reps"] == 3  # 2 requested repeats rounded up to a multiple of 3 cases


def test_o2_compares_sdpa_with_eager(ctx):
    rows = {r["label"]: r for r in run_experiment(get_experiment("O2"), ctx)}

    assert set(rows) == {"sdpa", "eager", "sdpa_control"}
    assert rows["eager"]["tokens_match"] == 1.0


def test_e1_rejects_batch_size_that_does_not_divide_requests(ctx):
    from dataclasses import replace

    bad = replace(ctx, cfg=replace(ctx.cfg, n_requests=5))
    with pytest.raises(ValueError, match="divisible"):
        run_experiment(get_experiment("E1"), bad)


def test_main_smoke_writes_json_with_hypothesis_and_provenance(tmp_path):
    code = main(["--exp", "O1", "--smoke", "--device", "cpu", "--tag", "t", "--out", str(tmp_path)])

    assert code == 0
    payload = json.loads((tmp_path / "t" / "exp_o1.json").read_text())
    assert payload["meta"]["experiment"]["hypothesis"]
    assert payload["meta"]["smoke"] is True
    assert {r["label"] for r in payload["rows"]} == {"day1", "day1_control", "deferred_sync"}
    assert (tmp_path / "t" / "exp_o1.csv").is_file()


def test_parse_args_defaults():
    args = parse_args([])
    assert args.exp == ["all"] and args.smoke is False


def test_values_override_requires_one_integer_experiment_and_the_baseline():
    from run_local_experiments import values_override

    e2 = get_experiment("E2")
    assert values_override([e2], [64, 512]) == {"E2": ((64, 512), 512)}
    with pytest.raises(SystemExit, match="baseline"):
        values_override([e2], [64, 128])
    with pytest.raises(SystemExit, match="exactly one"):
        values_override([e2, get_experiment("E3")], [512])
    with pytest.raises(SystemExit, match="integer"):
        values_override([get_experiment("O1")], [1])


def test_experiment_selection_rejects_unknown_ids_and_mixed_all():
    with pytest.raises(SystemExit, match="unknown"):
        selected_experiments(["E99"])
    with pytest.raises(SystemExit, match="cannot be combined"):
        selected_experiments(["all", "E1"])


@pytest.mark.parametrize("flag", [["--repeats", "0"], ["--repeats", "-2"], ["--warmup", "-1"]])
def test_main_rejects_invalid_repeat_and_warmup_before_loading_a_model(flag, tmp_path):
    with pytest.raises(SystemExit):
        main(["--exp", "O1", "--smoke", "--device", "cpu", "--out", str(tmp_path), *flag])


def test_values_override_rejects_duplicates_and_non_positive():
    from run_local_experiments import values_override

    e2 = get_experiment("E2")
    with pytest.raises(SystemExit, match="positive and unique"):
        values_override([e2], [512, 512])
    with pytest.raises(SystemExit, match="positive and unique"):
        values_override([e2], [512, 0])


def test_e1_rejects_indivisible_batch_sizes_before_any_work(ctx):
    from dataclasses import replace

    bad = replace(ctx, cfg=replace(ctx.cfg, n_requests=5, overrides={"E1": ((1, 2), 1)}))
    with pytest.raises(ValueError, match="divisible"):
        BUILDERS["E1"](bad, get_experiment("E1"))


def test_a_diverged_row_is_marked_invalid(ctx, monkeypatch):
    import run_local_experiments as mod

    def always_diverged(baseline):
        return lambda label, collected: 1.0 if label == baseline else 0.5

    monkeypatch.setitem(mod.BUILDERS, "O1", lambda c, e: mod.CaseSet(
        {"day1": lambda: [mod.run_day1(c.get_model("sdpa"), c.make_ids(8), 4, c.eos_ids)],
         "variant": lambda: [mod.run_day1(c.get_model("sdpa"), c.make_ids(8), 4, c.eos_ids)]},
        "day1", 1, always_diverged("day1")))

    rows = {r["label"]: r for r in run_experiment(get_experiment("O1"), ctx)}

    assert rows["variant"]["valid"] is False
    assert rows["day1"]["valid"] is True
