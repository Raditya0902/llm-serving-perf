from __future__ import annotations

import dataclasses
import math

import pytest

from sweep_config import (
    BASELINE_RATE_BRIEF,
    BenchParams,
    ServeParams,
    SweepRun,
    SWEEP_KNOBS,
    build_sweep,
    format_value,
)


def _flat(run: SweepRun) -> dict:
    return {**dataclasses.asdict(run.serve), **dataclasses.asdict(run.bench)}


def test_baseline_matches_brief_commands():
    # Arrange / Act
    runs = build_sweep()
    base = runs[0]
    serve = base.serve_command()
    bench = base.bench_command("out", rep=0)

    # Assert
    assert base.run_id == "baseline"
    assert serve[:3] == ["vllm", "serve", "Qwen/Qwen2.5-0.5B-Instruct"]
    assert serve[serve.index("--max-model-len") + 1] == "4096"
    assert serve[serve.index("--gpu-memory-utilization") + 1] == "0.85"
    assert bench[:4] == ["vllm", "bench", "serve", "--backend"]
    assert bench[bench.index("--random-input-len") + 1] == "512"
    assert bench[bench.index("--random-output-len") + 1] == "128"
    assert bench[bench.index("--num-prompts") + 1] == "100"
    assert bench[bench.index("--request-rate") + 1] == "2.0"
    assert "--save-result" in bench


def test_each_run_differs_from_baseline_in_exactly_one_field():
    runs = build_sweep()
    base = _flat(runs[0])

    assert len(runs) > 1
    for run in runs[1:]:
        diff = {k for k, v in _flat(run).items() if v != base[k]}
        assert diff == {run.knob}, run.run_id


def test_every_knob_in_the_brief_is_swept():
    runs = build_sweep()
    knobs = {r.knob for r in runs[1:]}

    assert knobs == set(SWEEP_KNOBS)
    assert {"max_num_seqs", "max_num_batched_tokens", "gpu_memory_utilization",
            "request_rate", "input_len"} <= knobs


def test_baseline_value_is_not_repeated_inside_a_sweep():
    runs = build_sweep()
    ids = [r.run_id for r in runs]

    assert len(ids) == len(set(ids))


def test_only_filter_keeps_baseline_and_selected_knobs():
    runs = build_sweep(only=["max_num_seqs"])

    assert runs[0].run_id == "baseline"
    assert {r.knob for r in runs[1:]} == {"max_num_seqs"}


def test_unknown_knob_in_only_raises():
    with pytest.raises(ValueError, match="unknown knob"):
        build_sweep(only=["tensor_parallel"])


def test_saturated_load_uses_infinite_request_rate_for_baseline():
    runs = build_sweep(load="saturated")

    assert math.isinf(runs[0].bench.request_rate)
    assert runs[0].bench_command("out", 0)[
        runs[0].bench_command("out", 0).index("--request-rate") + 1] == "inf"
    # request_rate is swept only under the paced (brief) load profile
    assert "request_rate" not in {r.knob for r in runs}


def test_unknown_load_profile_raises():
    with pytest.raises(ValueError, match="load"):
        build_sweep(load="chaos")


def test_model_override_flows_into_both_commands():
    run = build_sweep(model="Qwen/Qwen2.5-1.5B-Instruct")[0]

    assert "Qwen/Qwen2.5-1.5B-Instruct" in run.serve_command()
    bench = run.bench_command("out", 0)
    assert bench[bench.index("--model") + 1] == "Qwen/Qwen2.5-1.5B-Instruct"


def test_bench_command_embeds_self_describing_metadata_and_filename():
    run = [r for r in build_sweep() if r.run_id == "max_num_seqs-16"][0]

    bench = run.bench_command("results/raw", rep=2)

    assert bench[bench.index("--result-filename") + 1] == "max_num_seqs-16__r2.json"
    assert bench[bench.index("--result-dir") + 1] == "results/raw"
    meta = bench[bench.index("--metadata") + 1:]
    assert "sweep_knob=max_num_seqs" in meta
    assert "sweep_run_id=max_num_seqs-16" in meta
    assert "p_max_num_seqs=16" in meta
    assert "p_request_rate=2.0" in meta


def test_ignore_eos_can_be_disabled():
    run = build_sweep(ignore_eos=False)[0]

    assert "--ignore-eos" not in run.bench_command("out", 0)
    assert "--ignore-eos" in build_sweep()[0].bench_command("out", 0)


def test_bench_and_serve_ports_agree():
    run = build_sweep(port=9001)[0]

    assert run.serve_command()[run.serve_command().index("--port") + 1] == "9001"
    assert run.bench_command("o", 0)[run.bench_command("o", 0).index("--port") + 1] == "9001"
    assert run.base_url == "http://127.0.0.1:9001"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"gpu_memory_utilization": 0.0},
        {"gpu_memory_utilization": 1.5},
        {"max_num_seqs": 0},
        {"max_num_batched_tokens": -1},
        {"max_model_len": 0},
        {"port": 70000},
        {"model": ""},
    ],
)
def test_serve_params_validation(kwargs):
    with pytest.raises(ValueError):
        ServeParams(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"request_rate": 0.0},
        {"input_len": 0},
        {"output_len": 0},
        {"num_prompts": 0},
        {"port": 0},
    ],
)
def test_bench_params_validation(kwargs):
    with pytest.raises(ValueError):
        BenchParams(**kwargs)


def test_input_plus_output_must_fit_model_len():
    with pytest.raises(ValueError, match="max_model_len"):
        SweepRun(
            run_id="x", knob="input_len", value="4000",
            serve=ServeParams(max_model_len=4096),
            bench=BenchParams(input_len=4000, output_len=128),
        )


def test_format_value_handles_floats_ints_and_inf():
    assert format_value(16) == "16"
    assert format_value(0.85) == "0.85"
    assert format_value(2.0) == "2"
    assert format_value(float("inf")) == "inf"


def test_brief_rate_constant():
    assert BASELINE_RATE_BRIEF == 2.0


# --- review fixes: loopback binding, model validation, config fingerprint ---------------
def test_server_and_client_are_pinned_to_loopback():
    run = build_sweep()[0]

    serve, bench = run.serve_command(), run.bench_command("out", 0)

    assert serve[serve.index("--host") + 1] == "127.0.0.1"
    assert bench[bench.index("--host") + 1] == "127.0.0.1"


@pytest.mark.parametrize(
    "model", ["--host=0.0.0.0", "-m", "two words", "bad\nname", "a;b", "$(x)", ""]
)
def test_model_must_not_look_like_an_option_or_contain_odd_characters(model):
    with pytest.raises(ValueError, match="model"):
        ServeParams(model=model)


@pytest.mark.parametrize(
    "model", ["Qwen/Qwen2.5-0.5B-Instruct", "/models/qwen-0.5b", "org/name_v1.2", "./local"]
)
def test_ordinary_model_names_and_paths_are_accepted(model):
    assert ServeParams(model=model).model == model


def test_config_id_is_stable_and_ignores_the_port():
    a = build_sweep(port=8000)[0].config_id
    b = build_sweep(port=9001)[0].config_id

    assert a == b and len(a) == 12


@pytest.mark.parametrize(
    "kwargs",
    [
        {"load": "saturated"},
        {"model": "Qwen/Qwen2.5-1.5B-Instruct"},
        {"num_prompts": 50},
        {"seed": 1},
        {"ignore_eos": False},
    ],
)
def test_config_id_changes_when_any_experiment_setting_changes(kwargs):
    assert build_sweep(**kwargs)[0].config_id != build_sweep()[0].config_id


def test_config_id_differs_between_runs_of_one_sweep():
    runs = build_sweep()

    assert len({r.config_id for r in runs}) == len(runs)


def test_metadata_records_config_id_and_every_setting():
    run = build_sweep(num_prompts=77, seed=5)[0]

    meta = run.bench_command("o", 0)
    meta = meta[meta.index("--metadata") + 1:]

    assert f"sweep_config_id={run.config_id}" in meta
    assert "p_num_prompts=77" in meta
    assert "p_seed=5" in meta
    assert "p_ignore_eos=True" in meta
