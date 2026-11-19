"""The 12-case matrix definition and the vLLM commands it generates."""
from __future__ import annotations

import dataclasses
import itertools

import pytest

import matrix as mx

SHELL_METACHARS = set(";|&$`<>(){}\\\"' \n")


def test_matrix_has_twelve_distinct_cases():
    cases = mx.build_matrix()
    assert len(cases) == 12
    assert len({c.case_id for c in cases}) == 12
    combos = {(c.workload.key, c.concurrency, c.config.name) for c in cases}
    assert combos == set(itertools.product("AB", mx.CONCURRENCY_LEVELS, ("baseline", "tuned")))


def test_three_concurrency_levels_and_two_configs():
    assert mx.CONCURRENCY_LEVELS == (4, 16, 64)
    assert [c.name for c in mx.CONFIGS.values()] == ["baseline", "tuned"]


def test_tuned_differs_from_baseline_in_exactly_the_two_documented_knobs():
    base, tuned = mx.CONFIGS["baseline"], mx.CONFIGS["tuned"]
    changed = {
        f.name
        for f in dataclasses.fields(base)
        if f.name != "name" and getattr(base, f.name) != getattr(tuned, f.name)
    }
    assert changed == {"gpu_memory_utilization", "max_num_batched_tokens"}


def test_max_num_seqs_is_never_binding_in_the_matrix():
    for cfg in mx.CONFIGS.values():
        assert cfg.max_num_seqs > max(mx.CONCURRENCY_LEVELS)


@pytest.mark.parametrize("key", ["A", "B"])
def test_worst_case_sequence_fits_max_model_len(key):
    w = mx.WORKLOADS[key]
    assert w.max_input_len + w.max_output_len <= mx.MAX_MODEL_LEN


def test_workload_b_is_much_longer_than_a():
    a, b = mx.WORKLOADS["A"], mx.WORKLOADS["B"]
    assert b.input_len >= 8 * a.input_len and b.output_len > a.output_len


def test_workload_rejects_sequences_over_max_model_len():
    with pytest.raises(ValueError, match="max_model_len"):
        mx.WorkloadSpec("X", "too long", 3800, 400, 0.25, "x")


def test_num_prompts_scales_with_concurrency():
    by_conc = {c.concurrency: c.num_prompts for c in mx.build_matrix()}
    assert by_conc[4] == 100 and by_conc[16] == 128 and by_conc[64] == 512


def test_bench_command_is_closed_loop_with_p95():
    case = next(c for c in mx.build_matrix() if c.case_id == "B-c16-tuned")
    cmd = case.bench_command("out", rep=2, port=8123)
    flag = lambda name: cmd[cmd.index(name) + 1]
    assert flag("--max-concurrency") == "16"
    assert flag("--request-rate") == "inf"
    assert flag("--metric-percentiles") == "50,95,99"
    assert flag("--random-input-len") == "2560" and flag("--random-output-len") == "384"
    assert flag("--random-range-ratio") == "0.25"
    assert flag("--port") == "8123"
    assert "--save-detailed" in cmd and "--ignore-eos" in cmd and "--save-result" in cmd
    assert flag("--result-filename") == "B-c16-tuned__r2.json"
    assert cmd.index("--metadata") > cmd.index("--save-result")  # nargs="*" must come last


def test_bench_seed_differs_per_rep():
    case = mx.build_matrix()[0]
    seeds = {case.bench_command("o", rep=r, port=8000)[
        case.bench_command("o", rep=r, port=8000).index("--seed") + 1] for r in range(3)}
    assert len(seeds) == 3


def test_serve_command_carries_the_config_knobs_and_binds_loopback():
    case = next(c for c in mx.build_matrix() if c.case_id == "A-c4-tuned")
    cmd = case.serve_command(port=8000)
    assert cmd[:3] == ["vllm", "serve", case.model]
    flag = lambda name: cmd[cmd.index(name) + 1]
    assert flag("--gpu-memory-utilization") == "0.92"
    assert flag("--max-num-batched-tokens") == "2048"
    assert flag("--max-model-len") == str(mx.MAX_MODEL_LEN)
    assert flag("--host") == "127.0.0.1"


def test_commands_contain_no_shell_metacharacters_in_values():
    for case in mx.build_matrix():
        for tok in case.serve_command(8000) + case.bench_command("out", 0, 8000):
            if tok.startswith("-"):
                continue
            assert not (set(tok) & SHELL_METACHARS), tok


def test_config_id_ignores_port_and_rep_but_tracks_knobs():
    case = mx.build_matrix()[0]
    other = dataclasses.replace(case, config=dataclasses.replace(case.config, max_num_batched_tokens=4096))
    assert case.config_id == mx.build_matrix()[0].config_id
    assert case.config_id != other.config_id


def test_model_must_be_a_safe_hub_id():
    with pytest.raises(ValueError, match="model"):
        mx.build_matrix(model="--evil; rm -rf /")


def test_build_runs_counts_key_cases_and_aa_control():
    runs = mx.build_runs(reps=3, key_reps=5)
    key = [r for r in runs if r.case.concurrency == max(mx.CONCURRENCY_LEVELS) and not r.case.is_control]
    aa = [r for r in runs if r.case.is_control]
    assert len(key) == 4 * 5
    assert len(aa) == 5
    assert len(runs) == 8 * 3 + 4 * 5 + 5
    assert len({r.run_id for r in runs}) == len(runs)


def test_aa_control_is_baseline_b16_with_same_config_id():
    control = next(r.case for r in mx.build_runs(3, 5) if r.case.is_control)
    original = next(c for c in mx.build_matrix() if c.case_id == "B-c16-baseline")
    assert control.case_id == "B-c16-baseline-aa"
    assert control.config_id == original.config_id


def test_build_runs_can_skip_the_control():
    assert not any(r.case.is_control for r in mx.build_runs(3, 5, include_control=False))


def test_build_runs_rejects_nonpositive_reps():
    with pytest.raises(ValueError, match="reps"):
        mx.build_runs(reps=0, key_reps=5)
    with pytest.raises(ValueError, match="key_reps"):
        mx.build_runs(reps=3, key_reps=2)


def test_capacity_pressure_only_for_workload_b_at_highest_concurrency():
    pressured = {c.case_id for c in mx.build_matrix() if c.predicted_pressure()["pressured"]}
    assert pressured == {"B-c64-baseline", "B-c64-tuned"}


# --- KV-pool cap: run on a larger GPU while reproducing the modeled (smaller) GPU's pool ---------

KV_CAP_FLAG = "--kv-cache-memory-bytes"


def _capped_case(case_id: str, **overrides) -> mx.MatrixCase:
    return next(c for c in mx.build_matrix(cap_kv_pool=True, **overrides) if c.case_id == case_id)


def test_serve_command_has_no_kv_cap_by_default():
    case = next(c for c in mx.build_matrix() if c.case_id == "B-c64-baseline")
    assert KV_CAP_FLAG not in case.serve_command(port=8000)
    assert case.kv_cache_cap_bytes is None


def test_cap_pins_pool_to_the_modeled_hardware_pool():
    case = _capped_case("B-c64-baseline")
    model = mx.kv_sizing.model_spec(case.model_preset)
    hardware = mx.kv_sizing.HARDWARE[case.hardware]
    blocks = mx.kv_sizing.kv_pool_blocks(model, hardware, case.config.gpu_memory_utilization)
    expected = blocks * mx.kv_sizing.BLOCK_SIZE * mx.kv_sizing.kv_bytes_per_token_for(model)

    cmd = case.serve_command(port=8000)

    assert cmd[cmd.index(KV_CAP_FLAG) + 1] == str(expected)
    assert case.kv_cache_cap_bytes == expected


def test_tuned_cap_is_larger_than_baseline_cap():
    baseline = _capped_case("B-c64-baseline").kv_cache_cap_bytes
    tuned = _capped_case("B-c64-tuned").kv_cache_cap_bytes
    assert baseline is not None and tuned is not None
    assert tuned > baseline


def test_cap_value_is_plain_digits():
    cmd = _capped_case("A-c4-tuned").serve_command(port=8000)
    assert cmd[cmd.index(KV_CAP_FLAG) + 1].isdigit()


def test_cap_changes_config_id_so_capped_and_uncapped_runs_never_mix():
    plain = next(c for c in mx.build_matrix() if c.case_id == "B-c64-baseline")
    capped = _capped_case("B-c64-baseline")
    assert plain.config_id != capped.config_id


def test_cap_is_rejected_when_the_modeled_pool_is_empty():
    with pytest.raises(ValueError, match="KV pool"):
        mx.build_matrix(model_preset="llama3-70b", hardware="a100-40gb", cap_kv_pool=True)


def test_extra_serve_args_are_appended_and_default_to_none():
    base = mx.MatrixCase(mx.WORKLOADS["B"], mx.CONFIGS["baseline"], 16)
    tp2 = mx.ServingConfig("tp2", 0.85, 8192, extra_serve_args=("--tensor-parallel-size", "2"))
    case = mx.MatrixCase(mx.WORKLOADS["B"], tp2, 16)
    assert "--tensor-parallel-size" not in base.serve_command(8000)
    cmd = case.serve_command(8000)
    assert cmd[cmd.index("--tensor-parallel-size") + 1] == "2"


def test_tensor_parallel_property_reads_the_serve_flag_and_defaults_to_one():
    assert mx.ServingConfig("x", 0.85, 8192).tensor_parallel == 1
    assert mx.ServingConfig("x", 0.85, 8192, extra_serve_args=("--tensor-parallel-size", "2")).tensor_parallel == 2
    nc = mx.ServingConfig("x", 0.85, 8192, extra_serve_args=("--tensor-parallel-size", "2", "--disable-custom-all-reduce"))
    assert nc.tensor_parallel == 2


def test_tp2_cap_is_the_per_gpu_modeled_pool_not_the_single_gpu_pool():
    import multi_gpu
    tp2 = mx.ServingConfig("tp2", 0.85, 8192, extra_serve_args=("--tensor-parallel-size", "2"))
    case = mx.MatrixCase(mx.WORKLOADS["B"], tp2, 64, hardware="a100-40gb", cap_kv_pool=True)
    blocks, block_bytes = multi_gpu.kv_pool_per_gpu("llama3-8b", "a100-40gb", 0.85, tp=2)
    assert case.kv_cache_cap_bytes == blocks * block_bytes
    cmd = case.serve_command(port=8000)
    assert cmd[cmd.index(KV_CAP_FLAG) + 1] == str(blocks * block_bytes)
    tp1 = mx.MatrixCase(mx.WORKLOADS["B"], mx.ServingConfig("tp1", 0.85, 8192), 64,
                        hardware="a100-40gb", cap_kv_pool=True)
    assert case.kv_cache_cap_bytes > tp1.kv_cache_cap_bytes  # more room per GPU once weights are sharded


def test_tp1_cap_value_is_unchanged_by_tp_support():
    case = _capped_case("B-c64-baseline")
    assert case.config.tensor_parallel == 1
    model = mx.kv_sizing.model_spec(case.model_preset)
    blocks = mx.kv_sizing.kv_pool_blocks(model, mx.kv_sizing.HARDWARE[case.hardware], 0.85)
    assert case.kv_cache_cap_bytes == blocks * mx.kv_sizing.BLOCK_SIZE * mx.kv_sizing.kv_bytes_per_token_for(model)
