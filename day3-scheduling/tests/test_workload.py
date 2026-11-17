from __future__ import annotations

from workload import WorkloadConfig, clone_requests, generate_workload, long_prompt_scenario


def test_generation_is_seeded_and_deterministic() -> None:
    a = [(r.arrival_time, r.prompt_tokens, r.max_output_tokens, r.priority) for r in generate_workload(WorkloadConfig(seed=3))]
    b = [(r.arrival_time, r.prompt_tokens, r.max_output_tokens, r.priority) for r in generate_workload(WorkloadConfig(seed=3))]
    c = [(r.arrival_time, r.prompt_tokens, r.max_output_tokens, r.priority) for r in generate_workload(WorkloadConfig(seed=4))]
    assert a == b and a != c


def test_default_workload_has_100_requests_sorted_by_arrival_with_unique_ids() -> None:
    reqs = generate_workload(WorkloadConfig())
    assert len(reqs) == 100
    assert [r.arrival_time for r in reqs] == sorted(r.arrival_time for r in reqs)
    assert len({r.request_id for r in reqs}) == 100


def test_workload_mixes_short_and_long_prompts_and_outputs_and_priorities() -> None:
    reqs = generate_workload(WorkloadConfig())
    prompts = [r.prompt_tokens for r in reqs]
    outputs = [r.max_output_tokens for r in reqs]
    assert min(prompts) <= 256 and max(prompts) >= 1024
    assert min(outputs) <= 64 and max(outputs) >= 200
    assert {r.priority for r in reqs} == {0, 1}
    assert len({r.arrival_time for r in reqs}) > 50  # different arrival times


def test_workload_has_a_burst() -> None:
    cfg = WorkloadConfig(burst_frac=0.3, burst_window_ms=200.0)
    reqs = generate_workload(cfg)
    in_window = [r for r in reqs if cfg.burst_start_ms() <= r.arrival_time <= cfg.burst_start_ms() + 200.0]
    assert len(in_window) >= 30


def test_clone_requests_returns_fresh_equivalents() -> None:
    reqs = generate_workload(WorkloadConfig(num_requests=5))
    reqs[0].generated_tokens = 3
    clones = clone_requests(reqs)
    assert clones[0].generated_tokens == 0
    assert [(c.request_id, c.prompt_tokens) for c in clones] == [(r.request_id, r.prompt_tokens) for r in reqs]


def test_long_prompt_scenario_places_long_prompts_after_decoders_start() -> None:
    reqs = long_prompt_scenario(num_decoders=4, long_prompt_tokens=2000, long_arrivals_ms=(300.0, 900.0))
    decoders = [r for r in reqs if r.prompt_tokens < 100]
    longs = [r for r in reqs if r.prompt_tokens == 2000]
    assert len(decoders) == 4 and len(longs) == 2
    assert min(l.arrival_time for l in longs) > max(d.arrival_time for d in decoders)
