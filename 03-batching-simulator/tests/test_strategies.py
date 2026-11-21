from __future__ import annotations

import pytest

from metrics import summarize
from simulator import Simulator
from strategies import STRATEGY_NAMES, StrategyParams, strategy_configs
from workload import WorkloadConfig, generate_workload


def test_four_named_strategies() -> None:
    assert STRATEGY_NAMES == ("sequential", "static", "continuous", "continuous_chunked")
    cfgs = strategy_configs(StrategyParams())
    assert tuple(cfgs) == STRATEGY_NAMES
    assert cfgs["sequential"].max_running_seqs == 1
    assert cfgs["static"].kind == "static"
    assert cfgs["continuous"].chunked_prefill is False
    assert cfgs["continuous_chunked"].chunked_prefill is True
    assert cfgs["continuous_chunked"].token_budget < cfgs["continuous"].token_budget


def test_invalid_params_rejected() -> None:
    with pytest.raises(ValueError):
        strategy_configs(StrategyParams(chunked_budget=8, max_running_seqs=32))


def test_full_workload_completes_for_every_strategy_and_orders_sensibly() -> None:
    wl = WorkloadConfig(num_requests=40, seed=1)
    out = {
        name: summarize(Simulator(c).run(generate_workload(wl)))
        for name, c in strategy_configs(StrategyParams()).items()
    }
    assert all(s["n_requests"] == 40 for s in out.values())
    assert out["continuous"]["throughput_tok_s"] > out["sequential"]["throughput_tok_s"]
    assert out["continuous"]["mean_latency_ms"] < out["sequential"]["mean_latency_ms"]
    assert out["continuous_chunked"]["worst_itl_p95_ms"] <= out["continuous"]["worst_itl_p95_ms"]
