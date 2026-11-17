from __future__ import annotations

import pytest

from cost_model import CostModel

CM = CostModel(overhead_ms=0.5, compute_ms_per_token=0.1, weight_read_ms=10.0, kv_read_ms_per_token=0.001)


def test_idle_iteration_costs_nothing() -> None:
    assert CM.iteration_time(0, 0, 0) == 0.0


def test_small_decode_batch_is_memory_bound() -> None:
    # compute = 0.1*4 = 0.4 ; memory = 10 + 0.001*400 = 10.4
    assert CM.iteration_time(0, 4, 400) == pytest.approx(0.5 + 10.4)


def test_large_prefill_is_compute_bound() -> None:
    # compute = 0.1*1000 = 100 ; memory = 10
    assert CM.iteration_time(1000, 0, 0) == pytest.approx(0.5 + 100.0)


def test_small_prefill_piggybacks_on_decode_for_free() -> None:
    base = CM.iteration_time(0, 4, 400)
    assert CM.iteration_time(50, 4, 400) == pytest.approx(base)  # 5.4 compute < 10.4 memory


def test_batching_amortises_weight_reads() -> None:
    per_token_1 = CM.iteration_time(0, 1, 100) / 1
    per_token_32 = CM.iteration_time(0, 32, 3200) / 32
    assert per_token_32 < per_token_1 / 5


def test_negative_inputs_rejected() -> None:
    with pytest.raises(ValueError):
        CM.iteration_time(-1, 0, 0)
