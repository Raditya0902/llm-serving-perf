from __future__ import annotations

import pytest

from request import FINISHED, RUNNING, WAITING, Request


def make(**kw) -> Request:
    args = dict(request_id=1, arrival_time=0.0, prompt_tokens=10, max_output_tokens=5)
    args.update(kw)
    return Request(**args)


def test_new_request_starts_waiting_with_zero_progress() -> None:
    r = make()
    assert r.status == WAITING == "WAITING"
    assert r.generated_tokens == 0
    assert r.kv_tokens == 0
    assert r.prefill_goal == 10
    assert r.is_prefilling


def test_peak_kv_tokens_excludes_last_generated_token() -> None:
    assert make(prompt_tokens=10, max_output_tokens=5).peak_kv_tokens == 14


@pytest.mark.parametrize("kw", [
    dict(prompt_tokens=0), dict(max_output_tokens=0), dict(arrival_time=-1.0),
])
def test_invalid_fields_raise_value_error(kw) -> None:
    with pytest.raises(ValueError):
        make(**kw)


def test_reset_for_recompute_extends_prefill_goal_by_generated_tokens() -> None:
    r = make()
    r.status = RUNNING
    r.kv_tokens = 12
    r.generated_tokens = 3
    wasted = r.reset_for_recompute()
    assert wasted == 12
    assert r.status == WAITING
    assert r.kv_tokens == 0
    assert r.generated_tokens == 3  # tokens already streamed to the user are kept
    assert r.prefill_goal == 13  # prompt + generated
    assert r.preemptions == 1
    assert r.is_prefilling


def test_is_done_when_all_tokens_generated() -> None:
    r = make(max_output_tokens=2)
    r.generated_tokens = 2
    r.status = FINISHED
    assert r.is_done


def test_latency_and_ttft_none_until_set() -> None:
    r = make(arrival_time=5.0)
    assert r.latency is None and r.ttft is None
    r.first_token_time, r.finish_time = 8.0, 20.0
    assert r.ttft == 3.0 and r.latency == 15.0
