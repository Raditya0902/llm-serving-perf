from __future__ import annotations

import pytest

from helpers import cfg


@pytest.mark.parametrize("kw", [
    dict(kind="bogus"), dict(policy="bogus"), dict(kv_mode="bogus"),
    dict(max_running_seqs=0), dict(token_budget=0), dict(num_kv_blocks=0), dict(block_size=0),
    dict(max_running_seqs=512, token_budget=512),  # decode alone would eat the whole budget
    dict(preempt_for_priority=True, policy="fcfs"),
    dict(kind="static", chunked_prefill=True),
])
def test_invalid_configs_rejected(kw) -> None:
    with pytest.raises(ValueError):
        cfg(**kw)


def test_valid_config_is_frozen() -> None:
    c = cfg()
    with pytest.raises(Exception):
        c.max_running_seqs = 3  # type: ignore[misc]
