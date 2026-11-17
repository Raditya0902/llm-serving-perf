"""The four strategies compared in experiment A, as scheduler configs.

sequential         one request at a time, run to completion (FCFS)
static             batch up to N requests, drain the whole batch, then form the next
continuous         iteration-level batching, whole-prompt prefill (large token budget)
continuous_chunked iteration-level batching, prompts prefilled in chunks under a small
                   token budget so decoders keep getting a token every iteration
"""
from __future__ import annotations

from dataclasses import dataclass

from scheduler import SchedulerConfig

STRATEGY_NAMES = ("sequential", "static", "continuous", "continuous_chunked")


@dataclass(frozen=True)
class StrategyParams:
    max_running_seqs: int = 32
    unchunked_budget: int = 4096
    chunked_budget: int = 512
    num_kv_blocks: int = 3072
    block_size: int = 16


def strategy_configs(p: StrategyParams) -> dict[str, SchedulerConfig]:
    common = dict(num_kv_blocks=p.num_kv_blocks, block_size=p.block_size)
    return {
        "sequential": SchedulerConfig(max_running_seqs=1, token_budget=p.unchunked_budget, **common),
        "static": SchedulerConfig(
            kind="static", max_running_seqs=p.max_running_seqs, token_budget=p.unchunked_budget, **common
        ),
        "continuous": SchedulerConfig(
            max_running_seqs=p.max_running_seqs, token_budget=p.unchunked_budget, **common
        ),
        "continuous_chunked": SchedulerConfig(
            max_running_seqs=p.max_running_seqs, token_budget=p.chunked_budget,
            chunked_prefill=True, **common,
        ),
    }
