"""Shared builders for scheduler/simulator tests."""
from __future__ import annotations

from cost_model import CostModel
from request import Request
from scheduler import SchedulerConfig
from simulator import Simulator

CM = CostModel()


def cfg(**kw) -> SchedulerConfig:
    base = dict(
        kind="continuous", policy="fcfs", max_running_seqs=8, token_budget=512,
        chunked_prefill=False, num_kv_blocks=1000, block_size=16,
        kv_mode="on_demand", preempt_for_priority=False,
    )
    base.update(kw)
    return SchedulerConfig(**base)


def req(i: int, arrival: float = 0.0, p: int = 32, o: int = 8, prio: int = 0) -> Request:
    return Request(request_id=i, arrival_time=arrival, prompt_tokens=p, max_output_tokens=o, priority=prio)


def run(config: SchedulerConfig, requests: list[Request]):
    return Simulator(config, CM).run(requests)


def admit_order(result) -> list[int]:
    return [e.request_id for e in result.events if e.kind == "ADMIT"]
