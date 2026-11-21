"""Discrete-event engine: runs a scheduler against a cost model.

Loop (time `t` in ms):
    1. move requests with arrival_time <= t into the waiting queue
    2. if nothing is running or waiting, jump `t` to the next arrival (idle time)
    3. ask the scheduler for an IterationPlan
    4. duration = CostModel.iteration_time(prefill, decode, decode context)
    5. advance `t`; apply progress at the *end* of the iteration:
         - prefill chunks add KV; a finished prefill emits the first token (TTFT)
         - each decode adds one KV token and emits one token
         - requests that reached max_output_tokens finish and free their KV

Static batching returns each request as soon as *its* last token is emitted
(streaming semantics), which is generous to the static baseline: a
non-streaming server would only answer when the whole batch is done.
"""
from __future__ import annotations

from dataclasses import dataclass

from cost_model import CostModel
from request import WAITING, Request
from scheduler import Event, EngineState, IterationPlan, SchedulerConfig, make_scheduler


@dataclass(frozen=True)
class IterationRecord:
    start_ms: float
    duration_ms: float
    prefill_tokens: int
    decode_tokens: int  # live decoding sequences (excludes padding)
    padded_slots: int
    running: int  # occupied batch slots after admission (incl. held static slots)
    waiting: int
    kv_used_blocks: int
    prefill_ids: tuple[tuple[int, int], ...]  # (request_id, chunk tokens)
    decode_ids: tuple[int, ...]


@dataclass(frozen=True)
class SimResult:
    config: SchedulerConfig
    cost_model: CostModel
    requests: tuple[Request, ...]
    iterations: tuple[IterationRecord, ...]
    events: tuple[Event, ...]
    first_arrival: float
    end_time: float
    preemptions: int
    kv_peak_blocks: int
    kv_failures: int
    kv_final_used_blocks: int


class Simulator:
    def __init__(self, config: SchedulerConfig, cost_model: CostModel | None = None) -> None:
        self.config = config
        self.cost_model = cost_model or CostModel()

    def validate(self, requests: list[Request]) -> None:
        cfg = self.config
        for r in requests:
            if r.status != WAITING or r.generated_tokens or r.kv_tokens:
                raise ValueError(f"request {r.request_id} is not fresh; build new Request objects per run")
            blocks = -(-r.peak_kv_tokens // cfg.block_size)
            if blocks > cfg.num_kv_blocks:
                raise ValueError(f"request {r.request_id} needs {blocks} KV blocks > pool of {cfg.num_kv_blocks}")
            if not cfg.chunked_prefill and r.peak_kv_tokens > cfg.token_budget:
                raise ValueError(
                    f"request {r.request_id}: without chunked prefill its prompt (or recompute) of up to "
                    f"{r.peak_kv_tokens} tokens must fit the token budget {cfg.token_budget}"
                )

    def run(self, requests: list[Request]) -> SimResult:
        self.validate(requests)
        pending = sorted(requests, key=lambda r: (r.arrival_time, r.request_id))
        state = EngineState(self.config)
        scheduler = make_scheduler(self.config)
        records: list[IterationRecord] = []
        t, nxt = 0.0, 0

        while nxt < len(pending) or state.waiting or state.running:
            state.now = t
            while nxt < len(pending) and pending[nxt].arrival_time <= t:
                state.arrive(pending[nxt])
                nxt += 1
            if not state.waiting and not state.running:
                t = pending[nxt].arrival_time
                continue

            plan = scheduler.plan(state)
            if plan.is_empty:
                if nxt < len(pending) and not state.running:
                    t = pending[nxt].arrival_time
                    continue
                raise RuntimeError("scheduler produced no work with requests pending (deadlock)")

            duration = self.cost_model.iteration_time(
                plan.prefill_tokens, plan.billed_decode_tokens, plan.decode_ctx_tokens
            )
            records.append(self._record(t, duration, plan, state))
            t += duration
            self._apply(plan, state, t)
            if not state.running:  # static batch drained: its held slots and KV go back to the pool
                state.release_held()

        first = pending[0].arrival_time if pending else 0.0
        return SimResult(
            config=self.config,
            cost_model=self.cost_model,
            requests=tuple(pending),
            iterations=tuple(records),
            events=tuple(state.events),
            first_arrival=first,
            end_time=t,
            preemptions=state.preemptions,
            kv_peak_blocks=state.kv.peak_used_blocks,
            kv_failures=state.kv.failures,
            kv_final_used_blocks=state.kv.used_blocks,
        )

    @staticmethod
    def _record(t: float, duration: float, plan: IterationPlan, state: EngineState) -> IterationRecord:
        return IterationRecord(
            start_ms=t,
            duration_ms=duration,
            prefill_tokens=plan.prefill_tokens,
            decode_tokens=len(plan.decode),
            padded_slots=plan.padded_slots,
            running=len(state.running) + len(state.held),
            waiting=len(state.waiting),
            kv_used_blocks=state.kv.used_blocks,
            prefill_ids=tuple((w.request.request_id, w.tokens) for w in plan.prefill),
            decode_ids=tuple(r.request_id for r in plan.decode),
        )

    @staticmethod
    def _emit_token(r: Request, state: EngineState, end: float) -> None:
        r.generated_tokens += 1
        if r.first_token_time is None:
            r.first_token_time = end
        if r.last_token_time is not None:
            r.token_gaps.append(end - r.last_token_time)
        r.last_token_time = end
        if r.is_done:
            state.finish(r, end)

    def _apply(self, plan: IterationPlan, state: EngineState, end: float) -> None:
        for work in plan.prefill:
            r = work.request
            r.kv_tokens += work.tokens
            if not r.is_prefilling:  # prompt (or recompute) complete -> first/next token
                self._emit_token(r, state, end)
        for r in plan.decode:
            r.kv_tokens += 1
            self._emit_token(r, state, end)
