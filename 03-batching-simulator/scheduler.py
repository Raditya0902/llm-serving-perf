"""Scheduling policies: who runs in each engine iteration.

Per iteration a scheduler turns the current `EngineState` into an
`IterationPlan` (which sequences decode, which prompt chunks prefill).

ContinuousScheduler (iteration-level / "continuous" batching)
    1. Make sure every decoding sequence has a KV slot for its next token. In
       `on_demand` KV mode a new block is needed every `block_size` tokens; when
       the pool is exhausted the lowest-ranked running request is preempted by
       recompute (KV freed, progress reset, request re-queued).
    2. Admit waiting requests in policy order while a running slot, KV blocks and
       token budget are available. Strict head-of-line: if the best-ranked waiting
       request cannot be admitted, nothing behind it is (this is FCFS HOL
       blocking; there is no bypass).
    3. Token budget: every decoder costs 1 token; the rest of the budget is spent
       on prefill. Unchunked mode must fit a whole prompt in the leftover budget;
       chunked mode feeds prompts in whatever budget remains each iteration.

StaticScheduler (batch-then-drain baseline)
    When idle, form a batch of up to `max_running_seqs` (full KV reservation),
    prefill it, then decode in lockstep. Slots of finished sequences stay
    occupied (padded) until the whole batch is done; nothing is admitted
    mid-batch.

Ranking: `fcfs` -> (arrival, id); `priority` -> (priority, arrival, id).
The *best* key is served first; the *max* key is the preemption victim.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from kv_blocks import KVBlockManager
from request import FINISHED, RUNNING, WAITING, Request

KINDS = ("continuous", "static")
POLICIES = ("fcfs", "priority")
KV_MODES = ("on_demand", "full_reserve")


@dataclass(frozen=True)
class SchedulerConfig:
    kind: str = "continuous"
    policy: str = "fcfs"
    max_running_seqs: int = 32
    token_budget: int = 4096  # max tokens (decode + prefill) per iteration
    chunked_prefill: bool = False
    num_kv_blocks: int = 3072
    block_size: int = 16
    kv_mode: str = "on_demand"  # static batching always reserves in full
    preempt_for_priority: bool = False

    def __post_init__(self) -> None:
        if self.kind not in KINDS or self.policy not in POLICIES or self.kv_mode not in KV_MODES:
            raise ValueError(f"kind/policy/kv_mode must be in {KINDS}/{POLICIES}/{KV_MODES}")
        if min(self.max_running_seqs, self.token_budget, self.num_kv_blocks, self.block_size) < 1:
            raise ValueError("sizes must be >= 1")
        if self.max_running_seqs >= self.token_budget:
            raise ValueError("max_running_seqs must be < token_budget (decode alone would fill it)")
        if self.preempt_for_priority and self.policy != "priority":
            raise ValueError("preempt_for_priority requires policy='priority'")
        if self.kind == "static" and (self.chunked_prefill or self.preempt_for_priority):
            raise ValueError("static batching supports neither chunked prefill nor preemption")


@dataclass(frozen=True)
class Event:
    time_ms: float
    kind: str  # ARRIVE | ADMIT | PREEMPT | FINISH
    request_id: int


@dataclass(frozen=True)
class PrefillWork:
    request: Request
    tokens: int


@dataclass(frozen=True)
class IterationPlan:
    prefill: tuple[PrefillWork, ...] = ()
    decode: tuple[Request, ...] = ()
    padded_slots: int = 0  # static batching: finished sequences still occupying slots
    padded_ctx_tokens: int = 0

    @property
    def prefill_tokens(self) -> int:
        return sum(w.tokens for w in self.prefill)

    @property
    def is_empty(self) -> bool:
        return not self.prefill and not self.decode

    @property
    def billed_decode_tokens(self) -> int:
        return len(self.decode) + self.padded_slots

    @property
    def decode_ctx_tokens(self) -> int:
        return sum(r.kv_tokens for r in self.decode) + self.padded_ctx_tokens


def rank_key(policy: str) -> Callable[[Request], tuple]:
    if policy == "priority":
        return lambda r: (r.priority, r.arrival_time, r.request_id)
    return lambda r: (r.arrival_time, r.request_id)


@dataclass
class EngineState:
    """Mutable simulator state shared between engine and scheduler."""

    config: SchedulerConfig
    kv: KVBlockManager = field(init=False)
    waiting: list[Request] = field(default_factory=list)
    running: list[Request] = field(default_factory=list)
    held: list[Request] = field(default_factory=list)  # static: finished, slot+KV retained
    events: list[Event] = field(default_factory=list)
    preemptions: int = 0
    now: float = 0.0

    def __post_init__(self) -> None:
        self.kv = KVBlockManager(self.config.num_kv_blocks, self.config.block_size)

    def log(self, kind: str, request: Request) -> None:
        self.events.append(Event(self.now, kind, request.request_id))

    def arrive(self, request: Request) -> None:
        self.waiting.append(request)
        # stamp the true arrival time, not the iteration boundary at which we noticed it
        self.events.append(Event(request.arrival_time, "ARRIVE", request.request_id))

    def admit(self, request: Request, reserve_tokens: int) -> None:
        if not self.kv.try_ensure(request.request_id, reserve_tokens):
            raise RuntimeError("admit() called without checking KV capacity")
        self.waiting.remove(request)
        self.running.append(request)
        request.status = RUNNING
        request.waiting_time += self.now - request.waiting_since
        if request.first_scheduled_time is None:
            request.first_scheduled_time = self.now
        self.log("ADMIT", request)

    def preempt(self, request: Request) -> None:
        self.kv.free(request.request_id)
        self.running.remove(request)
        request.reset_for_recompute()
        request.waiting_since = self.now
        self.waiting.append(request)
        self.preemptions += 1
        self.log("PREEMPT", request)

    def finish(self, request: Request, end_time: float) -> None:
        request.status = FINISHED
        request.finish_time = end_time
        self.running.remove(request)
        if self.config.kind == "static":
            self.held.append(request)  # slot and KV stay until the batch drains
        else:
            self.kv.free(request.request_id)
        self.events.append(Event(end_time, "FINISH", request.request_id))

    def release_held(self) -> None:
        for r in self.held:
            self.kv.free(r.request_id)
        self.held.clear()


class ContinuousScheduler:
    def __init__(self, config: SchedulerConfig) -> None:
        self.config = config
        self.key = rank_key(config.policy)

    def plan(self, state: EngineState) -> IterationPlan:
        self._ensure_decode_kv(state)
        self._admit(state)
        decoders, works = self._allocate(state)
        return IterationPlan(prefill=tuple(works), decode=tuple(decoders))

    # -- helpers --------------------------------------------------------
    def _reserve_tokens(self, r: Request) -> int:
        return r.peak_kv_tokens if self.config.kv_mode == "full_reserve" else r.prefill_goal

    def _ensure_decode_kv(self, state: EngineState) -> None:
        if self.config.kv_mode == "full_reserve":
            return
        decoders = sorted((r for r in state.running if not r.is_prefilling), key=self.key)
        for r in decoders:
            if r.status != RUNNING:  # preempted earlier in this loop
                continue
            while not state.kv.try_ensure(r.request_id, r.kv_tokens + 1):
                victim = max(state.running, key=self.key)
                state.preempt(victim)
                if victim is r:
                    break

    def _allocate(self, state: EngineState) -> tuple[list[Request], list[PrefillWork]]:
        """Split the token budget: decoders first, then prefill in rank order."""
        decoders = sorted((r for r in state.running if not r.is_prefilling), key=self.key)
        budget = self.config.token_budget - len(decoders)
        works: list[PrefillWork] = []
        for r in sorted((r for r in state.running if r.is_prefilling), key=self.key):
            remaining = r.prefill_goal - r.kv_tokens
            chunk = min(remaining, budget) if self.config.chunked_prefill else remaining
            if chunk <= 0 or chunk > budget:
                continue
            works.append(PrefillWork(r, chunk))
            budget -= chunk
        return decoders, works

    def _budget_left(self, state: EngineState) -> int:
        decoders, works = self._allocate(state)
        return self.config.token_budget - len(decoders) - sum(w.tokens for w in works)

    def _admit(self, state: EngineState) -> None:
        cfg = self.config
        while state.waiting:
            head = min(state.waiting, key=self.key)
            left = self._budget_left(state)
            needs = 1 if cfg.chunked_prefill else head.prefill_goal
            if left < needs:
                return
            if not self._can_admit(state, head):
                if cfg.preempt_for_priority and self._preempt_lower_priority(state, head):
                    continue
                return
            state.admit(head, self._reserve_tokens(head))

    def _can_admit(self, state: EngineState, head: Request) -> bool:
        return len(state.running) < self.config.max_running_seqs and state.kv.can_fit(
            head.request_id, self._reserve_tokens(head)
        )

    def _preempt_lower_priority(self, state: EngineState, head: Request) -> bool:
        """Evict the lowest-ranked lower-priority runner, but only if evicting *all* of them
        would actually let `head` in (a free slot and enough KV); otherwise the victim would
        lose its progress for nothing."""
        lower = sorted((r for r in state.running if r.priority > head.priority), key=self.key, reverse=True)
        if not lower:
            return False
        kv = state.kv
        needed = kv.blocks_for(self._reserve_tokens(head)) - kv.blocks_of(head.request_id)
        reclaimable = kv.free_blocks + sum(kv.blocks_of(r.request_id) for r in lower)
        has_slot = len(state.running) - len(lower) < self.config.max_running_seqs
        if not has_slot or reclaimable < needed:
            return False
        state.preempt(lower[0])
        return True


class StaticScheduler:
    def __init__(self, config: SchedulerConfig) -> None:
        self.config = config
        self.key = rank_key(config.policy)

    def plan(self, state: EngineState) -> IterationPlan:
        if not state.running:
            state.release_held()
            self._form_batch(state)
        prefilling = sorted((r for r in state.running if r.is_prefilling), key=self.key)
        if prefilling:
            return IterationPlan(prefill=self._pack_prefill(prefilling))
        if state.running:
            return IterationPlan(
                decode=tuple(sorted(state.running, key=self.key)),
                padded_slots=len(state.held),
                padded_ctx_tokens=sum(r.kv_tokens for r in state.held),
            )
        return IterationPlan()

    def _form_batch(self, state: EngineState) -> None:
        while state.waiting and len(state.running) < self.config.max_running_seqs:
            head = min(state.waiting, key=self.key)
            if not state.kv.can_fit(head.request_id, head.peak_kv_tokens):
                return
            state.admit(head, head.peak_kv_tokens)

    def _pack_prefill(self, prefilling: list[Request]) -> tuple[PrefillWork, ...]:
        works: list[PrefillWork] = []
        used = 0
        for r in prefilling:
            n = r.prefill_goal - r.kv_tokens
            if works and used + n > self.config.token_budget:
                break
            works.append(PrefillWork(r, n))
            used += n
        return tuple(works)


def make_scheduler(config: SchedulerConfig) -> ContinuousScheduler | StaticScheduler:
    return StaticScheduler(config) if config.kind == "static" else ContinuousScheduler(config)
