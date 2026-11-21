"""A single inference request and its lifecycle state.

`Request` is the one deliberately *mutable* class in the simulator: the engine
advances it iteration by iteration (progress counters, timestamps, status).
Everything else (configs, plans, records, results) is frozen.

Token accounting (matches how a real engine behaves):

* Prefill processes `prefill_goal` tokens (initially the prompt). When the last
  prefill chunk finishes, the model emits the first output token
  (`generated_tokens` += 1). That instant is the time-to-first-token (TTFT).
* Each decode iteration feeds the previously generated token, adds one KV entry
  (`kv_tokens` += 1) and emits one more token.
* The simulator has no EOS: a request always generates exactly
  `max_output_tokens`, so the KV footprint peaks at
  `prompt_tokens + max_output_tokens - 1` (the last token's KV is never stored).
* Preemption by recompute frees the KV cache. The tokens already streamed are
  kept, and the next prefill must rebuild KV for prompt + generated tokens, so
  `prefill_goal` grows to `prompt_tokens + generated_tokens`.
"""
from __future__ import annotations

WAITING = "WAITING"
RUNNING = "RUNNING"
FINISHED = "FINISHED"


class Request:
    def __init__(
        self,
        request_id: int,
        arrival_time: float,
        prompt_tokens: int,
        max_output_tokens: int,
        priority: int = 0,
    ) -> None:
        if prompt_tokens < 1 or max_output_tokens < 1:
            raise ValueError("prompt_tokens and max_output_tokens must be >= 1")
        if arrival_time < 0:
            raise ValueError("arrival_time must be >= 0")
        self.request_id = request_id
        self.arrival_time = arrival_time  # ms
        self.prompt_tokens = prompt_tokens
        self.max_output_tokens = max_output_tokens
        self.priority = priority  # lower value = more important
        self.generated_tokens = 0
        self.status = WAITING

        # progress
        self.kv_tokens = 0  # tokens whose KV entries currently exist
        self.prefill_goal = prompt_tokens

        # timestamps (ms) and accounting
        self.waiting_since: float = arrival_time
        self.waiting_time = 0.0  # total time spent WAITING (including after preemption)
        self.first_scheduled_time: float | None = None
        self.first_token_time: float | None = None
        self.finish_time: float | None = None
        self.last_token_time: float | None = None
        self.token_gaps: list[float] = []  # every gap between consecutive output tokens (ms)
        self.preemptions = 0
        self.recomputed_tokens = 0  # KV tokens thrown away by preemption

    # ---- derived state -------------------------------------------------
    @property
    def is_prefilling(self) -> bool:
        return self.kv_tokens < self.prefill_goal

    @property
    def is_done(self) -> bool:
        return self.generated_tokens >= self.max_output_tokens

    @property
    def peak_kv_tokens(self) -> int:
        return self.prompt_tokens + self.max_output_tokens - 1

    @property
    def max_itl(self) -> float:
        """Worst gap between two consecutive output tokens of this request."""
        return max(self.token_gaps, default=0.0)

    @property
    def latency(self) -> float | None:
        return None if self.finish_time is None else self.finish_time - self.arrival_time

    @property
    def ttft(self) -> float | None:
        return None if self.first_token_time is None else self.first_token_time - self.arrival_time

    # ---- transitions ---------------------------------------------------
    def reset_for_recompute(self) -> int:
        """Drop the KV cache; return how many KV tokens were discarded."""
        wasted = self.kv_tokens
        self.recomputed_tokens += wasted
        self.kv_tokens = 0
        self.prefill_goal = self.prompt_tokens + self.generated_tokens
        self.status = WAITING
        self.preemptions += 1
        return wasted

    def __repr__(self) -> str:
        return (
            f"Request(id={self.request_id}, t={self.arrival_time:.1f}, P={self.prompt_tokens}, "
            f"O={self.max_output_tokens}, prio={self.priority}, {self.status}, gen={self.generated_tokens})"
        )
