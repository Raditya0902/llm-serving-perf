"""Per-iteration compute-cost model (roofline style).

One engine iteration processes some prefill tokens and some decode tokens
(one per decoding sequence) in a single forward pass. Its simulated duration is

    iter_ms = overhead_ms + max(compute_ms, memory_ms)

    compute_ms = compute_ms_per_token * (prefill_tokens + decode_tokens)
    memory_ms  = weight_read_ms + kv_read_ms_per_token * decode_ctx_tokens

* `compute_ms` is tensor-core work, linear in tokens processed.
* `memory_ms` is HBM traffic: all weights are streamed once per iteration no
  matter how many tokens ride along, plus every decoding sequence's KV cache.
* `overhead_ms` is a fixed kernel-launch / framework cost per iteration.

Consequences (the whole point of using this instead of a constant):

1. A small decode batch is memory-bound, so adding sequences is nearly free.
   That is why batching raises throughput.
2. A large prefill is compute-bound (2048 tokens ~ 82 ms vs ~7.5 ms for a decode
   step), so an unchunked long prompt stalls every decoder in the batch.
3. A small prefill chunk hides under the weight-read time for free, which is
   why chunked prefill can piggy-back on decode iterations.

Defaults are illustrative numbers in the ballpark of an 8B fp16 model on an
A100-class GPU (~25k prefill tok/s, ~7 ms memory floor per step); they are not
measurements. Attention's quadratic prefill cost is ignored (simplification).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CostModel:
    overhead_ms: float = 0.3
    compute_ms_per_token: float = 0.04
    weight_read_ms: float = 7.0
    kv_read_ms_per_token: float = 0.0001

    def iteration_time(self, prefill_tokens: int, decode_tokens: int, decode_ctx_tokens: int) -> float:
        """Simulated duration (ms) of one iteration. Zero if there is no work."""
        if min(prefill_tokens, decode_tokens, decode_ctx_tokens) < 0:
            raise ValueError("token counts must be non-negative")
        if prefill_tokens + decode_tokens == 0:
            return 0.0
        compute = self.compute_ms_per_token * (prefill_tokens + decode_tokens)
        memory = self.weight_read_ms + self.kv_read_ms_per_token * decode_ctx_tokens
        return self.overhead_ms + max(compute, memory)
