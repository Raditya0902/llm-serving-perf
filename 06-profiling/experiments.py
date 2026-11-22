"""The `06-profiling` experiment registry: variable, baseline, measurement and the hypothesis.

Hypotheses are written here *before* any measurement is taken, and are never edited
to match results. If a result disagrees, the report says so.

`runnable_on`:
  local - measured on the dev machine with the `01-inference-loop` engine (this folder)
  gpu   - vLLM scheduler/memory knobs; needs an NVIDIA host (see RUNBOOK.md). On the
          Mac only a `03-batching-simulator` *prediction* exists, labelled SIMULATED.
"""

from __future__ import annotations

from dataclasses import dataclass

RUNNABLE_ON = ("local", "gpu")


@dataclass(frozen=True)
class Experiment:
    exp_id: str
    name: str
    variable: str
    values: tuple[int | str, ...]
    baseline: int | str
    measurement: str
    hypothesis: str
    runnable_on: str

    def __post_init__(self) -> None:
        if not self.hypothesis.strip():
            raise ValueError(f"{self.exp_id}: hypothesis must be written before measuring")
        if self.baseline not in self.values:
            raise ValueError(f"{self.exp_id}: baseline {self.baseline!r} not in values {self.values}")
        if self.runnable_on not in RUNNABLE_ON:
            raise ValueError(f"{self.exp_id}: runnable_on must be one of {RUNNABLE_ON}")


EXPERIMENTS: tuple[Experiment, ...] = (
    Experiment(
        "E1", "Concurrent requests (static batch)", "batch_size", (1, 2, 4, 8), 1,
        "output tokens/s, per-step latency (TPOT), peak memory",
        "Batch-1 decode is bound by weight reads and per-op overhead, not arithmetic, so adding "
        "requests is nearly free per step: step time at B=8 is at most 1.5x the B=1 step time, "
        "so throughput at B=8 is at least 5x. Per-request TPOT rises modestly (<= 1.5x).",
        "local",
    ),
    Experiment(
        "E2", "Prompt length -> TTFT", "prompt_len", (128, 512, 1024, 2048), 512,
        "TTFT (prefill latency), prefill tokens/s",
        "Prefill is compute-bound at these lengths, so TTFT grows about linearly with prompt "
        "length: TTFT(2048) / TTFT(512) lies in [3.8, 5.0], and prefill tokens/s is roughly flat "
        "(`01-inference-loop`: ~3.3k tok/s at 512).",
        "local",
    ),
    Experiment(
        "E3", "Output length -> TPOT and total latency", "new_tokens", (32, 128, 256), 128,
        "TPOT, total latency",
        "TPOT is nearly independent of output length (<= 5% change between 32 and 256 tokens, "
        "KV reads are small next to the weights), so total latency is linear in output length.",
        "local",
    ),
    Experiment(
        "E4", "Maximum batched tokens", "max_num_batched_tokens", (512, 2048, 8192, 16384), 8192,
        "output tokens/s, P95 TTFT, P95 inter-token latency",
        "A small budget chunks long prefills: decoders keep stepping (lower P95 inter-token "
        "latency) but long prompts wait longer (higher P95 TTFT). Throughput plateaus once the "
        "budget exceeds what one iteration can use (~2048+).",
        "gpu",
    ),
    Experiment(
        "E5", "Maximum active sequences", "max_num_seqs", (16, 64, 256), 256,
        "queueing delay (TTFT), GPU occupancy, output tokens/s",
        "At a saturating request rate, raising max_num_seqs raises throughput and lowers TTFT "
        "(less queueing) until compute saturates or KV capacity binds; past that it is flat or "
        "slightly worse for per-token latency.",
        "gpu",
    ),
    Experiment(
        "E6", "KV-cache capacity", "gpu_memory_utilization", (0.30, 0.50, 0.70, 0.85), 0.85,
        "preemptions, output tokens/s, P95 TTFT, peak GPU memory",
        "Below the workload's peak KV demand, preemptions appear and throughput falls while P95 "
        "TTFT rises; above it, extra capacity changes nothing measurable.",
        "gpu",
    ),
    Experiment(
        "O1", "Optimization: defer host syncs", "decode_loop", ("day1", "deferred_sync"), "day1",
        "per-step latency, tokens/s (outputs must be identical)",
        "`01-inference-loop` syncs twice per step. If the CPU enqueue time is >= ~80% of the synced step time "
        "(CPU/launch-bound), removing syncs cannot help: improvement < 5%. A bigger gain would "
        "mean the step was waiting on sync latency, not on work.",
        "local",
    ),
    Experiment(
        "O2", "Optimization: attention implementation", "attn_impl", ("sdpa", "eager"), "sdpa",
        "TTFT, TPOT (outputs must match)",
        "At decode (query length 1) the attention core is a small share of the step, so sdpa vs "
        "eager differ by < 5% in TPOT; any difference shows up in prefill (TTFT) at 512+ tokens.",
        "local",
    ),
)


def get_experiment(exp_id: str) -> Experiment:
    for exp in EXPERIMENTS:
        if exp.exp_id == exp_id:
            return exp
    raise KeyError(exp_id)
