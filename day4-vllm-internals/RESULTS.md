# Session 3 results (measured on an NVIDIA A40)

First real GPU run of the Day 4 sweep tooling. Generated tables, plots and per-run environment:
[brief load](results/qwen0.5b-brief/RESULTS.md) · [saturated, 1000 prompts](results/qwen0.5b-saturated-1000/RESULTS.md) ·
[saturated, 100 prompts (first pass)](results/qwen0.5b-saturated/RESULTS.md) · [environment](results/env.txt).
Raw per-run JSON, `/metrics` scrapes and server logs stay on the GPU host (`results/raw/` is git-ignored).

## Setup

| | |
|---|---|
| Hardware | 1x NVIDIA A40 (46 GiB usable), RunPod container, driver 595.112.07 |
| Software | vLLM 0.34.0, torch 2.14.0+cu132, Python 3.12.3, chunked prefill on, CUDA graphs on (vLLM defaults) |
| Model | `Qwen/Qwen2.5-0.5B-Instruct`, `--max-model-len 4096` |
| Workload | `vllm bench serve`, random prompts, 512 in / 128 out (`--ignore-eos`), seed 0 |
| Design | one knob changed per run, baseline `max_num_seqs=256`, `max_num_batched_tokens=8192`, `gpu_memory_utilization=0.85`; **fresh server per run**; 3 repeats |
| Runs | brief (2 req/s unless the knob is the rate, 100 prompts): 66 runs · **saturated, 1000 prompts (`--request-rate inf`): 17 configs, 51 runs (the primary result)** · saturated, 100 prompts: 51 runs (first pass, kept for comparison) |
| Health | **0 failed requests, 0 preemptions in all 168 runs** |

Repeat-to-repeat noise on the 1000-prompt run (n=3): output throughput varies by about 1% typically; worst case `input_len=128` at 4.9%
(std 803 on 16,343 tok/s). **Differences of about 2% in throughput are noise.** p99 over 1000 requests is the 10th-slowest request, so it is a real
percentile, unlike the 100-prompt run where it was just the slowest one.

## Why there are two saturated runs

The first saturated pass sent 100 prompts. That capped concurrency at 100, so `max_num_seqs` of 256 and 1024 were identical by construction, and each
burst lasted only 1.6 s, so ramp-up and drain dominated the throughput. The rerun sends 1000 prompts (burst lasts 13.6 s at baseline), which lets
`max_num_seqs=1024` actually differ from 256. It changed the numbers, and one conclusion (the `max_num_seqs` "plateau") reversed.

## Findings (1000-prompt saturated run)

Hypotheses are the "expected direction" column of [CHECKPOINT.md](CHECKPOINT.md) Q6, written from the source before any run.

| Knob | Hypothesis | Result | Verdict |
|---|---|---|---|
| `max_num_seqs` | Larger = more throughput until the KV pool or token budget binds | Throughput 1,124 (4) -> 3,513 (16) -> 6,926 (64) -> **9,445 (256)** -> **8,399 (1024)** tok/s. At 1024: p99 ITL 442 ms vs 109 ms, p50 ITL 39 vs 11 ms, but p99 TTFT 11.3 s vs 12.2 s. Brief (2 req/s): no visible latency effect (throughput there is set by the arrival schedule) | **Held up to 256**, **not held beyond**: 1024 is 11% *slower* than 256 and has 4x worse tail ITL. The 100-prompt "plateau" was the workload size, not a limit. Cause **partly explained**: a CUDA-graph capture-range effect was tested and **refuted** (capture size raised to 1024: no change, Day 6 report 9.3), and profiler traces show steps carrying prefill chunks cost 42-135 ms vs 7 ms for pure decode, which explains the ITL tail but **not** the 9% throughput loss |
| `max_num_batched_tokens` | Smaller = smoother ITL at some cost to TTFT and throughput | p99 ITL 23.9 / 45.2 / 60.1 / 77.4 / 109 / 179 ms for 512 / 1024 / 2048 / 4096 / 8192 / 16384; throughput 5,063 / 5,330 / 6,878 / 8,501 / 9,445 / 9,578 tok/s; p99 TTFT 24.3 -> 11.9 s | **Held, and large.** 512 vs 8192: 4.6x lower p99 ITL for 46% less throughput and 2.0x higher p99 TTFT. 16384 vs 8192: +1.4% throughput (noise) for 64% worse p99 ITL. Mechanism (prefill chunks stalling decode steps) fits the numbers but was **not isolated** (no trace). **The p50 ITL at 1024 (36.7 ms) vs 512 (19.3) and 2048 (19.2) is explained** by the share of tokens emitted in prefill-carrying steps (Day 6 report 9.3). |
| `gpu_memory_utilization` | Sets KV pool size; no speed effect until the pool binds | KV pool from server logs: 1.75M / 2.53M / 3.11M / 3.50M tokens (0.5 / 0.7 / 0.85 / 0.95). Throughput 9,508 / 9,492 / 9,445 / 9,580 tok/s, p99 TTFT within 2% | **Held** for both halves. Peak KV demand never approached the pool (see below) |
| Prompt length | Longer = higher TTFT, more KV per request, more preemption risk | Throughput 16,343 / 9,445 / 5,464 / 2,581 / 1,231 tok/s and p99 TTFT 6.7 / 12.2 / 21.7 / 47.2 / 100.6 s for 128 / 512 / 1024 / 2048 / 3584 tokens. p50 ITL at 3584 is 234 ms (vs 11 ms at 512) | TTFT and throughput parts **held**. Preemption-risk part **untested** (no preemption). The 234 ms ITL at 3584 is not explained: it fits 256 concurrent long prompts' prefill chunks sharing every step, but that was not checked |

Brief-load findings (2 req/s, 100 prompts) are in [the brief report](results/qwen0.5b-brief/RESULTS.md): throughput tracks the arrival rate
(127 / 254 / 503 / 987 / 1,904 tok/s at 1 / 2 / 4 / 8 / 16 req/s) because the load is closed by the schedule, and unthrottled it reaches about 7,850 tok/s
with p99 TTFT jumping from 68 ms to 784 ms. The saturation point lies somewhere between 16 req/s and unthrottled and this sweep does not locate it.

## What changed from the 100-prompt first pass

| Quantity | 100 prompts | 1000 prompts |
|---|---|---|
| Baseline throughput | 7,806 tok/s | 9,445 tok/s (+21%: less ramp/drain in the average) |
| `max_num_seqs` 256 -> 1024 | 7,806 -> 7,900 ("plateau") | 9,445 -> 8,399 (**-11%**) |
| `max_num_batched_tokens` 512 vs 8192, p99 ITL | 3.7x lower, 29% less throughput | 4.6x lower, 46% less throughput |
| `max_num_seqs=4` throughput | 1,120 | 1,124 (same: it is bound by the cap, not the burst) |

The direction of every other conclusion survived; the sizes did not, so quote the 1000-prompt numbers.

## Preemption was never triggered

`vllm:num_preemptions_total` was 0 after every one of the 168 runs (a cumulative counter, so this is reliable even though it was scraped at the end).
Peak KV demand in the table is **computed** (the original `kv_cache_usage_perc` scrape came after the run, when it reads 0). A later rerun with a 5 Hz live sampler (Day 6) **measured** 5.3% peak KV at 256 seqs and 21.5% at 1024 seqs, matching the computed 5.3% and 21.8%:

| Case | Upper-bound KV demand | Pool | Headroom |
|---|---|---|---|
| baseline (256 seqs x 640 tokens) | 164k tokens | 3.11M | 19x |
| `max_num_seqs=1024` (1000 x 640) | 640k | 2.93M | 4.6x |
| `gpu_memory_utilization=0.5`, baseline load | 164k | 1.75M | 10.7x |
| `input_len=3584` (256 x 3,712) | 950k | 3.11M | 3.3x |

So CHECKPOINT Q4 (behaviour under real KV exhaustion) still has no measurement. Day 7's matrix with `--cap-kv-pool`, or a sweep with a much smaller
`--kv-cache-memory-bytes` here, is the experiment that can trigger it.

## Observation worth following up

At 4 req/s and above in the brief profile, p99 ITL rises from about 5 ms to 14-15 ms with throughput unchanged. That fits long prefills occasionally
delaying decode steps, but it is a hypothesis; a profile of a steady-state window (Day 6) would show it directly.

## Limits of this run

- **One GPU, one small model.** A 0.5B model decodes in about 3 ms per token and is far from compute-bound, so absolute numbers and the size of the
  effects will not transfer to 7B+ models. The 1.5B repeat in [RUNBOOK.md](RUNBOOK.md) section 4 was **not run**.
- **n=3 repeats.** Throughput gaps under about 3% and tail-latency gaps under about 25% are not resolvable.
- **"Saturated" means one burst of 1000 requests, not sustained load.** The run ends when the burst drains (13.6 s at baseline, 114 s for
  `max_num_seqs=4`), so throughput still includes a ramp-up and a shrinking batch at the end. Comparisons between knobs are valid; absolute tok/s
  are not sustained serving rates. TTFT here is mostly queueing time behind the burst (baseline p50 TTFT is 6 s).
- **Brief-profile throughput is set by the schedule** (every run lasts 50.5 s), so only its latency columns can show an effect.
- **Shared-cloud host.** RunPod container, GPU not otherwise loaded (checked idle between runs); host CPU contention was not measured.
- **No server-side tracing.** Explanations marked "consistent with" or "not isolated" are inferences from the numbers, not observations of the scheduler.
- The Day 3 simulator was **not** compared against these numbers here; [COMPARISON.md](COMPARISON.md) still describes the source-level comparison only.
