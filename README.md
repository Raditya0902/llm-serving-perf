# llm-serving-perf

Measuring and explaining LLM serving performance, from KV-cache bytes down to attention kernels, on real NVIDIA GPUs with vLLM.

Every experiment follows the same loop: write the prediction down first, run it on hardware with a measured noise floor, then explain the
result down to the kernel or the scheduler decision behind it. Predictions that failed are reported as failed.

**Hardware used:** NVIDIA A40, RTX A6000, A100 SXM4 80 GB (NVLink pair), A100 40 GB PCIe, 8x A100 SXM4 40 GB, H100 PCIe, H100 SXM5,
plus an Apple M4 for the from-scratch work. **Stack:** vLLM 0.34.0, PyTorch, CUDA, NCCL, Nsight Systems / Nsight Compute, torch profiler.
**Models:** Qwen2.5 0.5B-3B, Llama-3.1-8B.

## Headline results (all measured on GPUs)

### 1. A hidden attention penalty in mixed prefill/decode batches: +60% throughput from a one-flag change

On an A40, raising `max_num_seqs` from 256 to 1024 made vLLM **11% slower**. Tracing it down:

- Kernel attribution of profiler traces: in steps that mix prefill and decode, **attention costs 6.8x** what the same 256 sequences cost in a
  pure-decode step.
- A microbenchmark of vLLM's own `flash_attn_varlen_func` reproduces it. The penalty switches on as soon as **any sequence in the batch has
  2 query tokens**, and it scales with the GQA group size (group 1 / 4 / 7 gives 1.0x / 3.65x / 6.6x).
- **Nsight Compute DRAM counters confirm the mechanism:** in mixed batches FlashAttention 2 drops its grouped-query path and reads the KV cache
  once per query head (7.04x the bytes at GQA 7, 4.11x at GQA 4).
- **Fix:** `--attention-backend TRITON_ATTN` gives **+60% tok/s** at 3584-in / 128-out and +14% at 512 / 128. The 1024-vs-256 loss goes
  from -9.4% to +0.4%. Output is equivalent to FA2 at 1.5B, 3B and 8B. FlashAttention 3 on Hopper does not have the penalty.

Details: [06-profiling/performance-report.md §9.9-9.14](06-profiling/performance-report.md).

### 2. Tensor parallelism depends on the link, not the model

Llama-3.1-8B on two A100s, TP=2 vs TP=1:

| Interconnect | 256 MiB all-reduce | TPOT speedup (c16) | Verdict |
|---|---|---|---|
| NVLink (SXM4) | 158 GB/s | **1.65x** | TP=2 helps |
| PCIe host | 3.1 GB/s | **0.70x** | TP=2 is slower than one GPU |

Two independent TP=1 replicas beat TP=2 by 31% at concurrency 64 on a PCIe A40 pair, but not on NVLink (-4.1%). TP=2 also grows the KV pool 2-2.9x
and removes preemptions (62 to 0). Rule that came out of it: measure a 256 MiB all-reduce before choosing TP.
Details: [05-tensor-parallel/RESULTS.md](05-tensor-parallel/RESULTS.md), [07-capacity-study/GPU_RESULTS.md](07-capacity-study/GPU_RESULTS.md).

### 3. A capacity study with pre-registered predictions

Llama-3.1-8B, 49 fresh-server runs, 0 failed requests, A/A noise floor of 0.8% on throughput. Six predictions were made from a simulator
before the run: four held, one was borderline, one was half refuted. The "tuned" config cut TTFT 43% but **lost 4.8% throughput**.
Single-knob follow-up arms showed why: the larger KV pool alone cuts TTFT 48% (+2.4% tok/s), and the smaller token budget alone costs 6.2% tok/s
(the simulator had this sign wrong). Decision: keep the baseline config.
Details: [07-capacity-study/GPU_RESULTS.md](07-capacity-study/GPU_RESULTS.md).

### 4. vLLM scheduler knobs, swept one at a time

168 runs on an A40 (fresh server per run, 3 repeats). Example: `max_num_batched_tokens` 512 vs 8192 gives **4.6x lower p99 inter-token latency
for 46% less throughput**. Each knob's expected direction was derived from the vLLM source before running.
Details: [04-vllm-internals/RESULTS.md](04-vllm-internals/RESULTS.md).

## Repository layout

| Folder | What it covers | Evidence |
|---|---|---|
| [01-inference-loop](01-inference-loop/) | Hand-written greedy decoding with and without a KV cache; prefill vs decode cost | Apple M4 (8.5-16.5x KV-cache speedup) |
| [02-kv-cache-memory](02-kv-cache-memory/) | Memory calculator, paged KV allocator vs contiguous reservation, GQA savings | Arithmetic + simulation |
| [03-batching-simulator](03-batching-simulator/) | Discrete-event continuous-batching scheduler: chunked prefill, preemption, priorities | Simulated (95 tests, 99% coverage) |
| [04-vllm-internals](04-vllm-internals/) | Annotated vLLM request path (151 machine-checked `file:line` symbols) and the knob sweep | A40 |
| [05-tensor-parallel](05-tensor-parallel/) | Collectives, Megatron-style column/row-parallel layers, TP=1 vs TP=2 vs replicas, NCCL | A100 NVLink, A40 PCIe |
| [06-profiling](06-profiling/) | Step- and kernel-level profiling, attention microbenchmarks, Nsight counters, backend A/B | A40, A6000, H100 |
| [07-capacity-study](07-capacity-study/) | 8B serving matrix, simulator-vs-GPU scoring, KV sizing, multi-GPU model | A40, A100, H100 |

Each folder has its own README; GPU folders also have a `RUNBOOK.md` (exact commands) and pre-registration files where predictions were made.

## Measured vs simulated

Folders 01-03 were built on a Mac without an NVIDIA GPU. Their numbers come from Apple MPS or from simulation, and the reports tag each claim
(`[measured-Mac]`, `[SIM]`, `[source]`, `[assumed]`). Every number in "Headline results" above was measured on the NVIDIA GPU named next to it.
Raw traces and per-run JSON from GPU hosts stay off git (`results/raw/` is ignored); summaries, tables and environment captures are committed.

## Reproduce

Each folder is self-contained, with its own `requirements.txt` and test suite (1,196 tests in total):

```bash
cd 03-batching-simulator
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -m pytest tests -q
```

GPU experiments need a Linux host with vLLM 0.34.0. Run `04-vllm-internals/setup_gpu_host.sh`, then follow the `RUNBOOK.md` in the folder
you want to reproduce.
