# Day 5 on two real GPUs: NCCL baseline and vLLM TP=1 vs TP=2

Measured on one RunPod pod with **2x NVIDIA A100-SXM4-80GB joined by NVLink** (`nvidia-smi topo -m`: GPU0-GPU1 = `NV12`),
vLLM 0.34.0, torch 2.14.0+cu132 (NCCL 2.30.7), `Qwen/Qwen2.5-1.5B-Instruct`, `--max-model-len 4096 --gpu-memory-utilization 0.85`.
Procedure: [RUNBOOK.md](RUNBOOK.md), driven by [run_tp_compare.sh](run_tp_compare.sh) (fresh server per run, 3 repeats, 0 failed requests in
every kept run). Evidence: [results/gpu_a100_nvlink/](results/gpu_a100_nvlink/) (`topo.txt`, `gpus.txt`, `nccl_all_reduce.txt`,
`kv_cache_lines.txt`, `summary.md`, NCCL demo logs). Raw per-run JSON and memory logs stay in `results/raw/` (git-ignored).
The Gloo-on-CPU results in `results/exp_*.json` are a different experiment (cost structure on a Mac); do not mix them with these.

## NCCL on this pair (nccl-tests `all_reduce_perf`, 2 ranks)

| | measured | Exp C assumption |
|---|---|---|
| small-message latency (1 KB) | 12.8-13.8 us | alpha = 15 us |
| bus bandwidth at 256 MB | 187 GB/s (still rising slowly) | 200 GB/s |
| average bus bandwidth over 1 KB-256 MB | 57 GB/s | n/a |

The assumed numbers were close. Note: nccl-tests was built against the container's system NCCL 2.25.1, not the NCCL 2.30.7 that torch and vLLM
use, so treat it as the interconnect's ceiling, not as vLLM's exact collective path.
Day 5's own NCCL path ran for the first time too: `collectives.py` (all-reduce, broadcast, all-gather, barrier) and `tp_demo.py` (column-parallel,
row-parallel and the MLP match the unsharded reference to ~1e-5 relative error ~1e-7) both pass under `torchrun` with the NCCL backend.
`run_experiments.py` is CPU/Gloo only and was not run on the GPUs.

## TP=1 vs TP=2 (mean of 3 repeats; random 512 in / 128 out; change is TP=2 vs TP=1 at the same concurrency)

| concurrency | TP=1 tok/s | TP=2 tok/s | TP=2 vs TP=1 | TPOT TP=1 / TP=2 ms | TTFT TP=1 / TP=2 ms |
|---|---|---|---|---|---|
| 1 | 235 | 316 | **+34.8%** | 4.06 / 3.02 (-25.5%) | 30 / 20 |
| 8 | 1,715 | 2,054 | **+19.8%** | 4.09 / 3.48 (-14.9%) | 77 / 56 |
| 32 | 4,790 | 6,184 | **+29.1%** | 5.19 / 4.11 (-20.8%) | 192 / 136 |
| 128 | 8,668 | 11,944 | **+37.8%** | 12.20 / 8.67 (-28.9%) | 327 / 258 |

Run-to-run CV is 0.1-1.3%, so every gap above is real. Per-GPU memory used is ~69 GB for every config (the 0.85 setting fixes it); what changes is the
KV pool the server reports: **2,363,392 tokens at TP=1 vs 4,816,896 at TP=2 (2.04x)**.

## Two TP=1 replicas vs one TP=2 server (same total concurrency, prompts split in half)

| concurrency | 2 x TP=1 tok/s | TP=2 tok/s | 2 x TP=1 vs TP=2 |
|---|---|---|---|
| 32 | 6,015 | 6,184 | -2.7% (about even; TP=2 CV was 1.3%) |
| 128 | 14,274 | 11,944 | **+19.5%** |

TPOT: 4.38 vs 4.11 ms at 32, 6.85 vs 8.67 ms at 128 (replicas faster at 128). The replica throughput is the sum of two simultaneous benchmark
runs, so it slightly overstates a single client's view. Only concurrency 32 and 128 were run.

## Hypotheses (RUNBOOK section 5)

| # | Hypothesis | Verdict |
|---|---|---|
| 1 | TP=2 does not double throughput for a 1.5B model | **Held**: best gain +37.8%. The claim that this is because the all-reduces are latency-bound was not measured (see limits). |
| 2 | TP=2 lowers TPOT/TTFT at low concurrency on NVLink, loses on PCIe | **Half tested**: NVLink side held (TPOT -25.5%, TTFT -33% at concurrency 1). PCIe was not run. |
| 3 | Two TP=1 replicas beat one TP=2 server on throughput | **Mixed**: clearly yes at concurrency 128 (+19.5%), about even at 32 (-2.7%). Not tested at 1 or 8. |
| 4 | Per-GPU weights halve, so more KV per GPU | **Held**: the KV pool doubles (2.04x group tokens). Memory in use per GPU does not change because the utilization setting fixes it. |
| 5 | Disabling custom all-reduce hurts TP=2 at small batch | **Held**: at concurrency 8 TP=2 drops from 2,054 to 1,774 tok/s (-13.6%); its gain over TP=1 shrinks from +19.8% to +3.4% and TPOT returns to the TP=1 value (4.09 ms). |

## What the numbers do and do not say

- At concurrency 1 the step is 4.06 ms at TP=1; weight reads alone would be ~1.5 ms (3 GB over ~2 TB/s), so a large part is not memory-bound
  (Day 6 found the same launch-bound behaviour on the A40). TP=2 therefore cannot give 2x. This is an interpretation from arithmetic plus Day 6, not a measurement here.
- Count check: 28 layers x 2 = 56 all-reduces per token. At the ~13 us small-message latency that is >= 0.7 ms, about 24% of the 3.02 ms TP=2 TPOT, as a floor
  (CUDA graphs and vLLM's custom kernel change the real cost). Not profiled.
- Why replicas win at 128 (a likely explanation, not isolated): each replica keeps the full model on its GPU and pays no per-token all-reduce, so two replicas reach
  82% of perfect scaling (14,274 vs 2 x 8,668 = 17,336 tok/s), while TP=2 reaches 69% (11,944) and pays communication every step. At 32 TP=2's lower step time
  roughly cancels that cost.

## Limits and what was not done

- **NVLink only, one node, one model (1.5B), one memory setting.** PCIe (hypothesis 2's other half) and any multi-node case were not run.
- **No Nsight Systems on the pod**, so the share of GPU time in all-reduce (RUNBOOK 3a) is not measured; the all-reduce statements above are arithmetic, not profile.
- DP baseline at concurrency 32 and 128 only. Its first attempt was invalid (the second replica never started because RunPod's nginx owns port 8001 and answers
  `/health` with 200, so the script treated it as healthy) and was rerun on ports 18000/18001 with a health check that requires the model id from `/v1/models`;
  the invalid files were discarded. The fixed script is [run_tp_compare.sh](run_tp_compare.sh).
- The A100 pair is 80 GB, so a 1.5B model never feels KV pressure; the KV doubling is a capacity fact, not a latency effect here.
- Day 7's 8B follow-up (`--tensor-parallel-size 2`) is in the Day 7 [GPU_RESULTS.md](../day7-capstone/GPU_RESULTS.md).
