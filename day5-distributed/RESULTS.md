# Day 5 on two real GPUs: NCCL baseline and vLLM TP=1 vs TP=2

Measured on one RunPod pod with **2x NVIDIA A100-SXM4-80GB joined by NVLink** (`nvidia-smi topo -m`: GPU0-GPU1 = `NV12`),
vLLM 0.34.0, torch 2.14.0+cu132 (NCCL 2.30.7), `Qwen/Qwen2.5-1.5B-Instruct`, `--max-model-len 4096 --gpu-memory-utilization 0.85`.
Procedure: [RUNBOOK.md](RUNBOOK.md), driven by [run_tp_compare.sh](run_tp_compare.sh) (fresh server per run, 3 repeats, 0 failed requests in
every kept run). Evidence: [results/gpu_a100_nvlink/](results/gpu_a100_nvlink/) (`topo.txt`, `gpus.txt`, `nccl_all_reduce.txt`,
`kv_cache_lines.txt`, `summary.md`, NCCL demo logs). Raw per-run JSON and memory logs stay in `results/raw/` (git-ignored).
The Gloo-on-CPU results in `results/exp_*.json` are a different experiment (cost structure on a Mac); do not mix them with these.
The same experiment was repeated on a **PCIe-only pair (2x A40, no NVLink)**; it has its own section below and its own evidence folder
[results/gpu_a40_pcie/](results/gpu_a40_pcie/). The tables above that section are the NVLink A100 pod only.

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
| 1 | TP=2 does not double throughput for a 1.5B model | **Held** on both pods: best gain +37.8% (NVLink A100) and +56.0% (PCIe A40), neither near 2x. The "latency-bound" reason was not isolated (see the nsys paragraph in the PCIe section for what was measured). |
| 2 | TP=2 lowers TPOT/TTFT at low concurrency on NVLink, loses on PCIe | **NVLink side held** (TPOT -25.5%, TTFT -33% at concurrency 1). **PCIe side: partly refuted.** On the PCIe A40 pair TP=2 still lowers TPOT at low concurrency (-37.7% at 1, -25.2% at 8) and raises throughput (+56.0%, +27.3%), so it does not "lose" there. What PCIe changes: the TTFT gain disappears (43 vs 43 ms at concurrency 1, then +3% to +16% worse at 8-128) and TP=2 stops paying off between concurrency 32 (+5.9%) and 128 (**-11.2% throughput, +12.2% TPOT**). The two pods differ in GPU class as well as interconnect, so this is "TP=2 on PCIe A40" against "TP=2 on NVLink A100", not a clean interconnect swap. |
| 3 | Two TP=1 replicas beat one TP=2 server on throughput | **NVLink: mixed** (yes at 128, +19.5%; about even at 32, -2.7%). **PCIe: yes at both points tested** (+27.4% at 32, +95.9% at 128). Not tested at 1 or 8 on either pod. |
| 4 | Per-GPU weights halve, so more KV per GPU | **Held**: the KV pool doubles (2.04x group tokens). Memory in use per GPU does not change because the utilization setting fixes it. |
| 5 | Disabling custom all-reduce hurts TP=2 at small batch | **Held** on both pods. NVLink: at concurrency 8 TP=2 drops from 2,054 to 1,774 tok/s (-13.6%); its gain over TP=1 shrinks from +19.8% to +3.4% and TPOT returns to the TP=1 value (4.09 ms). PCIe: 1,165 to 1,101 tok/s (-5.5%), gain over TP=1 from +27.3% to +20.3%; smaller because the all-reduce is a smaller part of the step there (see nsys below). |

## The same experiment on a PCIe-only pair (2x A40)

Measured on a second pod: **2x NVIDIA A40 48 GB, no NVLink** (`nvidia-smi topo -m`: GPU0-GPU1 = `PIX`, one PCIe bridge; no `NV#` entries), same vLLM 0.34.0 /
torch 2.14.0+cu132 / NCCL 2.30.7, same model, flags, prompts, seeds, 3 repeats, fresh server per run, 0 failed requests. Evidence:
[results/gpu_a40_pcie/](results/gpu_a40_pcie/) (`summary.md`, `topo.txt`, `nccl_all_reduce.txt`, `kv_cache_lines.txt`, NCCL demo logs, `nsys_tp2/`).
Both Day 5 NCCL demos (`collectives.py`, `tp_demo.py`) pass on this pair too.

**Interconnect (nccl-tests `all_reduce_perf`, 2 ranks).** Two different A40 hosts were tried, and they differ:

| | small message (1 KB) | bus bandwidth at 256 MB | average over 1 KB-256 MB |
|---|---|---|---|
| NVLink, 2x A100 (earlier pod) | 12.8-13.8 us | 187 GB/s | 57 GB/s |
| PCIe, 2x A40, host used for everything below | 25-28 us | **7.6 GB/s** | 4.4 GB/s |
| PCIe, 2x A40, another host (topology `NODE`; this pod was abandoned, see limits) | 12-13 us | 13.6 GB/s | 7.3 GB/s |
| Exp C assumption for PCIe | 30 us | 12 GB/s | n/a |

So PCIe bus bandwidth was 4-7% of NVLink's, and two A40 hosts differed by 1.8x, so "PCIe" is not one number. The assumed latency (30 us) was close to this host's, and the assumed bandwidth (12 GB/s)
fell between the two hosts' measurements.

**TP=1 vs TP=2 (mean of 3 repeats; random 512 in / 128 out; change is TP=2 vs TP=1 at the same concurrency):**

| concurrency | TP=1 tok/s | TP=2 tok/s | TP=2 vs TP=1 | TPOT TP=1 / TP=2 ms | TTFT TP=1 / TP=2 ms |
|---|---|---|---|---|---|
| 1 | 136 | 213 | **+56.0%** | 7.05 / 4.40 (-37.7%) | 43 / 43 |
| 8 | 915 | 1,165 | **+27.3%** | 7.66 / 5.73 (-25.2%) | 145 / 150 |
| 32 | 2,318 | 2,454 | **+5.9%** | 10.92 / 10.00 (-8.4%) | 373 / 394 |
| 128 | 3,802 | 3,377 | **-11.2%** | 29.17 / 32.71 (+12.2%) | 581 / 675 |

Run-to-run CV is 0.1-1.8%, so every gap above is real (the smallest, +5.9%, has a 1.3% CV on the TP=2 side). The KV pool the server reports: 1,255,424 tokens at TP=1
vs 2,600,960 at TP=2 (**2.07x**; 2,588,672 with `--disable-custom-all-reduce`). Per-GPU memory in use is ~38.4-39.5 GB in every config (the 0.85 setting fixes it).
The TP=1 numbers on the abandoned first A40 host agree with these to within about 2% (136 / 912 / 2,333 / 3,860 tok/s), so single-GPU behaviour is the same across hosts.

NVLink A100 against PCIe A40, same experiment (throughput change of TP=2 over TP=1):

| concurrency | NVLink A100 | PCIe A40 |
|---|---|---|
| 1 | +34.8% | +56.0% |
| 8 | +19.8% | +27.3% |
| 32 | +29.1% | +5.9% |
| 128 | +37.8% | -11.2% |

At low concurrency the A40 gains *more*. A plausible reason (arithmetic, not isolated): the 3 GB of 1.5B weights take ~4.3 ms to read at the A40's 0.7 TB/s, so a TP=1
step is memory-bound there (7.05 ms TPOT) and halving the reads helps a lot, while on the A100 a larger share of the 4 ms step is launch overhead that TP does not touch.
At high concurrency the pattern reverses: the all-reduce message grows with the batch while the link is ~4% of NVLink's bandwidth, which is a likely reason for the crossover between
concurrency 32 and 128 (nsys was run at concurrency 8 only, so this is not measured).

**Two TP=1 replicas vs one TP=2 server (PCIe; same total concurrency, prompts split in half):**

| concurrency | 2 x TP=1 tok/s | TP=2 tok/s | 2 x TP=1 vs TP=2 | vs TP=1 | TPOT 2 x TP=1 / TP=2 ms |
|---|---|---|---|---|---|
| 32 | 3,126 | 2,454 | **+27.4%** | +34.8% | 8.49 / 10.00 |
| 128 | 6,614 | 3,377 | **+95.9%** | +74.0% | 15.30 / 32.71 |

As before, the replica throughput is the sum of two simultaneous benchmark runs, so it slightly overstates a single client's view. On PCIe the second GPU is clearly better
used as a replica than as a tensor-parallel partner for this model, at the concurrencies tested (32 and 128 only).

**All-reduce share of GPU time (Nsight Systems, TP=2 at concurrency 8, first measurement of RUNBOOK 3a).** Capture window of ~6 s of steady-state serving on both GPUs, with
`--cuda-graph-trace=node` (needed to see kernels inside the CUDA graphs; see below):

| arm | all-reduces per GPU per decode step | avg time per all-reduce | all-reduce share of busy GPU time | all-reduce time per step |
|---|---|---|---|---|
| vLLM custom kernel (`cross_device_reduce_1stage`) | 57.0 | 21.8 us | **18.9%** | 1.24 ms of a 6.56 ms busy step |
| NCCL (`--disable-custom-all-reduce`, `ncclDevKernel_AllReduce_Sum_bf16_RING_LL`) | 57.0 | 28.3 us | **23.1%** | 1.61 ms of a 6.96 ms busy step |

- The count check from the NVLink section holds: 56 (28 layers x 2) plus one for the vocab-parallel embedding = 57 per step.
- The custom kernel is ~23% faster per call than NCCL, which is consistent with its measured +5.8% throughput and -7.1% TPOT at this concurrency; the +16% seen on NVLink was larger, plausibly because
  the A100 step is shorter so the same per-step saving is a bigger fraction (not isolated).
- The GPUs are 94-95% busy in the window, so at concurrency 8 this is not an idle-GPU story.
- **Caveats.** Node-level graph tracing slows execution (6.9 ms per step in the window vs 5.73 ms TPOT unprofiled, about +20%), so use the shares, not the milliseconds. A collective
  kernel's duration includes waiting for the slower peer, so the share is an **upper bound on pure communication**. It is measured on PCIe A40 only; there is no NVLink capture,
  so it does not validate the 24% arithmetic floor in the NVLink section.
- **A first capture was invalid and discarded:** with nsys's default graph tracing the decode kernels are invisible (one entry per CUDA-graph launch), which gave a 34% share computed from only
  ~1 all-reduce per step. `analyze_nsys.py` now detects and flags that case and `run_nsys_tp2.sh` passes `--cuda-graph-trace=node`.

## What the numbers do and do not say

- At concurrency 1 the step is 4.06 ms at TP=1; weight reads alone would be ~1.5 ms (3 GB over ~2 TB/s), so a large part is not memory-bound
  (Day 6 found the same launch-bound behaviour on the A40). TP=2 therefore cannot give 2x. This is an interpretation from arithmetic plus Day 6, not a measurement here.
- Count check: 28 layers x 2 = 56 all-reduces per token. At the ~13 us small-message latency that is >= 0.7 ms, about 24% of the 3.02 ms TP=2 TPOT, as a floor
  (CUDA graphs and vLLM's custom kernel change the real cost). Not profiled.
- Why replicas win at 128 (a likely explanation, not isolated): each replica keeps the full model on its GPU and pays no per-token all-reduce, so two replicas reach
  82% of perfect scaling (14,274 vs 2 x 8,668 = 17,336 tok/s), while TP=2 reaches 69% (11,944) and pays communication every step. At 32 TP=2's lower step time
  roughly cancels that cost.

## Limits and what was not done

- **One node, one model (1.5B), one memory setting, two different GPU classes.** The NVLink pod is A100 and the PCIe pod is A40, so interconnect and GPU class are confounded;
  a same-GPU comparison (for example 2x A100 PCIe against the NVLink pod) was not run. Multi-node was not run.
- **Nsight Systems was not available on the NVLink pod**, so the NVLink all-reduce statements in the section above are arithmetic, not profile. The PCIe pod has a measurement (above), at concurrency 8 only.
- The first A40 pod was abandoned: every TP=2 server start failed there with `CUDA error: unspecified launch failure` and GPU 1 then became unusable (no driver reset in the container), while NCCL tests
  and the Day 5 NCCL demos had passed. The replacement pod ran TP=2 with the default custom all-reduce and with NCCL only without trouble, so this was most likely a bad host or GPU, not a property of
  PCIe; the cause was not established. Its nccl-tests result is kept above for comparison; its TP=1 runs and failure logs are in `results/raw/` (git-ignored).
- DP baseline at concurrency 32 and 128 only. Its first attempt was invalid (the second replica never started because RunPod's nginx owns port 8001 and answers
  `/health` with 200, so the script treated it as healthy) and was rerun on ports 18000/18001 with a health check that requires the model id from `/v1/models`;
  the invalid files were discarded. The fixed script is [run_tp_compare.sh](run_tp_compare.sh).
- The A100 pair is 80 GB, so a 1.5B model never feels KV pressure; the KV doubling is a capacity fact, not a latency effect here.
- Day 7's 8B follow-up (`--tensor-parallel-size 2`) is in the Day 7 [GPU_RESULTS.md](../day7-capstone/GPU_RESULTS.md).
