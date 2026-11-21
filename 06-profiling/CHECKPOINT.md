# Day 6 knowledge checkpoint

Evidence tags: **[measured]** = from the Apple M4 / MPS runs in [performance-report.md](performance-report.md);
**[measured A40]** = from the NVIDIA A40 runs (report section 9); **[mechanism]** = standard reasoning I did not test here;
**[GPU pending]** = still not run (Nsight Compute kernel counters were blocked in a RunPod container, `ERR_NVGPUCTRPERM`, and were read in round 5 on Lambda VMs, report 9.14; the 8B at 1,024 sequences and models above 8B remain open, 9.13). nsys timelines, models up to 7B, the kernel-level cause of the mixed-step decode cost (FlashAttention 2 loses its GQA decode path, 9.9), the `max_num_seqs=1024` isolation (9.10) and 8B vLLM step costs (9.11) have since been run. Round 4 (9.12) added: TRITON_ATTN is output-equivalent to FA2 on the 0.5B, the Triton gain holds at 1.5B / 3B / 8B, FlashAttention 3 on an H100 has no mixed-step penalty, the ratio of 18 is a block-launch cost hidden at long context, E1/E3 hold and E2 is refuted under vLLM at every size.

> **A40 update that changes the answers below.** The Day 1 loop on CUDA is **launch-bound, not memory-bound**: enqueue is 0.97-0.98 of the step (unprofiled), a loop with no syncs takes exactly its enqueue time, GPU device work is at most about 27% of the step, step time is 16.3 ms at every batch size 1-8, and TTFT is flat to 1024 tokens. So Q1's "decode is bandwidth-bound" holds for the roofline *argument* but not for this loop on this GPU (about 1.4 ms of weight reads at 696 GB/s inside a 16 ms step), and Q3's "ask whether the CPU can feed the device" is answered **yes, it cannot** [measured A40]. Q2's utilization point is also confirmed: "GPU busy" was true but the device did at most about 4 ms of work in a 15 ms step. The Mac text below stays as the MPS result.

## 1. How do you determine whether inference is compute-bound or memory-bound?

Compare achieved rates with the hardware's two ceilings (a roofline), then confirm by changing the work.
- **Arithmetic intensity.** Decode at batch 1 does about 2 FLOPs per weight byte read at fp16 (1 FLOP/byte), far below any GPU's FLOPs-per-byte ridge, so it should be bandwidth-bound; prefill reuses each weight across hundreds of tokens, so it should be compute-bound. [mechanism]
- **Achieved bandwidth.** Replaying the real matmuls gave 51-93 GB/s effective against Day 1's assumed 120 GB/s, at 0.99 GB of weights per step. [measured] Decode time is set by bytes moved.
- **Scale the work and watch the time.** Eight times the decode work (batch 8) cost only 1.31-1.44x the step time in four independent runs: the weight read is paid once, so the step was memory-bound. [measured] Prefill instead scaled with tokens (TTFT 50 / 162 / 320 / 708 ms for 128 / 512 / 1024 / 2048 tokens, about 3.2k tok/s from 512 up): compute-bound. [measured]
- **Caveat.** "Memory-bound" at this model size also includes per-op overhead; on MPS I could not read hardware counters (SM/occupancy, DRAM throughput). On NVIDIA, Nsight Compute provides them (tried in round 3 and refused by the container with `ERR_NVGPUCTRPERM`, so the kernel-level mechanism of 9.9 rests on timings, not counters; nsys busy fraction and kernel counts were measured, report 9.5). [GPU pending for counters]

## 2. Why can GPU utilization be misleading?

It reports the fraction of time *any* kernel was resident, not how much of the GPU's compute or bandwidth was used. A single small kernel occupying one SM reads "100% utilized". [mechanism]
Here, Day 1's decode loop is "busy" the whole time yet its matmuls reach only 51-93 GB/s of an assumed 120, and the per-step sync puts a bubble at every step boundary (removing it was worth 6.6%). [measured]
Day 3's `util_proxy` is likewise a model quantity, not a hardware counter. Better signals: achieved bandwidth and FLOP/s versus peak, kernel-level occupancy, and busy fraction with idle gaps from a timeline (`gap_analysis`, needs device events). [measured A40: nsys and torch-profiler busy fraction 13% (0.5B) to 69% (7B), report 9.5]

## 3. How do you identify kernel launch overhead?

Ask whether the CPU can feed the device fast enough.
- **Enqueue vs synced step time** (`launch_probe.py`): time the forward call returning without a sync, and again with one. Ratio near 1 means CPU-bound. Here: 3.0 ms enqueue vs 15.9 ms step = 0.19, so **not** launch-bound; this refuted Day 1's guess. [measured]
- **Timeline gaps**: many short kernels separated by idle device time, and CPU-side `cudaLaunchKernel` rows lining up with the gaps (nsys / Chrome trace; `gap_analysis` computes busy fraction and gap percentiles). [measured A40: nsys sees the same 33,760 kernels as the torch profiler, GPU busy 13% of the 0.5B decode window, report 9.5]
- **Counts**: thousands of tiny ops per step (4,470 `aten::` ops per decode step here) make launch cost plausible; the profiler gives op counts but on MPS only CPU-side times. [measured]
- **Fix check**: if CUDA graphs or fusing ops moves step time, it was launch overhead; a bigger batch should then add little time. [mechanism]

## 4. Why might increasing batch size improve throughput but worsen latency?

Decode reuses one weight read across all sequences, so throughput rises while per-step time grows only a little; but each sequence now shares the step, and prefill (compute-bound) does not amortize.
Measured at 8 requests, batch 1 -> 8: output tok/s up 2.8-3.5x, but median TPOT up 31-44% (e.g. 15.75 -> 22.47 ms). [measured]
Throughput gain stops short of the decode-only 5.6-6.1x because prefill is about 15% of wall time at B=1 and 47-59% at B=8 (derived from medians). [measured]
P95 TTFT *fell* (9.5 s -> 2.7 s) only because with all requests arriving at once, batch 1 queues the last request behind seven others; under light load batching would instead add waiting time to fill a batch. [measured, queueing effect]
Also: bigger batches grow KV memory (967 -> 1024 MB here) and preemption risk. [measured / E6 simulated]

## 5. How do you compare two inference engines fairly?

Hold constant: model weights and dtype/quantization, hardware and clocks, prompts and output lengths (seeded, `ignore_eos` so lengths match), sampling (greedy), request pattern (open- vs closed-loop, same rate and concurrency cap), and metric definitions (vLLM's TPOT excludes the first token and its percentile choices differ by tool). Then:
- warm up; interleave runs and **rotate their order** (a fixed order created a fake 13-55% TTFT gap in my first harness); fresh server per run to avoid prefix-cache inflation; [measured: harness bug; mechanism for servers]
- run an **A/A control** (same engine twice) to get the noise floor (it was -0.1% for medians, but +5.1% for P95 TPOT); [measured]
- **validate outputs** match (token equality) before comparing speed; [measured]
- repeat across sessions and report ranges; call a difference real only if repeat ranges do not overlap and it exceeds the A/A floor. [measured]
Not done here: a cross-engine comparison (only variants of one engine).

## 6. How would you investigate a sudden increase in P99 latency?

1. **Confirm it is real**: same percentile definition, enough samples, not a measurement change. (Two of my own "findings" were harness artifacts, and one session ran long prefills 50% slower with no code change.) [measured]
2. **Locate it**: split end-to-end into queueing, TTFT and TPOT, and look at whether it is all requests or a class (long prompts, a tenant). Head-of-line blocking from long prefills is the classic one (Day 3 sim). [mechanism / simulated]
3. **Check the engine's signals**: waiting-queue depth, KV-cache utilization, preemption/recompute count (`vllm:num_preemptions`), batch size over time. [measured A40 with a 5 Hz `/metrics` sampler: report 9.3 and 9.7]
4. **Check the hardware and host**: clocks/thermal throttling, GPU memory pressure, CPU contention or GC pauses delaying launches (`enqueue_fraction`), noisy neighbours. [measured on Mac for thermal-like drift; on the A40 the same 0.5B loop's step time drifted 15.8-25.7 ms between hosts and passes, report 9.1/9.5]
5. **Take a trace of a bad window** (profiler delay/max-iterations window in the runbook) and compare with a good one: gaps, new syncs, allocator retries. [measured A40: the 3584-token ITL anomaly was explained this way, report 9.6]
6. **Bisect the change** (deploy, config, traffic mix) and re-run with an A/A control to be sure the fix, not noise, moved the tail.
