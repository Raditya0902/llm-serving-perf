# Round 5 pre-registration (written before any round-5 run)

Predictions and decision rules for the items still open after round 4 (performance-report 9.13). Hardware: Lambda Cloud. A section is
added per pod before that pod is used; results and refutations are appended under "Outcomes" afterwards, with errors I made.
Status of Nsight Compute counters on a Lambda VM is unknown (RunPod blocked them); item A starts with a gate and says so if blocked.

## Pod 1: single RTX A6000 (GA102, sm86, the A40's chip with 768 GB/s)

### A. Nsight Compute counters on FlashAttention 2 (vLLM's attention call, `gpu/attn_microbench.py` under `ncu`, 0.5B geometry GQA 7 and 8B geometry GQA 4)

Quantities per launch: `launch__grid_size`, `launch__waves_per_multiprocessor`, `dram__bytes_read.sum`, duration. Batches: pure decode (256 decodes), prefill only (7,600 tokens), mixed (both). Contexts 512 and 3,584.

| # | Prediction | Decision rule |
|---|---|---|
| A1 (9.9 byte story) | At ctx 3,584 the decodes in the mixed launch stream the KV once per query head: (DRAM bytes mixed - DRAM bytes prefill-only) / DRAM bytes pure decode is about the GQA group size | Held if the ratio is in [5, 9] for GQA 7 and [3, 5] for GQA 4. Refuted if it is below 2 for either (then time grows without bytes: the byte story is wrong) |
| A2 (9.12 Q4b block launch) | The mixed launch has far more thread blocks than the pure decode launch because the grid is sized by the longest query | Held if grid(mixed) >= 10 x grid(pure decode) at both contexts. Refuted if < 3 x |
| A3 (Q4b: hidden at long context) | Achieved DRAM throughput of the mixed launch is a larger share of peak at ctx 3,584 than at ctx 512 | Held if (mixed share at 3,584) > (mixed share at 512); no threshold |

If the attention kernel is launched as more than one kernel per call, I add the launches of that call; the number of kernels per call is reported.

### B. Busy fraction at 256 vs 1,024 running sequences (Nsight Systems, cuda trace only, steady-state window of 150 engine iterations after a delay of 200 or 600; 0.5B; 512x128 and 128x128; FA2 and TRITON_ATTN)

Round 3 measured 0.81-0.83 (256 seqs) vs 0.60-0.77 (1,024) with the torch profiler for FA2. H2 (per-step host cost that grows with running sequences) predicts idle time rises with the limit; the attention penalty predicts device time rises with no extra idle.

| # | Prediction | Decision rule |
|---|---|---|
| B1 | FA2 @ 512x128: busy fraction falls by >= 10 points from 256 to 1,024 seqs (replicates round 3 with a different tracer) | Held >= 10 points; refuted < 5 |
| B2 | TRITON @ 512x128: busy fraction changes by less than 5 points either way (throughput recovered at this cell, so no extra idle) | Held < 5 points in magnitude |
| B3 | TRITON @ 128x128 (where Triton still lost 5.7% at 1,024 seqs on the A40): busy fraction falls >= 5 points, i.e. the remaining loss is host-side (H2) | H2 supported for that cell if the fall is >= 5 points; **H2 not supported** if the magnitude is < 2 points (loss is in device time); 2-5 inconclusive |

Traced throughput is compared with the untraced backend A/B run on the same pod (step `ab`, 4,000 prompts, 3 repeats) to bound what tracing cost. The A6000 is not the A40; if its A/B does not reproduce the A40 direction (Triton gain at 1,024 seqs, loss at 128x128), B3 is read against the A6000's own A/B, not the A40's.

### C. Triton output check at 1.5B, 3B and 8B (A6000; `gpu/quality_run.py` + `gpu/quality_compare.py`, the five criteria of round 4 unchanged, 200 prompts, 64 new tokens, greedy; fp32 HF reference on 20 prompts)

Prediction: all five criteria hold at 1.5B and 3B (as at 0.5B). At 8B they hold too, but the margin is thinner (larger models have more near-ties in absolute terms and bf16 accumulation differences grow with depth): I put 70% on all five holding at 8B. Any failed criterion is reported with its number; a failure at 8B only does not change the verdict on 1.5B/3B.

## Pod 2: one H100 80GB SXM5 (Hopper, sm90; FlashAttention 3 is vLLM's default there)

Backends: `FLASH_ATTN:3` (default on Hopper), `FLASH_ATTN:2` (forced), `TRITON_ATTN`. Round 4 on an H100 PCIe (0.5B): FA3 6,523 / FA2 3,568 / Triton 4,277 tok/s at 3584x128; at 1,024 seqs and 512x128 Triton lost 5.3% against 256 seqs and FA3 1.7%.

| # | Prediction | Decision rule |
|---|---|---|
| D1 (counters) | FA3's mixed launch does not stream the KV once per query head: (DRAM bytes mixed - prefill-only) / pure-decode bytes is about 1 for FA3 and about the GQA group size for FA2 forced on the same GPU | Held if FA3's ratio is in [0.7, 2] and FA2's in [5, 9] (GQA 7) / [3, 5] (GQA 4). Refuted for FA3 if its ratio is above 3 |
| D2 (Triton check) | Triton is output-equivalent to the default FA3 path at 0.5B, 1.5B, 3B and 8B under the five round-4 criteria (65% that all four sizes hold: FA3 is a different kernel family from FA2, so the noise floor is the same-kernel control, not FA2) | Per size: all five hold = equivalent; any failure is reported with its number |
| D3 (busy fraction, 0.5B, 512x128, 256 -> 1,024 seqs) | The attention penalty shows as device time, not idle: FA2 forced keeps its busy fraction within 5 points while throughput falls. A host-side cost shows as idle: Triton's busy fraction falls >= 5 points where its throughput falls | Triton: H2 supported if busy falls >= 5 points and tok/s falls; not supported if |change| < 2 points with tok/s falling (loss then in device time). FA2 forced: held if |change| < 5 points. FA3: |change| < 5 points |
| D4 (backend matrix, 1.5B / 3B / 8B) | FA3 is the fastest or within 2% of the fastest backend at every cell; FA2 forced is at least 5% slower than FA3 at 1.5B 3584x128 | Held / refuted per cell, counts reported |

## Pods 3 and 4: the same A100 GPU with two interconnects (2x A100 40GB PCIe host; 8x A100 40GB SXM4 host, GPUs 0 and 1 only)

Driver: `07-capacity-study/run_round5_a100.sh` (Llama-3.1-8B ungated mirror, workload B, c16 and c64, 2 repeats, real 40 GB pools, no `--cap-kv-pool`). Arms: TP=1, TP=2, TP=2 with `--disable-custom-all-reduce`, two TP=1 replicas (data parallel, c8+c8 and c32+c32), plus on the NVLink host the no-custom TP=2 arm with `NCCL_P2P_DISABLE=1` (traffic through host memory). Reference points from earlier rounds: 2x A40 PCIe, replicas vs TP=2: -13.5% at c16, +31.3% at c64; A100-80GB NVLink, TP=2 over TP=1: +61.7% (c16) and +76.5% (c64), TPOT step speedup 1.77x. The two hosts are different machines (CPU, PCIe root, RAM), so the pair isolates the interconnect only up to that caveat.

| # | Prediction | Decision rule |
|---|---|---|
| E1 (8B data-parallel on NVLink) | Replicas beat TP=2 at c64 (a 40 GB TP=1 pool is about 120k tokens; each replica holds 32 sequences and avoids most preemption): +10% to +40% output tok/s. At c16 TP=2 wins (replicas -5% to -20%) as on PCIe A40 | Sign and range per load; the c64 and c16 signs must both match to call the A40 pattern reproduced |
| E2 (TP=2 step speedup, NVLink) | TPOT TP=1 / TP=2 at c16 is 1.6-1.9x | Held inside the range; the A100-80GB value was 1.77x |
| F1 (interconnect effect on TP=2) | TPOT speedup at c16 is higher on NVLink than on PCIe by at least 0.25x (PCIe 1.2-1.5x) | Held if NVLink - PCIe >= 0.25x; refuted if < 0.1x |
| F2 (custom all-reduce) | Worth +1% to +4% throughput at c64 on NVLink and 0% to +2% on PCIe | Reported per host; refuted if the PCIe gain exceeds the NVLink gain |
| F3 (NCCL over host memory on the NVLink host) | TP=2 no-custom with `NCCL_P2P_DISABLE=1` is at least 5% slower than no-custom over NVLink at c16 and lands within 10% of the PCIe host's no-custom arm | Both clauses must hold |
| F4 (link probe) | Bus bandwidth at 256 MiB: NVLink >= 150 GB/s, PCIe <= 30 GB/s; 8 KiB all-reduce latency <= 30 us on NVLink | Held per number |
| F5 (real 40 GB pool) | TP=1 at c64 preempts on the real 40 GB card (the Day 7 capped emulation predicted 64 preemptions per run); TP=2 does not | Held if TP=1 preemptions >= 20 per run and TP=2 has 0 |


---

# Outcomes (appended after the runs; numbers, tables and caveats in performance-report.md 9.14)

Data: `results/measured/a6000-round5/`, `results/measured/h100-round5/`, `../07-capacity-study/results/measured/round5-a100-pcie/`, `../07-capacity-study/results/measured/round5-a100-sxm4/`.

## Pod 1 (RTX A6000)

- **A1, A2, A3 held.** The extra DRAM bytes of the mixed launch over the prefill-only launch, divided by the pure-decode bytes, are 7.04 (0.5B, GQA 7) and 4.11 (8B, GQA 4) at ctx 3,584, and 7.40 / 4.14 at ctx 512: the GQA group size, so the 9.9 byte story is observed, not inferred. The mixed launch has 836x (0.5B) and 478x (8B) the blocks of the pure-decode launch (428,162 vs 512; 978,656 vs 2,048). Achieved DRAM share of the mixed launch is 71.6% at ctx 3,584 against 19.9% at ctx 512 (0.5B), 84.3% against 26.7% (8B): the penalty is hidden at long context, as Q4b said.
- **B1 refuted, B2 refuted, B3 not testable.** B1: FA2 at 512x128 did not lose busy fraction from 256 to 1,024 seqs; the clean re-run gave 0.72 -> 0.84 (a rise). B2: Triton 512x128 0.68 -> 0.79 (+10.7 points, not within 5). B3 could not be tested because the untraced A/B on the A6000 did not reproduce Triton's 128x128 loss at 1,024 seqs. Net: H2 (host idle growing with running sequences) is **not supported** for 512x128 on this GPU; the loss there is device time. The measure is noisy (window to window +-0.07, up to 0.2) and it is a busy fraction by interval union, not the step-based metric of round 3; the first runs had windows shorter than the run's delay and are kept in `busyfrac_run1_short_windows/` and `busyfrac/`, the valid 1,024-seq re-runs are in `busyfrac_1024b/`.
- **C held at 1.5B, 3B and 8B (sm86).** All five criteria hold; the fp32 reference errors are equal (0.0272 vs 0.0273; 0.0252 vs 0.0253; 0.0191 vs 0.0191); exact match is 61.5% / 58.0% / 52.5% against the FA2-vs-FA2 control's 59.5% / 61.5% / 51.5%. The 8B margin was as thin as I feared (near-tie share 98.9%) but not crossed.

## Pod 2 (H100 SXM5)

- **D1 held.** FA3's mixed launch is a persistent grid of 132 blocks (the SM count) for both contexts and both models, bytes ratio 1.00; FA2 forced on the same GPU shows 7.05 / 4.06 (GQA 7 / 4) and 836x / 478x the blocks. Penalty in time (mixed / pure): FA2 11.0 and 5.3 at ctx 3,584, FA3 3.1 and 1.7.
- **D2 refuted as written, with a caveat.** Under the five criteria Triton against FA3 passes at 8B and fails at 0.5B, 1.5B and 3B. The failures come from the criteria's yardstick, not from a bigger error: the control (FA2 against FA2 with a different token budget) is nearly exact on this GPU (78.5% / 90% / 73% exact match, prompt-logprob difference 0.0005 / 0.0002 / 0.0001), so "within 2x of the control" cannot be met by any different kernel; the fp32 reference errors are equal at all four sizes (0.0281 vs 0.0281; 0.0265 vs 0.0267; 0.0247 vs 0.0244; 0.0193 vs 0.0194) and every divergence is a near-tie (97.4-100%). I report D2 as refuted and add that the same-kernel control makes criteria 1-3 unsatisfiable on this GPU. This is a finding about the test, not evidence that Triton is wrong.
- **D3 refuted** (busy-fraction predictions, 0.5B on the H100): FA2 forced and FA3 busy fractions moved by more than 5 points between 256 and 1,024 seqs (FA2 512x128 0.40 -> 0.91; FA3 128x128 0.20 -> 0.40); the windows are noisy (0.12-0.36 on one cell) and the H100 is mostly idle at this model size, so the cell does not separate H2 from the attention penalty.
- **D4 held.** FA3 is within 2% of the fastest backend at all 6 cells (best at 4 of them; Triton ahead by 0.5% and 1.2% at 8B), and FA2 forced is 44.1% slower than FA3 at 1.5B 3584x128 (also -6.8% to -32.1% at the other cells).

## Pods 3 and 4 (the same A100 GPU with two interconnects)

Hosts: 2x A100 40GB PCIe (topology PHB, no NVLink) and 8x A100 40GB SXM4 (NV12, GPUs 0 and 1). Llama-3.1-8B, workload B, 2 repeats, 24 runs on the NVLink host and 20 on the PCIe host, all with 0 failed requests; real 40 GB pools (KV 136,192 tokens on TP=1, 388,096-398,336 on TP=2).

| arm (median of 2) | PCIe c16 TPOT / tok/s | PCIe c64 | NVLink c16 | NVLink c64 |
|---|---|---|---|---|
| TP=1 | 25.4 ms / 599 | 57.4 ms / 816 | 23.6 ms / 645 | 51.1 ms / 915 |
| TP=2 | 36.4 / 413 | 123.2 / 505 | 14.3 / 1,063 | 36.5 / 1,705 |
| TP=2, no custom all-reduce | 36.3 / 415 | 124.0 / 503 | 14.9 / 1,015 | 37.3 / 1,665 |
| 2 replicas (aggregate of both) | 18.8 / ~800 | 41.6 / ~1,480 | 17.9 / ~841 | 37.6 / ~1,635 |
| TP=2 no custom, `NCCL_P2P_DISABLE=1` | n/a | n/a | 46.0 / 328 | 158.6 / 392 |

- **E1 partly refuted.** NVLink: replicas vs TP=2 at c64 is -4.1% (1,635 vs 1,705), not +10..+40%: on NVLink the replicas do not win, the two are level. At c16 TP=2 wins by 26% (replicas -20.9%, just outside -5..-20). So the A40 pattern (replicas win at c64) is **not reproduced** on NVLink. The replicas win everywhere on the PCIe host (+94% at c16, +193% at c64), but only because TP=2 is slower than one GPU there. The reason replicas can only tie TP=2 here: TP=1 at c64 preempts 57-66 times per run (F5), but each replica at c32 does not (peak KV 0.70, 0 preemptions), and a replica's TPOT at c32 (37.6 ms) equals TP=2's at c64 (36.5 ms).
- **E2 held on NVLink** (TPOT TP=1 / TP=2 = 1.65x at c16, inside 1.6-1.9; 1.40x at c64). On the PCIe host it is 0.70x at c16 and 0.47x at c64: TP=2 is slower than one GPU.
- **F1 held.** NVLink minus PCIe speedup at c16 = 1.65 - 0.70 = 0.95x (rule: >= 0.25x). My expectation that the PCIe host would be 1.2-1.5x was wrong: it is below 1.
- **F2 held.** Custom all-reduce, throughput: NVLink +2.4% at c64 (+4.7% at c16), PCIe +0.4% at c64 (-0.5% at c16): inside +1..+4% and 0..+2%.
- **F3 refuted (second clause).** `NCCL_P2P_DISABLE=1` on the NVLink host is 3.1x slower than NVLink at c16 (first clause held) but also 27% (c16) and 28% (c64) slower than the PCIe host's no-custom arm (46.0 vs 36.3 ms; 158.6 vs 124.0 ms), far outside the 10% band. The probe agrees: 2.2 GB/s through host memory on the NVLink host against 3.1 GB/s on the PCIe host.
- **F4 held for bandwidth, refuted for latency.** 256 MiB all-reduce bus bandwidth 158.1 GB/s (NVLink, rule >= 150) and 3.1 GB/s (PCIe, rule <= 30). 8 KiB latency 47.4 us on NVLink against the rule's <= 30 us (PCIe 45.1 us): at 8 KiB the probe sees launch overhead, not the link.
- **F5 held on both hosts.** TP=1 at c64 preempts 62 / 58 (PCIe) and 66 / 57 (NVLink) times with KV at 100%; TP=2 and every replica 0.
- On PCIe the `NCCL_P2P_DISABLE=1` probe is identical to the default one (3.0-3.1 GB/s): P2P is not effectively used on that VM either way.

Errors made on the way: the first attempt on the NVLink host used a hardware tag the driver did not accept (`a100-40gb-sxm4`), failed in 20 s without running a benchmark, and was re-run with `a100-40gb`; its probe-only output is kept in `sxm4_badtag_probe_only/` and not used. I first quoted the PCIe data-parallel throughput per replica (745 tok/s) as if it were the arm's total; the arm is two replicas run together and the aggregate is about 1,480 (E1 is judged on the aggregate). The two hosts are different machines (CPU, PCIe root, RAM), so the PCIe/NVLink pair isolates the interconnect only up to that caveat.
