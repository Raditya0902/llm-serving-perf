# Day 6 performance report: profiling the Day 1 inference loop

> **What is and is not measured.** Everything marked *measured* was run on an Apple M4 (MPS, fp16,
> Qwen2.5-0.5B-Instruct, Day 1's hand-written cached greedy loop). **No NVIDIA GPU was available:**
> there is no nsys trace, no vLLM profile, no GPU peak-memory figure, and experiments E4-E6 are
> **simulated predictions only** (Day 3 simulator, illustrative cost model). Cells that need a GPU say
> *not measured*. The GPU procedure is in [RUNBOOK.md](RUNBOOK.md), unrun.

## 1. Environment and method

| | |
|---|---|
| Machine / software | Apple M4, macOS 27.2.1, torch 2.15.0, transformers 5.23.0, fp16, MPS |
| Code state | repo commit `9d8d845` plus uncommitted Day 6 files (`git_dirty: true` in every result file); Day 1 code unchanged |
| Workload | prompt 512 tokens, 64 new tokens (EOS suppressed so every run generates exactly 64), greedy |
| Repeats | 2 warmups per case, then interleaved timed rounds; repeats rounded up so every case holds every round position equally (6-8 timed repeats per case); case order rotates each round |
| Validation | every row records the fraction of generated token rows identical to the reference (all rows below: **1.00**); an A/A control case (identical work) gives the noise floor |
| Raw data | [results/measured/](results/measured/) (JSON + CSV per run, with config fingerprint and git hash); traces (38-50 MB) stay in gitignored `results/raw/` |

**Measurement problems found and fixed on the way (and why they matter).**
The first harness ran each round in a fixed case order after a 5 s sleep. The first case in each round
paid an idle-to-active GPU ramp: `day1` and `deferred_sync` run *identical* prefill code but showed TTFT
244 ms vs 157 ms, and the apparent "+16%" for deferring syncs was mostly that artifact
([results/measured/mps-v1-position-biased](results/measured/mps-v1-position-biased/)). Fixes: rotate case order,
balance repeats, drop the in-experiment sleep, add the A/A control. Those first runs are kept only as evidence.

## 2. Baseline table

Baseline = Day 1's loop. Modified = the same loop with per-step host syncs removed (O1, section 5).
Both rows come from the same interleaved run ([generated](results/performance-tables-o1.generated.md)).

| Metric | Baseline (`day1`) | Modified (`deferred_sync`) | Difference |
|---|---|---|---|
| Requests/sec | 0.880 | 0.938 | +0.058 (+6.6%) |
| Output tokens/sec | 56.3 | 60.0 | +3.7 (+6.6%) |
| P95 TTFT (ms) | 161.0 | 159.7 | -1.4 (-0.9%, prefill is unchanged) |
| P95 TPOT (ms) | 15.83 | 14.58 | -1.25 (-7.9%) |
| Peak memory (MB, live tensors on MPS) | 967.2 | 967.2 | +0.0 |
| Peak **GPU** memory (nvidia-smi) | not measured | not measured | not measured |

Reading it: A/A control (`day1_control`) matched the baseline within -0.1% requests/s, with repeat ranges
that do not overlap the modified row's (0.862-0.884 vs 0.925-0.946 req/s). **P95 TPOT is the noisiest column**: the
control moved +5.1% on it, so the median TPOT change (15.52 -> 14.43 ms, -7.0%) is the more reliable figure.
Tokens identical in all repeats.

A second comparison, concurrency (E1, 8 requests, batch 1 vs batch 8, run `mps-r2`, [generated](results/performance-tables-e1.generated.md)):

| Metric | Batch 1 | Batch 8 | Difference |
|---|---|---|---|
| Requests/sec | 0.858 | 2.676 | +1.82 (+212%) |
| Output tokens/sec | 54.9 | 171.3 | +116 (+212%) |
| P95 TTFT (ms) | 9503 | 2736 | -6767 (-71%) |
| P95 TPOT (ms) | 22.86 | 24.98 | +2.1 (+9%) |
| Peak memory (MB) | 967 | 1024 | +57 (+6%) |

**Caveat:** this run's repeat spread was 43-54% and the batch-8 gain ranged 2.8-3.5x across four independent
runs, so quote the ratio range (section 4), not these single-run P95 values.

## 3. Profiling findings (Session 2)

Source: [results/measured/mps-baseline/profile_summary.json](results/measured/mps-baseline/profile_summary.json),
[mps-eager](results/measured/mps-eager/profile_summary.json); figure `results/profile_breakdown.png`.
**Limitation that shapes everything here:** on MPS, `torch.profiler` records CPU-side operators only. 86% of decode
CPU self-time is the single `.item()` wait, so profiler op times say nothing about device time per op. Device time was
obtained differently: matmuls by *replaying* every `nn.Linear` with its real captured input back-to-back (one sync), attention
core by sync-bracketing `self_attn` minus the projections minus a calibrated 169 us per bracketed call (median of 3 passes). The matmul
numbers are solid; the attention-core number is an **estimate** (uncertainty of order 1 ms; the code flags the split if the pieces fail to reconcile, and they did reconcile here).

| Question | Answer (sdpa, 512-token prompt) |
|---|---|
| Which operators consume the most time? | Decode step 15.9 ms. MLP linears 7.2 ms, `lm_head` 2.9 ms, q/k/v/o projections 1.7 ms: all matmuls. The profiler's top CPU op is `aten::_local_scalar_dense` (the `.item()` wait), which is the CPU waiting for the device. |
| Time on matrix multiplications | Decode **11.8 ms = 74%** of the step; prefill 127 ms = 79%. |
| Time on attention | Decode core about **3.1 ms = 19%** (estimate); prefill about 14 ms = 9%. Attention *module* total including projections is larger. |
| Gaps between GPU operations | **Not observable on MPS** (no device timeline, `gaps.available=false`). Indirect: the CPU enqueues a whole step in 3.0 ms of a 15.9 ms step (19%), so the queue is never starved *within* a step. Day 1's per-step sync does create an idle bubble at each step boundary; O1 bounds it at about 1.1 ms/step (7%). |
| Memory allocations during generation? | Yes, and they are exactly the KV cache: live device bytes grow by **12,288 B/step** in 32 of 32 steps, equal to the analytic KV size per token (`growth_matches_kv=true`). The profiler's `[memory]` events on MPS are small host-side allocations (3,328 per 32 steps, 20 KB total) not device memory. Transient device allocations that are freed within a step are invisible to the live-bytes counter. |
| Is the CPU delaying the GPU? | **No.** `launch_probe`: enqueue 3.0 ms vs synced step 15.9 ms (fraction 0.19, "device-bound"). Eager attention: 5.4 / 19.7 ms (0.27), still device-bound. |

**Day 1's open question is answered, and its guess was wrong.** Day 1 measured about 16 ms/token against an assumed
8 ms bandwidth floor (0.99 GB of weights at 120 GB/s, both figures from Day 1's README, not re-measured here) and guessed per-layer launch overhead. Launch overhead
is not it. The matmuls alone take 11.8 ms (effective 51-93 GB/s depending on layer size: MLP 88, `lm_head` 93, small
projections 51; weights per token: MLP 627 MB, `lm_head` 272 MB, projections 88 MB), about 3.5 ms above the floor; attention core plus everything else is about 4.1 ms. The gap is split
between matmul efficiency and non-matmul device work, not CPU launch cost.

## 4. Experiments (Session 3): hypothesis, result, evidence

Hypotheses were fixed in [experiments.py](experiments.py) before the runs and not edited afterwards.
"Run" labels: v1 position-biased (not used for conclusions), v2 unbalanced, v3 = main run (`mps`), r2 = independent rerun of E1/E2.

| Exp | Hypothesis | Verdict | Evidence |
|---|---|---|---|
| E1 concurrent requests (batch 1-8, 8 requests) | step time at B=8 <= 1.5x B=1; output tok/s at B=8 >= 5x | **Step time: confirmed. Throughput: refuted.** | TPOT p50 ratio B8/B1: 1.44 / 1.41 / 1.31 / 1.43 across v1/v2/v3/r2. Throughput ratio only 3.5 / 2.8 / 3.0 / 3.1x. All batch outputs equal their single-request tokens. |
| E2 prompt length -> TTFT | TTFT(2048)/TTFT(512) in [3.8, 5.0]; prefill tok/s about flat | **Mostly confirmed, with an unstable tail.** | TTFT p50 128/512/1024/2048 = 50/162/320/708 ms (r2); ratio 4.36 (r2), 4.84 (v3), 6.61 (v2). Prefill tok/s 2.6k/3.2k/3.2k/2.9k (r2): flat from 512 up, lower at 128. |
| E3 output length -> TPOT | TPOT changes <= 5% between 32 and 256 tokens | **Confirmed.** | TPOT p50 15.78 / 16.04 / 16.04 ms (v3): +1.6%; v2: -2.4%. Total latency linear (0.66 / 2.20 / 4.26 s). |
| E4 max batched tokens | small budget: lower P95 ITL, higher TTFT; throughput plateaus ~2048+ | **Simulated only.** Not a result. | Day 3 sim: throughput 2440 / 2368 / 2343 / 2339 tok/s for budgets 512 / 2048 / 8192 / 16384 (flat within 4%); P95 TTFT flat (1.9-2.0 s); P95 token gap non-monotonic: 20.8 / 82.2 / 12.9 / 12.9 ms (not investigated). |
| E5 max active sequences | throughput up and TTFT down until compute/KV binds | **Simulated only**; direction as predicted. | 16 / 64 / 256 seqs: 1345 / 2347 / 2343 tok/s; P95 TTFT 8.4 / 2.2 / 1.9 s. Plateau by 64. |
| E6 KV capacity | preemptions only below peak KV demand; throughput falls, TTFT rises | **Simulated only**; direction as predicted. | KV blocks 8192 / 4096 / 2048 / 1024 / 512 / 256: preemptions 0 / 0 / 7 / 17 / 27 / 27; throughput 2343 / 2343 / 2120 / 1505 / 946 / 512 tok/s. |
| O1 defer host syncs | improvement < 5% (premise: CPU-bound) | **Refuted, narrowly, and the premise was wrong.** | +6.6% req/s (v3), +9.6% (v2); A/A control -0.1%; ranges disjoint; tokens identical. The loop is device-bound (probe), so the gain is not "CPU was the bottleneck". |
| O2 sdpa vs eager attention | differ < 5% at decode | **Refuted.** eager is slower. | TPOT 20.68 vs 15.50 ms (+33%), TTFT +25%, tokens identical, A/A -0.1%. Profile: attention core 6.7 vs 3.1 ms, matmul unchanged (11.4 vs 11.8 ms), 5,895 vs 4,470 aten ops/step (+32%), 4,928 vs 3,328 allocation events (+48%). |

**Why E1's throughput gain is only about 3x.** Decode batches well: decode-only throughput (rows per step time) is 5.6-6.1x at B=8, meeting the hypothesis.
But prefill is compute-bound and does not amortize: derived from the medians, prefill is about 15% of wall time at B=1 and 47-59% at B=8, so Amdahl caps the end-to-end gain.
(Derived figure, not separately timed.) The P95 TTFT *drop* with batching is queueing: at B=1 the last of 8 requests waits for seven others.

**Reproducibility.** Within a session, interleaved comparisons are tight (O1/O2 repeat spread 2-8%; A/A -0.1%). Across sessions, the same
configuration moves about 5% (B=1 output tok/s 49.2-54.9; E3 TPOT 15.9-16.7 ms), and one session (v2) ran long prefills about 50% slower
(2048-token TTFT 1128 vs 708 ms) with no change in short ones. E1 repeats were unstable in the main run (spread up to 55%, drift 14-41%) and again in the rerun (43-54%), so
E1's *medians* are consistent across runs but its P95 values are soft. O1's +6.6% is established within a session; its size
is close to cross-session variation, which is why it was checked in three sessions: positive in all three (+16% biased, +9.6%, +6.6%).
Possible causes of the late-session instability (thermal state, background load) were not isolated.

## 5. Interpretation

**Primary bottleneck:** decode is device-bound on weight reads. Matmuls are 74% of a 15.9 ms step and run at 51-93 GB/s,
a fraction of the assumed 120 GB/s peak. The CPU is not the limit (enqueue is 16% of the step), and launch overhead,
the Day 1 hypothesis, is refuted. The next-largest cost is the attention core (about 19%, estimated), which the choice of implementation moves by about 3.6 ms (sdpa vs eager).

**Did the optimization improve the system?** Partly, and modestly.
- *Deferring host syncs (O1)* gave a real, validated **6.6% throughput / 7% median TPOT** improvement with identical outputs, at the cost that the loop can no longer stop on EOS without a sync
  (it generates a fixed length). It does not touch the primary bottleneck, so it is a small win, not a fix. Plausible mechanism, not isolated: removing two host round-trips per step (calibrated about 0.17 ms each) and letting the CPU enqueue the next step while the GPU finishes the current one.
- *Keeping sdpa (O2)* is the right default: eager costs +33% TPOT.
- *Batching (E1)* is the only change that moves throughput substantially (2.8-3.5x at B=8) and it is not free: per-request TPOT rises 31-44%, and a request's latency now depends on its batch.
- *Not tried (would attack the real bottleneck):* weight quantization (int8/4-bit reduces bytes read per token) and prefix caching. Not claimed.

## 6. Limitations

- Single model, single machine; unified-memory Apple GPU, so findings about bandwidth and sync cost do not transfer to discrete NVIDIA GPUs.
- Device-time attribution on MPS is indirect (replay and sync-bracketing). The attention-core figure is an estimate; the 120 GB/s peak is Day 1's assumed figure.
- E4-E6 are simulated with illustrative constants (8B/A100-class guess), so only direction means anything; vLLM behaviour is untested here.
- Mac thermals and background load were not controlled; long sessions showed drift.
- P95 over 6-8 repeats of a single request (E2, E3, O1, O2) is effectively the maximum; only E1 pools 48-64 request samples. Prefer the medians and ranges for those rows.
- E1's static batches model concurrency as "all requests arrive at t=0", not an open-loop arrival process.
- All token-equality checks are greedy and fp16 on one device; they show identical outputs here, not in general.

## 7. Review notes

The code was reviewed (code-reviewer agent) after the first results. Fixes that affect how to read the numbers: the nsys window
was documented but never opened (now `--nsys` + `cudaProfilerApi`, still unrun); diverged outputs now set `valid=false` and
warn instead of printing a clean speedup; the report generator now requires the experiment's own baseline row and recomputes the
range overlap from the two rows it compares; attribution now flags clamped/negative pieces. The measured results above were not re-run for the
report-generator and warning changes (they do not alter the measurements); the profiles were re-run after the attribution change,
which is why the profile numbers differ slightly from my first pass.

## 8. Reproduce

```bash
cd day6-profiling && uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python profile_run.py --device mps --tag mps-baseline                    # profile (sdpa)
.venv/bin/python profile_run.py --device mps --attn-impl eager --tag mps-eager
.venv/bin/python run_local_experiments.py --exp all --device mps --tag mps         # E1-E3, O1, O2 (~25 min)
.venv/bin/python sim_predictions.py                                                # E4-E6, SIMULATED
.venv/bin/python make_performance_report.py --raw results/raw/mps --baseline O1:day1 --modified O1:deferred_sync
.venv/bin/python plot_results.py && .venv/bin/python -m pytest --cov=.
```

Expect your absolute numbers to differ by about 5% across sessions; the verdicts above should hold.
