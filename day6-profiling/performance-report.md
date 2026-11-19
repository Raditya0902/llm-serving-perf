# Day 6 performance report: profiling the Day 1 inference loop

> **What is and is not measured.** Sections 1-8 were measured on an Apple M4 (MPS, fp16,
> Qwen2.5-0.5B-Instruct, Day 1's hand-written cached greedy loop). **Section 9 adds a run on one NVIDIA A40
> (2026-11-18)** and it **reverses the headline Mac conclusion**: on CUDA the same loop is CPU-launch-bound,
> not device-bound, so statements in sections 3 and 5 about "device-bound" and "launch overhead refuted" hold
> **for MPS only**. Not run on the A40: nsys (not installed in the container) and the RUNBOOK section 5 sweeps
> (Day 4's 1000-prompt sweep covers E4-E6 instead, see 9.4). The simulated E4-E6 rows in section 4 are kept
> as the pre-registered predictions.

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

**Day 1's open question: answered for MPS only. On an NVIDIA A40 its guess was right (section 9.1).** Day 1 measured about 16 ms/token against an assumed
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

## 9. NVIDIA A40 measurements (2026-11-18)

Hardware/software: 1x A40 (48 GB, RunPod container), torch 2.14.0+cu132, transformers 5.22.2, vLLM 0.34.0, fp16 Day 1 loop (sdpa),
Qwen2.5-0.5B-Instruct, prompt 512, same harness as the Mac runs (interleaved repeats, A/A control, token-equality check). Data:
[results/measured/cuda](results/measured/cuda/), [cuda-baseline](results/measured/cuda-baseline/), [cuda-vllm](results/measured/cuda-vllm/).
Generated tables: [O1](results/performance-tables-o1-cuda.generated.md), [E1](results/performance-tables-e1-cuda.generated.md).

### 9.1 The Day 1 loop on CUDA is CPU-launch-bound (reverses the Mac result)

| Question | A40 answer (`profile_run.py --device cuda`, 512 prompt, 32 decode steps) |
|---|---|
| CPU delaying the GPU? | **Yes.** Re-checked with no profiler in the process (5 passes of `launch_probe`): enqueue 15.4-18.4 ms of a 15.8-18.8 ms synced step = **0.97-0.98, "cpu-bound"** (MPS: 0.19, device-bound). A loop with **no sync between 32 steps takes 15.3-15.7 ms/step in total, equal to its enqueue time**: the GPU is done as soon as the CPU stops issuing. (`profile_run.py`'s own probe ran after the profiler windows and reported 25.0 / 25.4 ms; that inflated figure is superseded, cause not isolated.) Step time moved 15.8 -> 18.8 ms between passes on this shared host, so quote the range |
| Gaps between GPU ops | Decode window (**profiler on**, so CPU-side overhead is inflated): 33,760 kernels (about 1,050 per step), GPU busy 12.8% of a 33 ms profiled step, gap p50 18.5 us, p95 80 us, max 2.07 ms. Kernel durations are not inflated by CPU overhead: about 4.2 ms of device work per step, i.e. **at most about 27% of an unprofiled 15.4 ms step**. Prefill window (512 tokens): busy 25%, 1,007 kernels, 34 ms window |
| Allocations | KV growth 12,288 B/step as on MPS, but `growth_matches_kv=false` on CUDA (only 16 of 32 steps show growth: the caching allocator hides per-step growth). Not investigated further |
| Caveat | The op classifier labels most CUDA kernels `other` (attention shows 0 device time), so category shares are not trustworthy on CUDA; the busy fraction and gap numbers do not depend on it |

This also explains two Day 1 oddities. Decode costs 16.2-16.5 ms/token at every batch size and every context length (below), and TTFT is a flat
17.5-18.3 ms from 128 to 1024 prompt tokens (it only rises, to 33 ms, at 2048). Both are a launch floor of roughly 16-17 ms set by the CPU issuing about
1,000 small kernels per forward, not by the GPU's work. The Day 1 README's "prefill grows linearly with prompt length" is not visible below 2048 tokens on this GPU.

### 9.2 Experiments E1-E3, O1, O2 on CUDA (hypotheses unchanged from `experiments.py`)

| Exp | Hypothesis | CUDA result (A40) | Verdict (MPS verdict for contrast) |
|---|---|---|---|
| E1 batch 1-8 | step time at B=8 <= 1.5x B=1; tok/s at B=8 >= 5x | TPOT p50 16.36 -> 16.57 ms (1.01x); output tok/s 60.8 / 121.1 / 236.0 / 463.9 for B=1/2/4/8 (**7.6x**); tokens identical | **Both confirmed** (MPS: step time confirmed, throughput refuted at 2.8-3.5x). Throughput scales almost linearly because a step costs the same regardless of batch: the loop is launch-bound |
| E2 prompt length -> TTFT | TTFT(2048)/TTFT(512) in [3.8, 5.0] | TTFT p50 17.7 / 17.5 / 18.3 / 33.1 ms for 128 / 512 / 1024 / 2048; ratio **1.89** | **Refuted**: launch floor hides prefill compute below 2048 tokens (MPS: ratio 4.4-6.6, mostly confirmed) |
| E3 output length -> TPOT | TPOT changes <= 5% from 32 to 256 tokens | 16.33 / 16.18 / 16.46 ms (+0.8%) | **Confirmed** (as on MPS) |
| O1 defer host syncs | improvement < 5% | +2.6% req/s, median TPOT 16.27 -> 15.85 ms; A/A control -0.6%; repeat ranges disjoint; tokens identical | **Consistent with the hypothesis** (MPS: refuted at +6.6%). The gain is small because the CPU enqueue, not the sync, is the long pole |
| O2 sdpa vs eager | differ < 5% at decode | eager: -19.5% tok/s, TPOT 20.21 vs 16.30 ms (+24%), TTFT 23.6 vs 17.9 ms; A/A +0.07%; tokens identical | **Refuted**, same direction as MPS. More ops per step costs more when launch-bound |

Single-request rows (E2, E3, O1, O2) use 6 repeats, so P95 is effectively the maximum; medians are quoted. Repeat spread was 1-4%.

### 9.3 vLLM on the A40: where the time goes in a step (torch profiler, RUNBOOK section 3)

Server: baseline flags (256 seqs, 8192 tokens), 1000-prompt burst, profiler window of 20 iterations after a delay
([window 150](results/measured/cuda-vllm/trace_window150/), [window 500](results/measured/cuda-vllm/trace_window500/)). Only the rank-0 operator tables are kept in git; the traces stay on the pod volume.

- **Pure decode steps are cheap, steps carrying any prefill are not.** Two profiled windows of 100 engine steps taken after the benchmark's warm-up request ([trace_step_costs_output.txt](results/measured/cuda-vllm/rerun/trace_step_costs_output.txt)): with `max_num_seqs=256`, 85 pure-decode steps (256 sequences) averaged **7.3 ms** of GPU time and 15 mixed steps (about 7k prefill tokens, 134 decodes) averaged **93 ms**. With `max_num_seqs=1024`, 34 pure-decode steps (999 sequences) averaged **26 ms** and 66 mixed steps (about 7k prefill tokens, 516 decodes) averaged **132 ms**.
- **Decodes riding in a mixed step cost about 4x more each.** Same prefill size (about 7k tokens), 134 vs 516 decodes: 93 vs 132 ms, so about 0.10 ms per extra decode sequence, against 0.026-0.029 ms per sequence in pure decode steps (7.3 ms / 256, 26 ms / 999). One sampled step with only about 1.5k prefill tokens but 995 decodes also took 133 ms (n=1). An earlier narrower window saw the attention kernel take 1.23 ms per layer call in mixed steps vs 0.16 ms in pure decode (2 steps), which fits. Rule of thumb that fits five vLLM configs: mixed step ms about 9 x (prefill tokens / 1000) + 0.1 x (running decodes) (e.g. bt=512 with 115 running: 16 ms predicted vs 19 ms median ITL; bt=1024 with 227 running: 32 vs 37; bt=2048 with 256 running: 44 vs 50). It is a rough fit, not a validated model.
- **This is the mechanism behind Day 4's ITL tails and the p50 ITL oddity.** Inter-token latency follows the step time, and a step that includes prefill is 5-18x a pure-decode step. The non-monotonic p50 ITL (bt=512 19 ms, bt=1024 37 ms, bt=2048 20 ms, bt=8192 11 ms) is explained by the *share of tokens emitted in mixed steps*: with budgets 512 and 1024 almost every step is mixed (88% and 90% of ITL samples exceed 15 ms), whereas with 2048 only 54% do, so the median falls back toward the pure-decode regime even though its mixed steps are slower than at 1024.
- **Measured (not computed) KV and concurrency from a 5 Hz `/metrics` sampler** (1000-prompt burst, [analysis_output.txt](results/measured/cuda-vllm/rerun/analysis_output.txt)): `max_num_seqs=256`: peak running 256, peak waiting about 485, **peak KV 5.3%**; `max_num_seqs=1024`: **peak running 996-999**, peak waiting about 550, **peak KV 21.5%**; budgets 512 / 1024 / 2048: peak running 115 / 227 / 256, peak KV 2.2% / 4.4% / 5.1%. 0 preemptions in all. These match Day 4's computed demand (5.3% and 21.8%).
- **Sampling is a large share of a pure-decode step** for this model: top-k/softmax/temperature/top-p kernels sum to roughly 3.3 of the 7.3 ms (about 45%, window 150), because the vocabulary is 151k and the model is tiny. Larger models would dilute this.
- **Hypothesis tested and refuted: the `max_num_seqs=1024` slowdown is not a CUDA-graph capture-range effect.** vLLM caps the captured size at 512 by default (51 full-graph sizes at 1024 seqs). Raising it with `-cc.max_cudagraph_capture_size=1024` (83 graphs), interleaved A/B, 3 fresh servers each, 1000 prompts ([data](results/measured/cuda-vllm/cg_test/)):

| Config | Output tok/s (3 runs) | p50 ITL | p99 ITL |
|---|---|---|---|
| `max_num_seqs=256` | 9,329 / 9,453 / 9,065 (mean 9,282) | 11 ms | 110-111 ms |
| `max_num_seqs=1024` | 8,512 / 8,432 / 8,515 (mean 8,487, **-8.6%**) | 39-42 ms | 432-481 ms |
| `max_num_seqs=1024` + capture size 1024 | 8,499 / 8,548 / 8,567 (mean 8,538) | 40 ms | 426-450 ms |

  The slowdown reproduces (-8.6% here, -11% in Day 4) and the graph setting does not move it. The slowdown is therefore not a graph effect. All 1000 requests are concurrent at 1024 (peak running 999) and the decodes spend more of the run riding in expensive mixed steps, which is consistent with the lower throughput, but I did not isolate it, so its cause stays an **open hypothesis**. A second interleaved pair with live metrics reproduced it: 256 seqs 9,445 / 9,457 tok/s vs 1024 seqs 8,218 / 8,581 (about -11%), p99 ITL 112-119 vs 415-423 ms.

### 9.4 Baseline vs modified on vLLM (RUNBOOK section 4) and the simulated E4-E6

`--max-concurrency 64`, 200 prompts, 3 repeats, fresh server each, 0 failed requests. Baseline `max_num_batched_tokens=8192`, modified 2048
([data](results/measured/cuda-vllm/baseline_vs_modified/)); mean [min-max]:

| Metric | Baseline (8192) | Modified (2048) |
|---|---|---|
| Output tok/s | 6,475 [6,461-6,484] | 6,510 [6,443-6,561] (within noise) |
| P50 / P95 / P99 TTFT (ms) | 274 / 490 / 515 | 260 / 493 / 550 (no resolvable difference) |
| P50 ITL (ms) | 5.4 | 5.4 |
| **P95 ITL (ms)** | **10.9** [9.3-12.1] | **22.9** [21.8-24.5] |
| **P99 ITL (ms)** | **55.8** [50.1-64.3] | **29.6** [29.5-29.8] |
| Peak whole-GPU memory | 38.9 GB | not summarized here |

A smaller budget **halved the p99 ITL but doubled the p95 ITL** at unchanged throughput. **Verified from the per-token ITLs** (`--save-detailed`, repeat 0 of each): with 8192, 4.5% of ITL samples exceed 15 ms and 1.1% exceed 50 ms; with 2048, 7.5% exceed 15 ms but 0.0% exceed 50 ms. With 8192 the prefill lands in a few huge steps that hit under 5% of tokens, but hit them hard; with 2048 it spreads over more steps, so more than 5% of tokens see a moderate stall. The p95 and p99 fall on opposite sides of that 5% line, so which
percentile you report decides which setting looks better. Peak GPU memory is the pre-allocated KV pool (0.85 x 44 GB), so it says nothing about live tensors.

E4-E6 against the Day 3 simulator, using Day 4's real 1000-prompt sweep (different workload and model than the simulator's 8B/A100-class guess, so only directions compare):

| Exp | Simulator prediction | Measured on the A40 (Day 4 data) | Verdict |
|---|---|---|---|
| E4 budget 512 -> 16384 | throughput flat within 4%; P95 TTFT flat | throughput 5,063 -> 9,578 tok/s (+89%); p99 TTFT 24.3 -> 11.9 s; p99 ITL 23.9 -> 179 ms | Simulator's **flat throughput and TTFT are refuted**; the ITL direction held |
| E5 seqs 16 / 64 / 256 | plateau by 64 | 3,513 / 6,926 / 9,445 tok/s (+36% from 64 to 256); at 1024 it drops to 8,399 | **Refuted** (no plateau at 64; a decline at 1024) |
| E6 KV capacity | preemptions below peak demand | no preemption in any run; KV pool was 3-19x above demand | **Untested** |

### 9.5 Larger models: the Day 1 loop stops being launch-bound between 3B and 7B (round 2, second A40 pod)

`profile_run.py --model <id>` with fp16, sdpa, prompt 512, 32 decode steps. The launch probe is run with **no profiler in the process** (`--probe-only`, 5 passes after 3 warm-ups), because the probe inside a profiled run is inflated (9.1).
Data: [results/measured/a40-round2/](results/measured/a40-round2/) (`probe-*`, `profile-*` operator tables and summaries, `nsys-*-summary.json`; raw traces stay on the pod).

| model | layers | enqueue ms (5 passes) | synced step ms | enqueue / step | verdict | GPU busy in the decode window: nsys / torch profiler | kernels per step |
|---|---|---|---|---|---|---|---|
| Qwen2.5-0.5B | 24 | 20.2-25.3 | 20.6-25.7 | 0.98 | cpu-bound | 13.2% / 10.8% | 1,055 |
| Qwen2.5-1.5B | 28 | 20.1-20.7 | 20.9-21.4 | 0.96 | cpu-bound | 25.3% / 21.4% | 1,283 |
| Qwen2.5-3B | 36 | 25.0-25.3 | 26.0-26.4 | 0.96 | cpu-bound | 33.0% / 29.4% | 1,643 |
| **Qwen2.5-7B** | 28 | 18.7-18.9 | **30.0** | **0.62-0.63** | **mixed** | **69.4% / 63.9%** | 1,171 |

- **Launch-bound up to 3B, not at 7B.** From 0.5B to 3B the synced step is 20-26 ms and equals the CPU enqueue time (within about 4%); at 7B the CPU still needs only 18.8 ms but the step takes 30 ms, so the device is now the long pole and the GPU is busy 64-69% of the window.
  Weight-read floors at the A40's 0.696 TB/s (fp16 weights / bandwidth) are 1.4 / 4.4 / 8.9 / 21.9 ms for the four models: below the ~20-26 ms enqueue for the first three, above it for 7B. That is the arithmetic behind the crossover; it was not isolated by a controlled experiment.
- **The CPU cost scales with layers, not parameters.** Enqueue time per layer is about 0.7 ms for 1.5B, 3B and 7B (20.4/28, 25.2/36, 18.9/28) and 0.8-1.0 ms for 0.5B, whose step time drifted between passes (25.1 -> 20.2 ms; the earlier 9.1 runs on the first pod saw 15.8-18.8 ms for the same model, so absolute step times depend on the host CPU).
  The 0.5B and 1.5B steps are nearly equal (about 21 ms) although the 1.5B has 3x the parameters, and 7B has the same layer count as 1.5B but a different regime.
- **nsys agrees with the torch profiler.** nsys 2026.5.2 (`--trace=cuda,nvtx --sample=none --cpuctxsw=none`, capture range = the profiled window) records the **identical 33,760 kernels** for the 0.5B decode window and a GPU busy fraction of 13.2% against 12.8% from the torch profiler in 9.1
  (10.8% in this round's torch-profiler run). The nsys numbers include nsys's own overhead, as the torch-profiler ones include Kineto's.
- **What this changes in 9.1.** The Day 1 findings "decode ms/token is flat across batch size and context" and "TTFT flat below 2048 tokens" were explained there by the launch floor, which this table shows holds up to 3B on this CPU. They should **not** be extrapolated to 7B: there the step has real device time, so batch scaling and context length
  would matter. E1-E3 were not re-run at 7B.
- Limits: one host CPU per model sweep, 5 passes each, single 512-token prompt, HF `generate`-style loop (not vLLM), fp16.

### 9.6 The 234 ms p50 ITL at `input_len=3584` is long-context decode cost inside mixed steps

Day 4 saw p50 ITL 234 ms at 3584-token prompts against 11 ms at 512, and the 9.3 step-cost rule predicted about 100 ms. Reproduced and traced here: fresh Day 4 baseline server (256 seqs, 8192-token budget, 0.5B), 1000 prompts, rate inf, `--profile` windows of 100 engine steps
after delays of 135 and 500 iterations, `--save-detailed` for per-token ITLs; per-step GPU time from `gpu/step_costs.py` (tests in `tests/test_step_costs.py`). Profiler is on, so throughputs below are not for comparison with Day 4.

| run | p50 ITL | pure-decode steps | mixed steps (prefill tokens, decodes) | rule of thumb (9 ms/1k prefill + 0.1 ms/decode) | fitted ms per decode in mixed steps |
|---|---|---|---|---|---|
| 512 in, window 135 | 11.2 ms | 86 x 7.3 ms (256 seqs) | 14 x 93.4 ms (8,052; 137) | 86 | 0.11 |
| 512 in, window 500 | 11.2 ms | 84 x 6.5 ms | 16 x 98.3 ms (7,364; 226) | 89 | not stable (16 steps) |
| 3584 in, window 135 (ramp-up) | 232.8 ms | none | 100 x 155 ms (8,076; 116) | 84 | 0.58 |
| **3584 in, window 500 (steady)** | **233.2 ms** | 11 x **24.8 ms** (256 seqs) | 89 x **231 ms** (7,641; 251) | **94** | **0.51** (intercept 23 ms, 10.5 ms/1k prefill tokens) |

- **The 234 ms reproduces** (232.8 and 233.2 ms, p90 241 ms) and equals the mean mixed-step time (231 ms): at 3584 tokens 89 of 100 steps carry about 7.6k prefill tokens, so nearly every decoded token is emitted from a step that takes about 231 ms.
- **The rule under-predicts by 2.5x here (231 vs 94 ms), but only because its decode coefficient was fitted at 512-token contexts.** The prefill coefficient barely moves (8.8 -> 10.5-11 ms per 1k tokens), while each decode riding in a mixed step costs **0.51-0.58 ms instead of 0.11 ms** (about 5x for 7x the context).
  Refit for 3584: `mixed ms = 23 + 10.5 x prefill/1000 + 0.51 x decodes`. A cross-window check: the steady-window coefficients applied to the ramp-up window's mean (8,076 prefill, 116 decodes) give 167 ms against 155 ms observed (+8%).
- **Pure decode shows the same effect directly.** A 256-sequence pure-decode step costs 24.8 ms at 3584 against 6.5-7.3 ms at 512, +18 ms; reading every sequence's KV once is 256 x ~3,650 tokens x 12 KiB (24 layers x 2 x 2 KV heads x 64 x 2 B) = 11.5 GB, which takes 16.5 ms at 0.696 TB/s. That fits.
- **What remains unexplained:** why a decode costs about 5x more inside a mixed step than in a pure step (0.51 vs 0.097 ms per sequence, the same 4-5x seen at 512 in 9.3). This round shows it grows with context length; it does not identify the kernel path responsible (no per-kernel attribution of the mixed steps was done).
- Limits: 100-step windows (n = 89-100 mixed steps), least-squares fits, one GPU, one model, profiler overhead present in absolute times.

### 9.7 E6 with a pool the workload can exhaust (`--kv-cache-memory-bytes`, one run per pool)

Day 4's utilization knob never pushed the pool below demand (9.4: E6 untested). Here the pool is set directly (`gpu/run_e6_cap.sh`): 0.5B, 1000 prompts of 512 in / 128 out at once, `max_num_seqs=256`. Computed peak demand is 256 x 640 = 164k tokens (about 2.0 GiB).
Peak KV, running and waiting come from a 5 Hz `/metrics` sampler; preemptions from the final counter.

| KV pool | pool (tokens) | preemptions | peak KV | peak running | out tok/s | mean TTFT s | p99 ITL ms |
|---|---|---|---|---|---|---|---|
| 4,096 MiB | 349,504 | 0 | 47% | 256 | 9,242 | 6.8 | 115 |
| 2,560 MiB | 218,432 | 0 | 75% | 256 | 9,557 | 6.6 | 117 |
| 2,048 MiB | 174,720 | 0 | 94% | 256 | 9,396 | 6.8 | 120 |
| **1,536 MiB** | **131,072** | **71** | 100% | 228 | 8,800 | 7.0 | 109 |
| 1,024 MiB | 87,360 | 75 | 100% | 152 | 8,359 | 7.5 | 101 |
| 512 MiB | 43,648 | 85 | 100% | 78 | 7,139 | 9.3 | 83 |

- **Verdict on E6's hypothesis (preemptions only below peak demand; throughput falls, TTFT rises): held.** Preemptions start between a 174,720-token pool (0; peak KV 94%, matching the computed 164k demand) and 131,072 tokens (71). Against the mean of the three non-preempting pools (9,398 tok/s) throughput is
  -6.4%, -11.1% and -24.0% in the three preempting pools, TTFT rises +4%, +12% and +38%, and p99 ITL falls (fewer sequences run at once: peak running 256 -> 78).
- **Against the Day 3 simulator (9.4, simulated E6):** direction held; the **magnitude was overstated** (simulated throughput -78% at the smallest pool, measured -24%), and the simulator's preemption counts (0 / 0 / 7 / 17 / 27 / 27) are 3-10x below the real counters (71-85), which count every preemption event, not distinct requests.
- n = 1 per pool; the three non-preempting pools differ by 3.4% among themselves (single runs), so the -6.4% step at 131k is only marginally outside that spread, while the last two steps are clearly outside it.

### 9.8 RUNBOOK section 5 sweeps (E4-E6 via Day 4's tooling), run again on a second pod

`python run_sweep.py --load saturated --repeat 1 --num-prompts 1000 --only max_num_batched_tokens max_num_seqs gpu_memory_utilization` on a different A40 host, 0.5B, 13 configs x 1 run, 0 failed requests, **0 preemptions in all 13 runs**
([data](../day4-vllm-internals/results/day6-sweep/), summaries only; raw JSON is git-ignored). This is a replication of Day 4's 1000-prompt saturated sweep (n = 3 there), not new information about the knobs:

| knob = value | out tok/s | p50 ITL ms | p99 ITL ms | p99 TTFT s | vs Day 4 (n=3) tok/s |
|---|---|---|---|---|---|
| `max_num_batched_tokens` = 512 | 5,067 | 19.2 | 25 | 24.3 | +0.1% |
| = 1024 | 5,352 | 36.7 | 46 | 22.7 | +0.4% |
| = 2048 | 6,711 | 18.3 | 62 | 17.7 | -2.4% |
| = 4096 | 8,493 | 11.3 | 79 | 13.7 | -0.1% |
| = 16384 | 9,963 | 11.0 | 180 | 11.4 | +4.0% |
| `max_num_seqs` = 4 | 1,118 | 3.3 | 7 | 113.1 | -0.5% |
| = 16 | 3,509 | 3.6 | 18 | 35.4 | -0.1% |
| = 64 | 6,992 | 5.4 | 78 | 17.6 | +1.0% |
| = 1024 | 8,149 | 43.2 | 422 | 11.2 | -3.0% |
| `gpu_memory_utilization` = 0.5 / 0.7 / 0.95 | 9,502 / 9,354 / 9,304 | 11.2-11.3 | 110-112 | 12.1-12.4 | -0.1% / -1.4% / -2.9% |
| baseline (256 seqs, 8192 tokens, 0.85) | 9,352 | 12.8 | 110 | 12.3 | -1.0% |

- **All 13 configs agree with Day 4 within -3.0% to +4.0%** (Day 4's own repeat noise: about 1% typical, worst case 4.9%), so the E4/E5 rows of 9.4 reproduce on another host: no throughput plateau at 64 seqs (3,509 / 6,992 / 9,352 for 16 / 64 / 256), a decline at 1,024 (-12.9% vs 256 here, -11% in Day 4),
  throughput 5,067 -> 9,963 (+97%) from the smallest to the largest budget (Day 4: +89%), and the p50 ITL hump at `max_num_batched_tokens` = 1024 (36.7 ms) reproduces, which 9.3 explains by the share of mixed steps.
- **E6 through `gpu_memory_utilization` is still flat for the reason 9.4 gave** (the pool is far above demand, 0 preemptions), which is why 9.7 shrinks the pool directly.
- Single repeat per config, so differences under about 3% here are not interpretable on their own.

### 9.9 What was not done

Nsight Compute kernel counters; a kernel-level attribution of *why* a decode costs about 5x more inside a mixed vLLM step (9.6 shows it grows with context length but does not identify the kernel path); an isolating experiment for the `max_num_seqs=1024` throughput loss (still an open hypothesis; it reproduced again at -12.9% in 9.8);
E1-E3 and the vLLM step-cost rule at models above 0.5B (9.5 covers the Day 1 loop up to 7B, not vLLM); one host CPU per model sweep and n = 1-6 repeats throughout.
The first trace attempt (delay 20) only captured the benchmark's warm-up request, and my first window (delay 150) held 20 steps; the 9.3 numbers come from the 100-step windows after a 135-iteration delay.

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
