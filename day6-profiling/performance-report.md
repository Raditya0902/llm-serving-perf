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
- **Decodes riding in a mixed step cost about 4x more each** (cause: the attention kernel, see 9.9). Same prefill size (about 7k tokens), 134 vs 516 decodes: 93 vs 132 ms, so about 0.10 ms per extra decode sequence, against 0.026-0.029 ms per sequence in pure decode steps (7.3 ms / 256, 26 ms / 999). One sampled step with only about 1.5k prefill tokens but 995 decodes also took 133 ms (n=1). An earlier narrower window saw the attention kernel take 1.23 ms per layer call in mixed steps vs 0.16 ms in pure decode (2 steps), which fits. Rule of thumb that fits five vLLM configs: mixed step ms about 9 x (prefill tokens / 1000) + 0.1 x (running decodes) (e.g. bt=512 with 115 running: 16 ms predicted vs 19 ms median ITL; bt=1024 with 227 running: 32 vs 37; bt=2048 with 256 running: 44 vs 50). It is a rough fit, not a validated model.
- **This is the mechanism behind Day 4's ITL tails and the p50 ITL oddity.** Inter-token latency follows the step time, and a step that includes prefill is 5-18x a pure-decode step. The non-monotonic p50 ITL (bt=512 19 ms, bt=1024 37 ms, bt=2048 20 ms, bt=8192 11 ms) is explained by the *share of tokens emitted in mixed steps*: with budgets 512 and 1024 almost every step is mixed (88% and 90% of ITL samples exceed 15 ms), whereas with 2048 only 54% do, so the median falls back toward the pure-decode regime even though its mixed steps are slower than at 1024.
- **Measured (not computed) KV and concurrency from a 5 Hz `/metrics` sampler** (1000-prompt burst, [analysis_output.txt](results/measured/cuda-vllm/rerun/analysis_output.txt)): `max_num_seqs=256`: peak running 256, peak waiting about 485, **peak KV 5.3%**; `max_num_seqs=1024`: **peak running 996-999**, peak waiting about 550, **peak KV 21.5%**; budgets 512 / 1024 / 2048: peak running 115 / 227 / 256, peak KV 2.2% / 4.4% / 5.1%. 0 preemptions in all. These match Day 4's computed demand (5.3% and 21.8%).
- **Sampling is a large share of a pure-decode step** for this model: top-k/softmax/temperature/top-p kernels sum to roughly 3.3 of the 7.3 ms (about 45%, window 150), because the vocabulary is 151k and the model is tiny. Larger models would dilute this.
- **Hypothesis tested and refuted: the `max_num_seqs=1024` slowdown is not a CUDA-graph capture-range effect.** vLLM caps the captured size at 512 by default (51 full-graph sizes at 1024 seqs). Raising it with `-cc.max_cudagraph_capture_size=1024` (83 graphs), interleaved A/B, 3 fresh servers each, 1000 prompts ([data](results/measured/cuda-vllm/cg_test/)):

| Config | Output tok/s (3 runs) | p50 ITL | p99 ITL |
|---|---|---|---|
| `max_num_seqs=256` | 9,329 / 9,453 / 9,065 (mean 9,282) | 11 ms | 110-111 ms |
| `max_num_seqs=1024` | 8,512 / 8,432 / 8,515 (mean 8,487, **-8.6%**) | 39-42 ms | 432-481 ms |
| `max_num_seqs=1024` + capture size 1024 | 8,499 / 8,548 / 8,567 (mean 8,538) | 40 ms | 426-450 ms |

  The slowdown reproduces (-8.6% here, -11% in Day 4) and the graph setting does not move it. The slowdown is therefore not a graph effect. All 1000 requests are concurrent at 1024 (peak running 999) and the decodes spend more of the run riding in expensive mixed steps, which is consistent with the lower throughput, and round 3 isolated it (9.10): it is the FlashAttention 2 mixed-step penalty of 9.9, because the loss disappears with `--attention-backend TRITON_ATTN` (-9.4% to +0.4%). A second interleaved pair with live metrics reproduced it: 256 seqs 9,445 / 9,457 tok/s vs 1024 seqs 8,218 / 8,581 (about -11%), p99 ITL 112-119 vs 415-423 ms.

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
- **Why a decode costs about 5x more inside a mixed step (0.51 vs 0.097 ms per sequence, the same 4-5x seen at 512 in 9.3) is explained in 9.9**: the attention kernel (FlashAttention 2) loses its grouped-query decode path as soon as one sequence in the batch has more than one query token. **Correction:** the 0.51 ms/decode coefficient above is not identified (the decode count is 249-255 in every window step); only the constant 151 ms is, see 9.9.
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

### 9.9 Why a decode costs 4-7x more inside a mixed step: FlashAttention 2 stops using its GQA decode path (round 3, third A40 pod)

9.6 left one thing unexplained: a decode riding in a mixed step costs about 5x what it costs in a pure-decode step. Tools added for this: `gpu/kernel_costs.py` (assigns every profiler kernel to the step annotation that contains its start and totals per kernel family; per-step coverage is checked), `gpu/attn_microbench.py` (vLLM's own `flash_attn_varlen_func` on synthetic paged batches) and `gpu/run_backend_ab.sh`.
Same server as 9.6 (0.5B, 256 seqs, 8192-token budget, 1000 prompts at 3584 in / 128 out, steady 100-step window after 500 iterations). Data: [results/measured/a40-round3/](results/measured/a40-round3/) (`step_costs/`, `kernel_costs/`, `microbench/`, `backend-ab/`); hypotheses were written down first in [PREREGISTRATION_ROUND3.md](PREREGISTRATION_ROUND3.md).

**1. The extra cost is in the attention kernel.** Kernel coverage (kernel ms / step ms) is 99.9% on average, 98.4% at worst.

| step type (3584 context) | steps | mean step ms | attention ms | GEMM ms | norm/activation ms |
|---|---|---|---|---|---|
| pure decode, 256 seqs | 11 | 24.9 | 21.6 | 2.4 | 0.7 |
| mixed, about 7,640 prefill tokens + 251 decodes | 89 | 232.6 | **162.2** | 53.3 | 16.3 |
| mixed, `--attention-backend TRITON_ATTN` | 89 | 128.6 | **55.1** | 55.7 | 16.3 |

**A correction to 9.6.** The decode count in these 89 mixed steps is 249-255 (and 118-120 in the 8B windows of 9.11), so the least-squares "ms per decode" is not identified: it can only split a constant between the intercept and the decode term. The 0.51 ms/decode of 9.6 is really "23 + 0.51 x 251 = about 151 ms of cost that does not vary with the prefill size".
The well-determined fit is on total tokens: step = 151.5 + 10.3 ms per 1k tokens (residual 0.9 ms); attention = 147.7 + 1.84 ms per 1k tokens; GEMM = 3.5 + 6.3 ms per 1k tokens. So a mixed step spends **about 148 ms on attention however much prefill rides along, against 21.6 ms for the same 256 sequences in a pure step: 6.8x**. `gpu/step_costs.py` and `gpu/kernel_costs.py` now warn when the decode count barely varies. The one 1024-sequence window of 9.10 whose decode count varies (mean 711 decodes, 39 mixed steps) gives an identified 0.116 ms per decode, close to the 0.094 ms of the microbenchmark below.

**2. A microbenchmark of the kernel alone reproduces the gap and tracks the GQA group size.** One layer, 256 decodes with 3584 cached tokens each, plus one 7,600-token prefill chunk in the mixed batch; median of 50 calls, CUDA events; times in ms (FlashAttention 2, `fa_version=2`, as on an A40):

| geometry | GQA group (q heads / kv heads) | pure decode | mixed | prefill alone | mixed over pure, per decode | pre-registered |
|---|---|---|---|---|---|---|
| Qwen2.5-0.5B | 7 (14 / 2) | 0.888 | 7.337 | 1.500 | **6.58** | about 7 |
| same, with 14 kv heads (no GQA control) | 1 (14 / 14) | 6.070 | 7.681 | 1.601 | **1.00** | about 1 |
| Llama-3.1-8B | 4 (32 / 8) | 6.526 | 29.567 | 5.738 | **3.65** | about 4 |

All three predictions held. Cross-checks: the microbenchmark's 0.0228 ms per decode and layer x 24 layers = 0.55 ms per decode against 148 ms / 250 = 0.59 ms measured in the server, and 0.0035 x 24 = 0.084 ms against 21.6 / 256 = 0.084 ms for pure decode. The mixed-batch decode time of the GQA-7 model (7.337 - 1.500 = 5.8 ms) is about the pure-decode time of the no-GQA control (6.07 ms), which reads 7x the KV bytes.

**3. The penalty switches on at a query length of 2, not with the size of the prefill chunk.** 0.5B geometry, 256 decodes, 3584 context, mixed batch with one extra sequence of P query tokens:

| P (longest query in the batch) | 1 | 2 | 16 | 128 | 1,024 | 7,600 |
|---|---|---|---|---|---|---|
| mixed batch ms | 0.888 | 6.091 | 6.083 | 6.087 | 6.102 | 7.336 |

With P = 1 the batch is still "all decodes" and costs the same as a pure batch (0.887 ms); the moment one sequence has two query tokens the cost jumps 6.9x and then stays flat from P = 2 to 1,024. Across 32 / 64 / 128 / 256 / 512 decodes the ratio is 5.5 / 6.2 / 6.7 / 6.6 / 6.9. At 512 cached tokens the same step appears (6.5-6.8x for P = 2-1,024; 0.0039 vs 0.0006 ms per decode and layer = **0.094 vs 0.014 ms per decode over 24 layers, which is the "0.1 ms per decode" rule of thumb of 9.3**). With P = 7,600 at 512 context the ratio is 18, an extra effect that does not occur in the server (its longest query is bounded by the 512-token prompt); round 4 investigated it (9.12, Q4b: block-launch cost that grows with decodes x P and is hidden at long context).

**4. Changing the backend removes most of it, end to end.** Unprofiled A/B, fresh server per run, 1000 prompts, 0 failed requests ([backend-ab/](results/measured/a40-round3/backend-ab/)):

| workload (in x out), 256 seqs | FLASH_ATTN (default) tok/s | TRITON_ATTN tok/s | change | p50 ITL ms (FA2 / Triton) |
|---|---|---|---|---|
| 3584 x 128 (n = 2 each) | 1,226 (1,228; 1,225) | 1,965 (1,967; 1,962) | **+60%** | 236 / 134 |
| 512 x 128 (n = 3) | 9,334 | 10,655 | **+14%** | 11.3 / 10.5 |
| 128 x 128 (n = 3) | 17,052 | 17,291 | +1.4% (within noise, 15,994-18,081) | |

**What is measured and what is inferred.** Measured: where the time goes (attention), that a kernel-only benchmark reproduces it, the step at query length 2, the dependence on the GQA group size (1 / 4 / 7 gives 1.0 / 3.65 / 6.6), and the backend switch. Inferred, not observed: that FlashAttention 2 only uses its grouped-query path (one KV read per KV head) when the longest query in the batch is a single token, and otherwise streams the KV once per query head. It fits every number above (mixed GQA-7 time about equals 7x the bytes at the same bandwidth), but the DRAM byte counters that would confirm it could not be read: Nsight Compute 2026.4.0 installed fine and failed with `ERR_NVGPUCTRPERM` (the container may not read GPU performance counters; the setting is on the host), tested on a small matmul ([ncu_gate_a.txt](results/measured/a40-round3/ncu_gate_a.txt)).
Scope: A40 (sm86, FlashAttention 2), vLLM 0.34.0, Qwen2.5-0.5B end to end and the 8B and MHA geometries in the microbenchmark only. Hopper (FlashAttention 3: no penalty, 9.12), other models (1.5B / 3B / 8B, 9.12) and the output quality of the two backends (equivalent, 9.12) were tested in round 4; the 18 at 512 context is explained in 9.12 (Q4b). Profiled runs (the first table) are not comparable with unprofiled throughput; the A/B in 4 is unprofiled.

### 9.10 Isolating the `max_num_seqs=1024` loss

Fixed beforehand ([PREREGISTRATION_ROUND3.md](PREREGISTRATION_ROUND3.md)): H1 = decodes riding in expensive mixed steps; H2 = per-step CPU / scheduler / output-processing overhead that grows with running sequences; H3 (KV memory) already excluded (peak KV 21%, 0 preemptions).
`gpu/run_seqs_isolation.sh`: 0.5B, 1000 prompts at once, 8192-token budget, fresh server per run, repeat-major order, 3 repeats, 27 runs, 0 failed requests ([seqs-isolation/](results/measured/a40-round3/seqs-isolation/)). tok/s, mean [min-max]; this pod's baseline is about 3% below the earlier hosts (9,134 vs 9,282-9,457), so only ratios within this table are compared:

| workload (in x out) | 256 seqs | 1024 seqs | 1024 vs 256 | p50 ITL (256 / 1024) ms |
|---|---|---|---|---|
| 512 x 128 (baseline) | 9,134 [8,984-9,253] | 8,321 [8,195-8,459] | **-8.9%** | 11.1 / 41.0 |
| 128 x 128 | 15,885 [14,748-16,827] | 14,777 [14,628-14,993] | -7.0% | 10.6 / 37.9 |
| 512 x 512 | 15,201 [14,903-15,400] | 14,899 [13,377-15,670] | -2.0% | 12.1 / 45.4 |
| 128 x 512 | 20,920 [20,739-21,234] | 21,721 [20,711-23,148] | +3.8% (noise) | 9.9 / 37.9 |
| 512 x 128 at 512 seqs | | 9,021 [8,834-9,156] | -1.2% (dose-response) | 20.4 |

**The loss shrinks to zero when decode dominates the run** (512 output tokens), which is the H1 shape; at 128 x 128 it stays at -7.0%, which that simple reading does not predict (see the estimate below).

**The decisive test is an intervention on the mechanism of 9.9**: the same comparison with `TRITON_ATTN`, unprofiled, 3 repeats per cell, 0 failures ([backend-ab/](results/measured/a40-round3/backend-ab/)):

| workload | backend | 256 seqs tok/s | 1024 seqs tok/s | 1024 vs 256 |
|---|---|---|---|---|
| 512 x 128 | FLASH_ATTN | 9,334 [9,019-9,558] | 8,458 [8,450-8,472] | **-9.4%** |
| 512 x 128 | TRITON_ATTN | 10,655 [10,521-10,757] | 10,703 [10,613-10,782] | **+0.4%** |
| 128 x 128 | FLASH_ATTN | 17,052 | 16,528 | -3.1% |
| 128 x 128 | TRITON_ATTN | 17,291 | 17,172 | -0.7% (noise: runs span 15,994-18,574) |

**Verdict: H1 holds in the form "the FlashAttention 2 mixed-step penalty of 9.9 times the number of decodes that ride in mixed steps".** With a backend that has no such penalty the 1024-sequence arm is no slower than the 256 one, with tight repeats (8,450-8,472 against 10,613-10,782). Why 1024 pays more: its 1000 requests are admitted together, so about 63 prefill-carrying steps each carry hundreds of decodes (profile windows: 494 and 711), whereas with 256 sequences prefill arrives in waves of 16 steps with fewer decodes (139-227 in the windows).
A rough estimate (inferred, with guessed average decode counts of 500 and 128): about 31k decode-steps in mixed batches at 1024 against about 8k at 256, at 0.094 ms each = 2.9 s against 0.8 s, i.e. about 2 s extra against the 1.4 s actually lost (15.4 vs 14.0 s of run time). The right order of magnitude, not a fit. A similar rough estimate for 128 x 128 gives about -5%, against -7.0% in the grid and -3.1% in the later A/B (both FlashAttention 2): noisy, not contradictory.

**What the pre-registered rule says, and why I do not follow it.** Applied literally, `gpu/seqs_isolation.verdict` returns **H2**, because its idle-time criterion fires: GPU busy fraction (step GPU time / wall span, `step_costs.busy_fraction`) is 0.81-0.83 in two 256-seq windows (median inter-step gap about 4.4 ms) against 0.60-0.77 in the two steady 1024-seq windows (median gaps 20-22 ms), a difference of 14 points against the rule's 10.
I do not accept that as the cause: (a) those windows are profiled, and kernel coverage in the two steady ones is only 72-89% (against 97-100% for the 256-seq and 8B windows), i.e. the profiler's own overhead and unattributed time inflate exactly the arm with the most kernels; (b) if per-step overhead per running sequence were the cause, the loss would be largest in decode-dominated runs, but there it is absent (+3.8% at 128 x 512, -2.0% at 512 x 512); (c) swapping the attention backend removes the whole loss. The rule's idle-time criterion should have been measured without a profiler; that was a flaw in my pre-registration, so both the literal output and my reading are recorded here. Per-token pure-decode cost at 1024 is about equal to 256's in the unprofiled ITLs (p50 ITL / running sequences: 0.038-0.045 vs 0.041-0.047 ms).
**Correction from round 4 (9.12):** the backend swap removes the loss at 512x128 on the A40 for 0.5B, 1.5B and 3B (Triton +0.4%, +2.9%, +2.8%), but not at 128x128 with 6 repeats (Triton -5.7%, t = -2.7) nor on an H100 at 512x128 (Triton -5.3%); a part of the loss is therefore not the attention penalty.
Limits: 0.5B, one A40, one pod; the 1,024-seq windows are 100 steps each; n = 3 per cell and some cells have a 10% spread. The Day 4 1.5B run (-4.9%) was not re-examined.

### 9.11 Llama-3.1-8B under vLLM: step costs and the 0.5B rule

`gpu/run_itl_study.sh` with `MODEL=NousResearch/Meta-Llama-3.1-8B-Instruct` (fp16, 128 KiB of KV per token; pool 174k-178k tokens, so demand stays at 47-79% of it), `--max-num-batched-tokens 2048` (a step at 8192 would take about a second), 1000 prompts, 100-step windows after 150 and 400 iterations, profiler on, 0 failed requests. The two windows of each arm agree.

| arm | busy fraction | pure-decode step | mixed steps (prefill tokens, decodes) | attention / GEMM in mixed steps | total-token fit (mixed) |
|---|---|---|---|---|---|
| 512 in, 128 seqs | 0.98 | 48.1-48.4 ms (128 decodes: GEMM 29, attention 17) | 34-35 x 343-344 ms (1,860-1,910; 118-120) | 66-67 / 254-257 ms | 70-98 ms + 122-138 ms per 1k tokens |
| 2048 in, 64 seqs | 0.99 | 62.2 ms (64 decodes: GEMM 29, attention 32) | 65 x 407-410 ms (1,985; 61) | 126-127 / 259-261 ms | not stable (windows differ: context grows) |

Pre-registered predictions: **prefill coefficient 100-180 ms per 1k tokens: held** (122-138 vs the 9 ms of the 0.5B). **Pure decode about 23 ms plus KV reads: held in structure, 48 ms measured**: the GEMMs take 29 ms (floor 21.9-23 ms for 16 GB at 0.696 TB/s) and attention 17 ms (128 sequences x about 576 tokens x 128 KiB = 9.7 GB, 14 ms at peak bandwidth). **GPU busy fraction close to 1: held** (0.98-0.99, against 0.13 for the launch-bound 0.5B Day 1 loop in 9.5).
**Mixed-decode cost about 4x its pure value: not testable from the traces** (decodes constant, 118-120 and 61, so not identified); the microbenchmark gives 3.65x at 3584 and 6.3x at 512 cached tokens for the 8B geometry. In an 8B mixed step attention is only 19% of the time at 512 context (67 of 343 ms) against 70% for the 0.5B at 3584, which fits Day 4's observation that the `max_num_seqs=1024` loss shrinks with model size (-4.9% at 1.5B); the 8B cannot run 1,024 sequences (KV pool 173-176k tokens against 655k needed); 9.12 compares 64 and 192 seqs instead.
The 0.5B rule (9 ms per 1k prefill tokens + 0.1 ms per decode) is a 0.5B/A40 fit and does not transfer to 8B (about 15x the prefill coefficient).

### 9.12 Round 4: output check, other models, Hopper, the ratio of 18, loose ends (A40 pod 4, H100 PCIe pod)

Predictions and decision rules were written first ([PREREGISTRATION_ROUND4.md](PREREGISTRATION_ROUND4.md)); refuted ones come first below. Data: [results/measured/a40-round4/](results/measured/a40-round4/) and [results/measured/h100-round4/](results/measured/h100-round4/). Same stack as before: vLLM 0.34.0, torch 2.14.0+cu132, an A40 (sm86) and an H100 PCIe (sm90). Every bench run below has 0 failed requests; runs with failures are rejected by `gpu/backend_ab_summary.py`.

**Refuted or partly refuted**

| prediction | result |
|---|---|
| E2 (TTFT 2048 / 512 in [3.8, 5.0]) refuted at 1.5B, borderline at 3B, confirmed at 8B | **refuted at every size**: 1.99 / 1.76 / 2.02 / 2.22 (0.5B / 1.5B / 3B / 8B). TTFT(128 / 512 / 1024 / 2048) is 26 / 32 / 34 / 64 ms (0.5B), 38 / 48 / 56 / 85 (1.5B), 49 / 55 / 68 / 112 (3B), 55 / 89 / 107 / 197 (8B). A fixed per-request cost of about 25-50 ms (HTTP, tokenization, the weight read of a step) keeps TTFT(512) well above the prefill-only share; 512 to 2048 tokens adds about 70 ms per 1k tokens at 8B, against the 122-138 ms of the mixed steps of 9.11 (inferred difference: a mixed step also carries the decodes) |
| Q4: excess cost is the same at ctx 512 and 3584, linear in decodes x P (R^2 >= 0.9) | **refuted as written** (R^2 0.27; excess at ctx 3584 is about zero). See Q4 below for what the data do show |
| 128x128 spread at most 4% with 4,000-prompt runs | **refuted**: 4.6-5.2% at 256 seqs, 10-12% at 1024 seqs (n = 6 per cell) |
| Triton stays within +-1.5% when going to 1024 seqs | **partly refuted**: +2.9% (1.5B) and +2.8% (3B) at 512x128, i.e. Triton benefits from the larger batch while FA2 loses; and at 128x128 on the A40 Triton lost 5.7% (see below) |
| 1.5B Triton gain at 512x128 of +8 to +13% | +6.0%, a little below the range |
| 8B at 1024 seqs | **not testable**: 1,024 sequences of 640 tokens need 655k KV tokens, the pool is 173-176k. I ran 64 and 192 seqs, where FA2 gains +17.8% and Triton +30.0% from the larger batch |
| FA2 on the H100 at least 10% slower than Triton at 512x128 and at least 40% at 3584x128 | **refuted**: +1.6% (not slower) and -16.6% (4,277 / 3,568 = Triton 20% faster). The 0.5B on this H100 PCIe is probably limited by something other than attention (13.6k tok/s against 10.7k on the A40, for a GPU with about 3x the bandwidth; inferred, I did not measure the busy fraction) |

**Held**

- **Q1 Triton is numerically equivalent to FlashAttention 2 (0.5B, A40).** 200 prompts of 64-3,000 tokens cut from this folder's markdown, greedy, 64 new tokens, 2,048-token budget so decodes and prefill chunks share steps; control = FA2 with an 8,192-token budget (same kernel, different chunking = the noise floor); reference = HF float32 prompt log-probs on 20 prompts of at most 1,024 tokens ([quality/](results/measured/a40-round4/quality/)).

| | exact-match of completions | mean abs delta chosen-token log-prob | mean abs delta prompt log-prob | divergences that are near-ties (gap < 0.5 nats) |
|---|---|---|---|---|
| Triton vs FA2 | 58% | 0.0132 | 0.0286 | 84 of 84 |
| FA2 vs FA2, other chunking (control) | 54% | 0.0131 | 0.0292 | 92 of 92 |

  Mean prompt-log-prob error against the fp32 reference: FA2 0.0269, Triton 0.0273 (1.4% higher). All five pre-registered criteria hold. The point worth knowing: even the same kernel agrees on only 54% of 64-token greedy completions (fp16 drift flips near-ties, first divergence at token 16-19 on average), so exact-match is not a usable quality bar; the log-prob differences are. Scope: Qwen2.5-0.5B only, 200 prompts, completions of 64 tokens; I did not check Triton against FA2 on the larger models or on the H100.
- **Q2 the Triton gain and the FA2 loss hold at other GQA ratios** (unprofiled, 2 repeats per cell, spread 0.0-0.4%, KV pools 0.83-1.25M tokens for 1.5B / 3B and 173-176k for 8B, demand at most 79% of the pool). tok/s:

| model (GQA group) | seqs | workload | FA2 | Triton | Triton over FA2 | pre-registered |
|---|---|---|---|---|---|---|
| 1.5B (6) | 256 | 512x128 | 4,409 | 4,675 | +6.0% | +8 to +13% (slightly low) |
| 1.5B (6) | 1024 | 512x128 | 4,168 | 4,812 | +15.5% | |
| 1.5B (6) | 256 | 3584x128 | 562 | 745 | +32.5% | +30 to +55% (held, low end) |
| 3B (8) | 256 | 512x128 | 2,485 | 2,662 | +7.1% | +4 to +10% (held) |
| 3B (8) | 1024 | 512x128 | 2,356 | 2,736 | +16.1% | |
| 3B (8) | 128 | 3584x128 | 367 | 425 | +15.6% | +15 to +40% (held, low end) |
| 8B (4) | 64 | 512x128 | 816 | 857 | +4.9% | +1 to +5% (held, top) |
| 8B (4) | 192 | 512x128 | 962 | 1,114 | +15.8% | |

  Going from 256 to 1024 seqs: FA2 loses 5.5% (1.5B) and 5.2% (3B), inside the predicted -4 to -8% and -2 to -6%; Triton gains 2.9% and 2.8%. For the 8B the gain of Triton over FA2 grows with the batch (+4.9% at 64 seqs, +15.8% at 192), the same direction as the 0.5B: the more decodes ride in mixed steps, the more the FA2 penalty costs. p50 ITL at 3584x128 on 1.5B: 513 ms (FA2) against 365 ms (Triton).
- **Q3 Hopper: FlashAttention 3 does not have the penalty** (H100 PCIe, [h100-round4/](results/measured/h100-round4/)). Microbenchmark, 0.5B geometry, 256 decodes, 3584 cached tokens, one 7,600-token chunk, mixed over pure per decode: **FA2 on the H100 6.78** (A40: 6.58; pre-registered 5-8), **FA3 1.28** (at most 2). For the 8B geometry: FA2 4.16, FA3 1.00. At ctx 512 FA3 is below 1 (0.18-0.71): the mixed batch is cheaper per decode than the pure one, i.e. its kernel overlaps the decodes with the prefill. So the behaviour belongs to the FA2 kernel version, not to the A40.

  End to end on the H100 (0.5B, 2 repeats, tok/s; "vs Triton" = gain over TRITON_ATTN):

| seqs | workload | FA3 (default) | FA2 | Triton | FA3 vs Triton | FA2 vs Triton |
|---|---|---|---|---|---|---|
| 256 | 3584x128 | 6,523 | 3,568 | 4,277 | **+52.5%** | -16.6% |
| 256 | 512x128 | 13,249 | 13,859 | 13,637 | -2.8% (spread 7.3%) | +1.6% |
| 1024 | 512x128 | 13,029 | 12,973 | 12,916 | +0.9% | +0.4% |

  On Hopper the default (FA3) is the right choice and Triton would cost 34% at 3584x128 (not pre-registered: 4,277 vs 6,523). Within the runs of the table FA3 at 512x128 is within 5% of Triton, as predicted; FA3's 1024-seq change is -1.7% against FA2 -6.4% and Triton -5.3% (the FA2 and FA3 cells have 4-7% spread, so only Triton's drop, with 0.8% and 1.8% spread, is resolved).
- **Q4b refined model of the ratio of 18 (post hoc, then tested).** The first sweep ([sweep/fa2_0.5b.json](results/measured/a40-round4/sweep/fa2_0.5b.json), FA2, 0.5B geometry) shows what happens to d(P) = mixed(P) - prefill_only(P) as the prefill chunk P grows. At ctx 512 d rises linearly with decodes x P (256 decodes: 0.99 ms for P <= 512, then 1.14 / 1.41 / 2.00 / 2.89 ms at P = 1,024 / 2,048 / 4,096 / 7,600; 512 decodes: 2.31 to 5.78 ms; about 1.0e-6 ms per decode-token). At ctx 3584 d is flat or slightly falling (6.01 to 5.87 ms at 256 decodes). The ratio of 18 of 9.9 is this rise divided by a very small pure-decode time (0.15 ms). Model written down before the follow-up: d(P) = max(d_live, n (a + k P)), the cost of the blocks that FA2 launches for every decode sequence when the longest query is long, which exit at once, and which only show when the live decode work does not keep the SMs busy. Fitted on ctx 512 only (a = 3.4 us per decode, k = 1.03e-6 ms per decode-token), it predicts the unseen contexts 1024 and 2048 at 128 and 384 decodes with **median error 0.7% and maximum 19.5%** (limits 15% and 30%), and the excess falls with context at every decode count (n = 256, P = 7,600: 1.71 / 1.40 / 0.67 ms at ctx 512 / 1024 / 2048). On the wider set that also includes 64, 256 and 512 decodes the median is 1.4% and the maximum 30.8%, 0.8 points over the limit I had set for the registered set. Supported, with that qualification; it is a mechanism consistent with the data, not an observed one (no counters). Practical weight is small: the server's longest query is the 512-token prompt, where the model gives 1-4% of the attention time.

**Q5 loose ends**

- **128x128 with 4,000 prompts per run, 6 repeats (A40, 0.5B, [noise/](results/measured/a40-round4/noise/)).** The first attempt failed 75% of its requests because the client ran out of file descriptors (limit 1,024; fixed with `ulimit -n 65536`, run repeated, nothing from the failed run is used). tok/s: 256 seqs FA2 16,474 [15,940-16,764], Triton 17,010 [16,624-17,396] (+3.3%, Welch t = 3.0); 1024 seqs FA2 16,207 [15,528-17,093], Triton 16,033 [15,235-17,117] (-1.1%, t = -0.4). The 1024 arm against its own 256 arm: FA2 -1.6% (t = -0.95), **Triton -5.7% (t = -2.7)**. So at 128x128 a 1024-seq loss survives the backend switch (and on the H100 at 512x128 too, Triton -5.3%).
- **E1-E3 under vLLM** ([e123/](results/measured/a40-round4/e123/); concurrency 1-8 for E1, TTFT at 128-2048 input tokens for E2, TPOT at 32-256 output tokens for E3, `vllm bench serve`, one default server per model, A40):

| model | E1 tok/s at 8 vs 1 (>= 5x) | E1 TPOT at 8 vs 1 (<= 1.5x) | E2 TTFT 2048/512 ([3.8, 5.0]) | E3 TPOT 256 vs 32 (within 5%) |
|---|---|---|---|---|
| 0.5B | 6.33x | 1.17x | 1.99 (refuted) | +1.9% |
| 1.5B | 7.42x | 1.04x | 1.76 (refuted) | +0.9% |
| 3B | 7.32x | 1.06x | 2.02 (refuted) | +0.7% |
| 8B | 7.22x | 1.08x | 2.22 (refuted) | +0.1% |

  E1 and E3 confirmed at every size (TPOT at 1 sequence: 2.8 / 7.0 / 12.6 / 28.1 ms), E2 refuted at every size as on the Day 1 loop (9.2) and for the same reason: a per-request floor hides prefill growth. Concurrency 1 only, 12-24 requests per point.

**What this changes in 9.9-9.11.** (1) The advice "use `TRITON_ATTN`" is now backed by an output check (0.5B), by 1.5B / 3B / 8B throughput and by two GPUs, but only for FA2: on an H100 the default is FA3 and it is the fastest of the three. (2) The statement of 9.10 that swapping the backend removes the whole 1024-seq loss holds at 512x128 on the A40 (0.5B, 1.5B, 3B) and does not hold at 128x128 (Triton -5.7%, n = 6) or on the H100 (Triton -5.3%); the likely extra part is per-step host cost that grows with running sequences (H2), more visible when the GPU is not the limit. That is an inference from throughput, not a measurement (no busy fraction was taken on these runs). (3) The ratio of 18 is explained to the degree described under Q4b.

### 9.13 What was not done (as of round 4; most items were closed in round 5, section 9.14)

- ~~Nsight Compute counters~~ (closed in 9.14: blocked on RunPod, `ERR_NVGPUCTRPERM`; read on Lambda Cloud VMs with `sudo`).
- ~~A100 items: 8B data-parallel on NVLink and a same-GPU PCIe/NVLink pair~~ (closed in 9.14).
- ~~Triton output check beyond the 0.5B on an A40~~ (closed for 1.5B / 3B / 8B on an RTX A6000 and for 0.5B-8B on an H100 in 9.14; longer completions and non-greedy sampling are still not compared).
- **The 8B at 1,024 sequences**: impossible on this KV pool; would need fp8 KV or a larger GPU. **The 3584-token workload on the 8B** (about 7 minutes a run) and models above 8B.
- ~~A busy-fraction (unprofiled) measurement at 1,024 seqs~~ (done in 9.14 with Nsight Systems; the result is that H2 is not supported at 512x128 on the A6000, but the measure is noisy); E1-E3 at concurrency above 8 or with more than 12-24 requests per point are still open.
- Hopper: now one H100 SXM5 with all four model sizes (9.14); still no profiled end-to-end windows and no non-Qwen model other than Llama-3.1-8B.

### 9.14 Round 5 (Lambda Cloud: RTX A6000, H100 SXM5, 2x A100 PCIe, 8x A100 SXM4): counters, busy fraction, output check at 1.5B-8B, interconnect

Predictions and decision rules were written first ([PREREGISTRATION_ROUND5.md](PREREGISTRATION_ROUND5.md), which ends with the outcome of each, errors included). Data: [a6000-round5](results/measured/a6000-round5/), [h100-round5](results/measured/h100-round5/), and for the A100 pair `../day7-capstone/results/measured/round5-a100-pcie/` and `round5-a100-sxm4/`. Nsight Compute counters are readable on a Lambda VM with `sudo -E`; FA3 kernels need a demangled-name filter.

**Refuted or partly refuted**

| prediction | result |
|---|---|
| B1: FA2 busy fraction falls >= 10 points from 256 to 1,024 seqs (512x128) | **refuted**: the clean re-run gives 0.72 -> 0.84. H2 (host idle growing with running sequences) is not supported at 512x128 on the A6000; the loss is in device time. The Nsight Systems busy fraction is an interval union, noisy (+-0.07, up to 0.2) and not comparable to round 3's step-based number |
| B2: Triton busy fraction within 5 points at 512x128 | **refuted**: 0.68 -> 0.79 |
| B3 (Triton 128x128 loss is host-side) | **not testable**: the A6000's untraced A/B did not reproduce the A40's Triton loss at 1,024 seqs |
| D2: Triton output-equivalent to FA3 at 0.5B-8B | **refuted as written**: passes at 8B, fails the five criteria at 0.5B, 1.5B, 3B. The cause is the yardstick: on the H100 the same-kernel control is nearly exact (73-90% exact match, prompt-logprob difference 0.0001-0.0005), so "within 2x of the control" is unsatisfiable by another kernel; fp32 reference errors are equal at every size and every divergence is a near-tie |
| D3: Triton busy fraction falls >= 5 points where tok/s falls; FA2/FA3 within 5 | **refuted** (moves of 20-50 points on a mostly idle H100 at 0.5B; windows too noisy to separate H2 from the attention penalty) |
| E1: replicas beat TP=2 at c64 on NVLink by 10-40% | **refuted**: -4.1% (1,635 vs 1,705 tok/s); at c16 TP=2 wins by 26% (rule -5..-20%, measured -20.9%). The A40 pattern (replicas win at c64) is not reproduced on NVLink |
| F3: NCCL over host memory lands within 10% of the PCIe host | **refuted**: 27-28% slower (46.0 vs 36.3 ms c16; 158.6 vs 124.0 ms c64); the first clause (>= 5% slower than NVLink) held at 3.1x |
| F4: 8 KiB all-reduce latency <= 30 us on NVLink | **refuted**: 47.4 us (PCIe 45.1 us); the probe sees launch overhead at that size. Bandwidth held |

**Held**

- **The mixed-step penalty is observed at kernel level (A1-A3, D1).** FA2's mixed launch reads the KV about once per query head: extra DRAM bytes / pure-decode bytes = 7.04 (GQA 7) and 4.11 (GQA 4) at ctx 3,584 (7.40 and 4.14 at ctx 512), with 836x / 478x the thread blocks of the pure-decode launch, and the achieved DRAM share is 72% at ctx 3,584 against 20% at ctx 512 (so it is hidden at long context). FA3 on the H100 uses 132 blocks (the SM count) in every batch, bytes ratio 1.00, and its mixed/pure time is 1.7-3.1x at ctx 3,584 against FA2's 5.3-11.0x on the same GPU. 9.9's inference and 9.12's Q4b are now measurements.
- **Triton is output-equivalent to FA2 at 1.5B, 3B and 8B (A6000, C):** all five criteria hold; fp32 reference error 0.0272 vs 0.0273, 0.0252 vs 0.0253, 0.0191 vs 0.0191.
- **D4:** FA3 is within 2% of the fastest backend at 6 of 6 cells on the H100; FA2 forced is 44.1% slower than FA3 at 1.5B 3584x128.
- **Interconnect decides whether TP=2 helps (Llama-3.1-8B, workload B, A100 40GB, medians of 2):**

| | PCIe host c16 / c64 | NVLink host c16 / c64 |
|---|---|---|
| TPOT TP=1 / TP=2 | 0.70x / 0.47x | **1.65x** / 1.40x |
| output tok/s TP=1 / TP=2 / 2 replicas (aggregate) | 599 / 413 / ~800; 816 / 505 / ~1,480 | 645 / 1,063 / ~841; 915 / 1,705 / ~1,635 |
| custom all-reduce gain (tok/s) | -0.5% / +0.4% | +4.7% / +2.4% |
| 256 MiB all-reduce bus bandwidth | 3.1 GB/s (same with `NCCL_P2P_DISABLE=1`) | 158.1 GB/s (2.2 GB/s with `NCCL_P2P_DISABLE=1`) |

  TP=1 at c64 on the real 40 GB pool preempts 57-66 times per run (KV 100%) on both hosts; TP=2 and each replica at c32 never do. The PCIe VM's link is about 3 GB/s with or without P2P, so TP=2 loses to a single GPU there, and two replicas are 1.8x TP=1 at c64. On NVLink the replicas are level with TP=2 at c64 (a replica at c32 has TPOT 37.6 ms, TP=2 at c64 has 36.5 ms) and 21% behind at c16.

**What this changes.** (1) "Use TRITON_ATTN" is now supported for output equivalence at 1.5B-8B on sm86, and the advice stands only for FA2 hosts; on Hopper keep FA3. (2) Report 9.12's "likely extra part is per-step host cost (H2)" for the 1,024-seq loss is not supported by the busy-fraction measurement at 512x128; it remains open at 128x128 (not testable on the A6000). (3) Day 5's and Day 7's tensor-parallel advice depends on the link, not the model: measure a 256 MiB all-reduce first.

**Caveats.** One A6000, one H100 SXM5, one PCIe host and one SXM4 host (different CPUs and RAM, so the pair isolates the interconnect only up to that); two repeats on the A100 arms (TPOT spread under 1% on the TP arms); busy-fraction windows are short and noisy; the H100 output-check criteria cannot be satisfied by a different kernel (see D2).

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

GPU round 3 (a single A40 pod; `source ../day4-vllm-internals/.venv-gpu/bin/activate`, vLLM 0.34.0; run one job at a time, the GPU is shared by all of them):

```bash
LENS="3584 512" DELAYS=500 bash gpu/run_itl_study.sh results/raw/r3-itl                 # profiled windows; then python -m gpu.step_costs / gpu.kernel_costs <trace dir>
LENS=3584 DELAYS=500 EXTRA_SERVE="--attention-backend TRITON_ATTN" bash gpu/run_itl_study.sh results/raw/r3-itl-triton
for m in qwen2.5-0.5b qwen2.5-0.5b-mha llama3.1-8b; do python gpu/attn_microbench.py --model $m --ctx 3584 --decodes 256 --prefill 7600 --out mb_$m.json; done
REPS=2 bash gpu/run_backend_ab.sh results/raw/r3-backend-ab                              # unprofiled FLASH_ATTN vs TRITON_ATTN; SEQS=1024 for the max_num_seqs arm
REPS=3 bash gpu/run_seqs_isolation.sh results/raw/r3-seqs && python -m gpu.seqs_isolation results/raw/r3-seqs
MODEL=NousResearch/Meta-Llama-3.1-8B-Instruct SEQS=128 BT=2048 LENS=512 DELAYS="150 400" bash gpu/run_itl_study.sh results/raw/r3-itl-8b
bash gpu/run_ncu.sh        # needs readable GPU counters; ERR_NVGPUCTRPERM in a RunPod container
```

GPU round 4 (A40 pod, then a Hopper pod; same venv; `bash gpu/preflight_round4.sh` first; `ulimit -n` is raised by the scripts):

```bash
bash gpu/run_round4_a40.sh results/raw/round4-a40     # output check, ratio-of-18 sweep, 128x128 noise, 1.5B/3B/8B matrix, E1-E3, Q4b sweep (about 3.5 h)
python -m gpu.quality_compare results/raw/round4-a40/quality
python -m gpu.backend_ab_summary results/raw/round4-a40/models                  # --base TRITON_ATTN on a Hopper pod
python -m gpu.e123_summary results/raw/round4-a40/e123
bash gpu/run_round4_h100.sh results/raw/round4-h100   # FA2 / FA3 microbenchmark and FA3 vs FA2 vs Triton end to end (about 45 min)
```

GPU round 5 (Lambda Cloud VMs, user `ubuntu`; `bash gpu/setup_lambda.sh` once per VM, then `bash gpu/preflight_round4.sh`):

```bash
bash gpu/run_round5_a6000.sh results/raw/round5-a6000     # Nsight Compute counters, A/B, busy fraction (nsys), output check at 1.5B/3B/8B
bash gpu/run_round5_h100.sh  results/raw/round5-h100      # same on an H100 with FA3 / FA2 forced / Triton
cd ../day7-capstone && HW=a100-40gb STEPS="probe tp dp p2poff" bash run_round5_a100.sh results/raw/round5-a100/<host>   # 2x A100 PCIe: STEPS="probe tp dp"
```
