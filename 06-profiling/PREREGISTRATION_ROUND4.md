# Round 4 pre-registration (written before any round-4 run)

Scope: the open items of performance-report 9.13 (was 9.12) that fit on one A40 pod and one Hopper pod. Not in scope (A100 pods): 8B data-parallel on NVLink, same-GPU PCIe/NVLink pair; Nsight Compute counters stay blocked unless the host allows them (re-tested at the start of each pod, 2 minutes).
Everything below is a guess written down so it can be refuted. Nothing here has been measured on these configurations.

## Q1. Is `TRITON_ATTN` numerically equivalent to FlashAttention 2? (A40; `gpu/quality_run.py`, `gpu/quality_compare.py`)

Setup: offline vLLM, greedy, 200 prompts cut from the repo's own markdown (64-3,000 tokens), 64 new tokens, `max_num_batched_tokens=2048` so decodes and prefill chunks share steps. Runs: A = FLASH_ATTN, B = TRITON_ATTN, C = FLASH_ATTN with `max_num_batched_tokens=8192` (same kernel, different chunking: the **noise floor**). Reference: HF transformers in float32 on 20 prompts of at most 1,024 tokens (prompt log-probabilities).
Metrics: exact-match rate of completions, mean first-divergence position, mean |delta log-prob| of the chosen token over the common prefix, mean |delta| of prompt log-probs, and for each divergence the log-prob gap between the two candidate tokens.
**Decision rule (equivalent if all hold):** (1) B-vs-A mean |delta chosen-token log-prob| and mean |delta prompt log-prob| are each at most 2x the C-vs-A values; (2) B-vs-A exact-match rate is within 10 points of C-vs-A; (3) the mean prompt-log-prob error of B against the fp32 reference is at most 1.5x that of A; (4) at least 80% of B-vs-A divergences are near-ties (gap below 0.5 nats in run A). If (1)-(3) fail the "+14% to +60%" of 9.9 is not usable advice as it stands, whatever the throughput.
Prediction: equivalent (both kernels are fp16 with fp32 accumulation).

## Q2. Does the 9.9 / 9.10 finding hold at other GQA ratios end to end? (A40; `gpu/run_models_matrix.sh`, `gpu/backend_ab_summary.py`)

Models and GQA group: Qwen2.5-1.5B (12/2 = 6), Qwen2.5-3B (16/2 = 8), Llama-3.1-8B (32/8 = 4). 0.5B (14/2 = 7) is the reference from round 3. FLASH_ATTN vs TRITON_ATTN, `max_num_seqs` 256 vs 1024 (8B: 64 vs 192, because 1024 sequences x 640 tokens needs 655k KV tokens against a pool of about 175k; 8B uses `max_num_batched_tokens=2048`), workloads 512x128 (all seqs) and 3584x128 (1.5B at 256 seqs, 3B at 128 seqs: 256 x 3,712 tokens would not fit the 3B's pool of about 0.9M; 8B: not run, about 7 minutes a run), 2 repeats.
Predictions (attention is a smaller share of a mixed step in a larger model, so the gain shrinks):
- Triton over FA2 at 256 seqs, 512x128: 1.5B +8 to +13%, 3B +4 to +10%, 8B +1 to +5%. At 3584x128: 1.5B +30 to +55%, 3B +15 to +40%.
- FA2 loss at 1024 vs 256 (512x128): 1.5B -4 to -8% (Day 4 saw -4.9%), 3B -2 to -6%, 8B (192 vs 64) 0 to -3%. Triton: within +-1.5% at every size.
Refuted if: Triton is slower than FA2 at any size or workload by more than 2%, or the FA2 1024-seq loss is at least 3 points larger than predicted at any size, or Triton shows a loss of more than 3% at 1024.

## Q3. Does Hopper (FlashAttention 3) have the penalty? (H100; `gpu/attn_sweep.py --fa-version`, `gpu/run_backend_ab.sh` with `FLASH_ATTN:2` / `FLASH_ATTN:3` entries)

Microbenchmark, 0.5B geometry, 256 decodes, ctx 3584, P = 7,600, per-decode mixed-over-pure ratio (A40 FA2 measured 6.58):
- FA2 on the H100: 5 to 8 (the behaviour belongs to the kernel version, not the A40).
- FA3 on the H100: **at most 2** (FA3 packs the GQA group into the query dimension for any query length). If FA3 gives 4 or more, the penalty is not specific to FA2.
End to end (0.5B, 512x128 and 3584x128, 256 and 1024 seqs): FA3 default vs Triton within 5% at 512x128, FA2-on-H100 slower than Triton by at least 10% at 512x128 and 40% at 3584x128; the 1024-seq loss under FA3 within 3%.
If the H100 is faster in absolute terms but the FA2 ratio is unchanged, say so; the practical advice (use FA3 on Hopper) is then what the table supports.

## Q4. The unexplained ratio of 18 at 512 context with P = 7,600 (A40; `gpu/attn_sweep.py`)

Hypothesis H-dead: FlashAttention 2 launches a grid sized by the longest query in the batch (`ceil(max_seqlen_q / block_m)` x batch x heads); every decode sequence gets that many blocks that exit at once. The cost then depends on decodes x max query length, not on the context length.
Grid: ctx {512, 3584} x decodes {64, 256, 512} x P {2, 128, 512, 1,024, 2,048, 4,096, 7,600}, 0.5B geometry. The decode share of a mixed batch is d(P) = mixed(P) - prefill_only(P); the excess is d(P) - d(1,024).
Predictions: (a) the excess at P >= 2,048 is roughly the same number of milliseconds at ctx 512 and ctx 3584 (within 25%); (b) it grows linearly with decodes x P (R^2 >= 0.9 on the pooled points); (c) the fitted dead-block cost k x decodes x P is below 5% of the mixed time for P <= 512, i.e. negligible at the server's 512-token prompts.
Refuted if the excess at ctx 512 is under half or over double the ctx-3584 excess, or if R^2 < 0.7. Then the 18 stays unexplained and I say so.

## Q5. Loose ends (A40)

- **128x128 noise** (`NUM_PROMPTS=4000` in `gpu/run_backend_ab.sh`, 6 repeats, 256 and 1024 seqs, both backends). The 1000-prompt runs last 7-8 s and spread 15,994-18,574 tok/s. Prediction: the spread of a 4,000-prompt cell is at most 4% (max/min - 1) and Triton is ahead of FA2 by 1 to 4%. Refuted if the spread stays above 5%.
- **E1-E3 under vLLM at 1.5B, 3B, 8B** (`gpu/run_e123_vllm.sh`, `gpu/e123_summary.py`; one server per model, `vllm bench serve` at fixed concurrency). Hypotheses as in `experiments.py`: E1 tok/s at concurrency 8 >= 5x concurrency 1; E2 TTFT(2048) / TTFT(512) in [3.8, 5.0]; E3 TPOT changes by at most 5% from 32 to 256 output tokens.
  Predictions: E1 confirmed at all sizes (decode is weight-bandwidth-bound; 7x to 8x). E2: refuted at 1.5B (ratio under 3.8, a fixed per-step floor), borderline at 3B, confirmed at 8B (3.8-5.0, compute-bound). E3: confirmed at all sizes.

## Decision on what counts as "done"
A question is closed when its measurement exists with n >= 2 and 0 failed requests, and the report states confirmed / refuted per prediction above, with the refutations first. A pod item that cannot run (no H100 available, balance) is reported as not done, not substituted.

## Q4b. Added after the first Q4 sweep (A40, 2026-11-19), before the follow-up run

**What the first sweep showed (data: results/measured/a40-round4/sweep/fa2_0.5b.json).** Q4 as written is refuted: the pooled fit has R^2 = 0.27 and the excess at ctx 3584 is about zero (d(P) is flat, 5.87-6.01 ms at 256 decodes), not the same as at ctx 512. At ctx 512 the excess is real and linear in decodes x P (256 decodes: d = 0.99 ms for P <= 512, then 1.14 / 1.41 / 2.00 / 2.89 ms at P = 1,024 / 2,048 / 4,096 / 7,600; 512 decodes: 2.31 -> 5.78 ms; slope about 1.0e-6 ms per decode-token). The ratio of 18 is this effect divided by a very small pure-decode time.

**Refined model (constructed after seeing those numbers, so it is a post-hoc hypothesis and the ctx-3584 agreement does not count as a test):** FlashAttention 2's grid has `ceil(max_seqlen_q / block_m) x batch x heads` blocks and most of the decode sequences' blocks exit at once. Their dispatch cost is hidden while the live decode work keeps the SMs busy and becomes the floor when it does not:
`d(P) = max(d_live, n x (a + k x P))`, with d_live = d(P = 2) for the same (ctx, n), n = decodes.
**Prediction:** fit a and k on the ctx-512 rows only (n = 64, 256, 512, P >= 2,048); then on contexts and decode counts not used in the fit (ctx 1024 and 2048 with n = 128 and 384, P = 128 ... 7,600, same grid) the model predicts d(P) with median relative error at most 15% and maximum at most 30%. It also predicts that the excess is non-increasing with ctx at fixed n and P, and is zero (within 5% of d_live) wherever n(a + kP) < d_live.
**Refuted if** the median error exceeds 15%, or the excess rises with context. Then the dead-block story is wrong and the 18 is explained only as "grows with decodes x P at short context", which I will state as a description, not a mechanism.
Practical weight: low. The server's longest query is the 512-token prompt at 512 context for 512x128 (P = 512: 1-4% of the mixed attention time by this model) and the cost is masked at long context; the interest is that it explains the microbenchmark outlier.


---

# Outcomes (appended after the runs; numbers and caveats in performance-report.md 9.12)

1. **Q1 held.** Triton is equivalent to FlashAttention 2 under all five criteria (logprob differences 0.0132 / 0.0286 against the noise floor's 0.0131 / 0.0292; exact-match 58% vs 54%; 84 of 84 divergences near-ties; fp32-reference error 0.0273 vs 0.0269). The surprise is the control: the same kernel agrees on only 54% of completions.
2. **Q2 mostly held; four refutations.** Triton over FA2 at 512x128: 3B +7.1% and 8B +4.9% inside their ranges, 1.5B +6.0% below (+8 to +13%). 3584x128: +32.5% and +15.6%, both at the low end of their ranges. FA2 loss at 1024 seqs: -5.5% and -5.2%, inside. Refuted: Triton within +-1.5% at 1024 seqs (it gains +2.9% and +2.8%); the 8B 1024-seq arm is impossible on its KV pool and was replaced by 64 vs 192.
3. **Q3: microbenchmark held, end to end refuted in magnitude.** FA2 on the H100 6.78 (range 5-8), FA3 1.28 (at most 2). End to end FA2-on-H100 was not 10% / 40% slower than Triton (+1.6% and -16.6%). Not predicted: FA3 is 52.5% faster than Triton at 3584x128, and Triton loses 5.3% at 1024 seqs on the H100.
4. **Q4 refuted as written, Q4b supported.** R^2 = 0.27 and the 3584 excess is about zero. The post-hoc block-launch model then predicted unseen contexts with median error 0.7%, max 19.5% on the registered set (30.8% on a wider set, 0.8 points over the limit).
5. **Q5: spread refuted, E2 refuted at every size, E1 and E3 held.** 128x128 spread 4.6-5.2% (256 seqs) and 10-12% (1024 seqs); Triton +3.3% at 256 seqs, and -5.7% at 1024 seqs against its own 256 arm. E2 ratios 1.76-2.22 (predicted confirmed at 8B).

Errors made on the way: the first noise run failed 75% of its requests (file-descriptor limit; scripts now raise it and re-run any result with failures); my process-kill command matched my own ssh command line and killed the shell (no data lost).
