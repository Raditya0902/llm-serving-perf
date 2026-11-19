# Round 3 pre-registration (written on the Mac before any run)

Hypotheses and decision rules fixed before the GPU runs, so a refuted prediction is reported as refuted. A40, vLLM 0.34.0
(FlashAttention 2 on sm86), Qwen2.5-0.5B unless stated. Nothing below is a measurement.

## 1. Which kernel carries the extra cost of a decode inside a mixed step (9.6 / 9.9 report numbering)
- **H-attn:** the attention kernel. Decision rule: the kernel family whose per-decode slope in mixed steps is at least 70% of the total
  slope (`gpu/kernel_costs.py`, `decode_slope_share`). Attribution is only trusted when per-step coverage (kernel ms / step ms) is at least 80%.
- **H-other:** GEMM, norm/activation or sampling carries it instead (any other family reaching 70%). If no family reaches 70%, report "spread".
- **Candidate mechanism for H-attn (a guess from memory of the FlashAttention 2 code, to be checked in the microbenchmark, not assumed):**
  when the longest query in the batch is 1 token, FA2 folds the GQA query-head group into the sequence dimension so each KV head is read
  once per group; when any sequence has a longer query that shortcut is off. Prediction: the mixed/pure per-decode ratio in
  `gpu/attn_microbench.py` tracks the GQA group size: about 7 for Qwen2.5-0.5B (14 q / 2 kv heads; the server showed 4-5x), about 4 for
  Llama-3.1-8B (32 / 8), about 1 for the `qwen2.5-0.5b-mha` control (14 / 14). If the control ratio is also well above 1, the GQA story is refuted.
- **If the microbenchmark does not reproduce the server's gap** (mixed/pure per decode below 2x at 3584 context), the cause is outside the
  attention kernel call (CUDA-graph mode: pure decode replays a full graph, mixed steps run piecewise) and `--enforce-eager` / `--attention-backend` runs decide.

## 2. Nsight Compute (only if counters are readable in the container)
- Pure-decode attention is DRAM-bandwidth-bound (DRAM throughput above 60% of peak). The mixed-batch attention kernel moves the same KV bytes
  at a much lower DRAM throughput with low achieved occupancy or few waves per SM (latency-bound). If both are bandwidth-bound, the slowdown is extra bytes (KV re-reads), not stalls.

## 3. Why `max_num_seqs=1024` loses 9-13% (9.3 / 9.8)
- **H1:** decodes riding in expensive mixed steps. **H2:** per-step CPU / scheduler / output-processing overhead that grows with running sequences.
  H3 (KV memory) is already ruled out (peak KV 21%, no preemptions).
- Rule (`gpu/seqs_isolation.py: verdict`, loss = 1024 arm vs 256 arm, same workload, signed %): H2 if the 1024 arm's GPU busy fraction is at least 10
  points lower than the 256 arm's, or if the loss stays within 3 points of the baseline workload's across 128x128, 512x512, 128x512. H1 if the loss at 512x512 and at
  128x512 is at least 5 points smaller than at 512x128 and the busy fractions are within 5 points. Anything else: "mixed".

## 4. Llama-3.1-8B step costs under vLLM (BT=2048)
- Prefill coefficient 100-180 ms per 1k tokens (about 19x the 0.5B's 9 ms, scaled by non-embedding parameters).
- Pure-decode step at about 23 ms plus KV reads (16 GB of weights / 0.696 TB/s).
- Mixed-decode cost per sequence stays several times its pure-decode value; if H-attn and the GQA mechanism hold it is about 4x for 8B, below the 0.5B's 7x.
- GPU busy fraction near 1 (device-bound), unlike the launch-bound 0.5B.

---

# Outcomes (appended after the runs; numbers and caveats in performance-report.md 9.9-9.12)

1. **Kernel for the mixed-step decode cost: H-attn held.** Attention is 162 of 232 ms of a 3584-context mixed step (kernel coverage 99.9%). The decode-slope rule (70% of the per-decode slope) could not be applied: the decode count is constant in the window (249-255), so the slope is not identified; this also invalidated 9.6's 0.51 ms/decode as a per-decode number. The mechanism guess was **supported by the microbenchmark** (mixed/pure per decode 6.58 for GQA 7, 3.65 for GQA 4, 1.00 for the no-GQA control; predictions about 7 / 4 / 1). The microbenchmark also showed it is a step function at query length 2 (not grid size), and a `TRITON_ATTN` run confirmed it end to end (mixed attention 55 vs 162 ms; +60% throughput at 3584 in). The extra ratio of 18 at 512 context with a 7,600-token chunk was not predicted and not explained.
2. **Nsight Compute: not possible.** `ERR_NVGPUCTRPERM` on the pod (tool installed, counters refused by the host). The byte-level prediction was not tested.
3. **`max_num_seqs=1024` loss: H1 supported by the intervention, not by the rule as written.** Applied literally, the verdict rule returns H2 because its busy-fraction criterion fires (0.82 vs 0.68), but those windows are profiled with 72-89% kernel coverage, the unprofiled grid shows no loss where decode dominates (+3.8% at 128 x 512), and switching to `TRITON_ATTN` removes the loss (-9.4% to +0.4%). The busy-fraction criterion should have been specified without a profiler. Loss at 512 output tokens: -2.0%, as H1 predicted (a 5-point shrink); 128 x 128 stayed at -7.0% in the grid (-3.1% in the later A/B).
4. **8B step costs: three of four predictions held** (prefill 122-138 ms per 1k tokens; pure decode 48 ms = weight-read GEMM 29 + attention 17; busy fraction 0.98-0.99). The fourth (mixed-decode cost about 4x its pure value) is untestable from the traces (constant decode count) and was checked with the microbenchmark instead (3.65x at 3584, 6.3x at 512 context).
