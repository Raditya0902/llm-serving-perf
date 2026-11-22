# Closed-book assessment: ten answers

Each answer carries an equation, diagram, example, or a pointer to something implemented in this lab. Benchmark numbers are **SIMULATED**
(`03-batching-simulator` engine, illustrative A100-40GB constants) unless tagged **[measured-Mac]** (`06-profiling`, M4/MPS) or **[source]** (vLLM v0.34.0 source, `04-vllm-internals`).
Nothing here has GPU measurements. Section numbers refer to [REPORT.md](REPORT.md).

---

## 1. The entire inference pipeline, prompt to streamed tokens

```
HTTP request -> tokenize -> [engine: add to waiting queue]
   loop every step:  schedule (token budget + KV blocks) -> forward pass -> sample -> append token / stop check
   each new token id -> detokenize -> stream to the client            (TTFT = first token, TPOT = gap after)
```

1. **Frontend:** the API server tokenizes the prompt and sends it over ZMQ to the engine core (`AsyncLLM` -> `EngineCoreProc`) [source].
2. **Scheduling:** the `Scheduler` puts it in `waiting`. Each step it hands out tokens so each request's `num_computed_tokens` catches up with
   its `num_tokens`; running requests first, then waiting ones, subject to the token budget, `max_num_seqs` and free KV blocks.
3. **Prefill:** the prompt's tokens run through all layers in one or more chunks. Attention writes each token's K and V into the request's KV blocks.
   The last prompt token's logits give the **first output token** (end of TTFT).
4. **Decode:** each step feeds the last token, reads the whole KV cache for attention, appends one K/V entry per layer, and samples the next token.
5. **Sampling and stop:** the worker samples (greedy/top-p...), starts an async device-to-host copy and returns token ids; the scheduler appends them, checks EOS/length, frees blocks of finished requests.
6. **Streaming:** token ids return to the frontend, are detokenized incrementally and sent as SSE chunks.

Diagram: [results/architecture.png](results/architecture.png). Implemented in miniature: `01-inference-loop`'s cached greedy loop (real model, Mac) and `03-batching-simulator`'s iteration-level engine (simulated).

## 2. Why prefill is often compute-bound and decode often bandwidth-limited

**Arithmetic intensity** = FLOPs per byte moved. Each step must read all weights once (`W` bytes) and does `2 x params x T` FLOPs for `T` tokens in the batch:

```
intensity = (2 x params x T) / W = (2 x params x T) / (params x 2 B)  =  T  FLOP/byte      (fp16 weights)
ridge     = peak_FLOPS / HBM_BW  = 312e12 / 1.555e12 ~ 200 FLOP/byte  (A100, assumed; ~100 at 50% MFU)
```

- **Prefill:** `T` = prompt tokens (e.g. 2,560) >> ridge, so the step is limited by FLOPs: 2,560 tokens x 0.103 ms/token ~ **264 ms** of compute, against ~10 ms of weight reading.
- **Decode:** `T` = batch size (1-64) << ridge, so the step is limited by streaming weights (plus the KV cache): **10.3 ms** for Llama-3-8B on 1.555 TB/s, almost independent of `T`.
  That is why batching raises decode throughput nearly for free, up to `T` ~ 100-200.

Evidence: the simulator's cost model is `iter = overhead + max(compute, memory)` (`03-batching-simulator/cost_model.py`, constants derived in `kv_sizing.cost_model_for`). On real
hardware, `06-profiling` **[measured-Mac]** found decode is device-bound on weight reads (matmuls 74% of a 15.9 ms step) and batching decode gives ~5.6-6.1x at batch 8 while end-to-end only 2.8-3.5x
because compute-bound prefill does not amortize. Caveat: Apple unified memory, not NVIDIA.

## 3. Deriving the KV-cache requirement for a GQA transformer

Per layer, per token we store one K and one V vector for each **KV head** (not each query head):

```
bytes/token         = 2 x L x n_kv x d_head x b
bytes/sequence(S)   = S x bytes/token
bytes/pool          = sum over live sequences of their S  x  bytes/token
GQA saving vs MHA   = n_heads / n_kv
```

Example, Llama-3-8B (L = 32, n_kv = 8, d_head = 128, b = 2 B fp16): **131,072 B = 128 KiB/token**; an 8,192-token sequence needs 1 GiB; with 32 KV heads (MHA) it would be 4 GiB.
On a 40 GiB GPU at `util` 0.85 with 14.96 GiB of weights and a 2 GiB reserve, the pool is 17.04 GiB = **139,600 tokens** (`kv_sizing.kv_pool_blocks`; REPORT §2). Implemented:
`02-kv-cache-memory/memory_calculator.py` (`kv_bytes_per_token`, `max_concurrent_seqs`, `gqa_savings`).

## 4. Static, dynamic and continuous batching

```
static      [ r1 r2 r3 r4 ]----------------------------- wait for the slowest ----> next batch
dynamic     collect requests for up to X ms or N requests, then run them as one static batch
continuous  step: [r1 r2 r3 r4] -> r2 done -> [r1 r3 r4 r5] -> r1 done -> [r3 r4 r5 r6] ...   (membership changes every iteration)
```

- **Static:** fixed batch, runs to completion; finished sequences leave padded slots idle and nothing is admitted mid-batch. Throughput is bounded by the longest output.
- **Dynamic:** still a static batch once launched; the only difference is *when* the batch is formed (timeout or size trigger), trading queueing delay for fuller batches.
- **Continuous (iteration-level):** the scheduler re-plans every forward pass, so finished sequences free their slot and KV immediately and new ones join at once.

Implemented and compared in `03-batching-simulator` (`StaticScheduler` vs `ContinuousScheduler`): continuous batching gave 13x the throughput of sequential and 2.1x static there (`03-batching-simulator` README, simulated), and
`padded_slot_frac` quantifies the static waste. vLLM goes further: no separate prefill and decode phases at all [source].

## 5. How vLLM decides which requests receive computation

Every step `Scheduler.schedule()` (SCH:716) makes one pass over two budgets, `token_budget` and `input_budget` (`max_num_batched_tokens`) [source: `04-vllm-internals` CHECKPOINT Q2]:

```
for req in running:                                   # already-running requests first, in list order
    n = num_tokens_with_spec - num_computed_tokens    #   clamp: long_prefill_token_threshold, budgets, max_model_len
    if allocate_slots(req, n) fails: preempt a victim (FCFS: running[-1]) and retry
if nothing was preempted:
    for req in waiting (KV holders first, then policy order FCFS/priority):
        if slots (max_num_seqs), budget, and allocate_slots succeed: admit   else: stop (no eviction to admit)
```

So: *running requests -> KV holders -> policy order*, hard stop at the first waiting request that cannot get tokens or blocks. Without chunked prefill a prompt larger than the leftover budget blocks the
queue (head-of-line). Same logic in miniature: `03-batching-simulator/scheduler.py`; the differences (vLLM checks the whole prompt fits but allocates only the first chunk) are in `04-vllm-internals` `COMPARISON.md`.
Observed effect of the budget here: shrinking `max_num_batched_tokens` 8192 -> 2048 at B@64 improved mean TPOT 5% (REPORT §8) **[SIM]**.

## 6. Paged KV-cache allocation: how it works and what it solves

```
sequence logical tokens:   [t0..t15][t16..t31][t32..]            block size 16
block table (per request):  [ 7 ]     [ 2 ]     [ 9 ]   -> physical blocks anywhere in the pool
free list:                  blocks {1,3,4,...}               allocate one block when a sequence crosses a 16-token boundary
```

- **Mechanism:** the pool is divided into fixed-size blocks; each request holds a **block table** mapping logical positions to physical blocks; attention kernels follow the table. Blocks are allocated on demand and returned to a free list on completion;
  full blocks are hash-chained so identical prefixes can share blocks (ref-counted) [source: `BlockPool`, BP:107].
- **Problem solved:** contiguous allocation must reserve `max_len` per request up front, so memory is stranded by internal fragmentation (reserved but unused) and external fragmentation (holes between sequences),
  capping concurrency far below what actually fits. Paging makes waste at most one partial block per sequence.
- **Evidence in this lab:** `02-kv-cache-memory` Exp C (simulated): ~83% of reserved memory unused with contiguous allocation vs 1-16% with paging (`02-kv-cache-memory/paged_allocator.py` vs `ContiguousAllocator`).
  `07-capacity-study` relies on it: the "demand vs pool" arithmetic in REPORT §2 is only valid because waste is small.

## 7. What happens inside vLLM when it runs out of KV capacity

It depends on which side of the scheduler the request is on [source: `04-vllm-internals` CHECKPOINT Q4]:

- **A running request needs one more block** (`allocate_slots` -> `None`, SCH:930): vLLM **preempts a victim** (the last in `running` for FCFS; max `(priority, arrival)` for PRIORITY), calls `_preempt_request` (SCH:1853):
  frees its blocks, sets `num_computed_tokens = 0`, marks it `PREEMPTED`, puts it at the **front** of `waiting`. It is later **recomputed** (re-prefilled with prompt *and* generated tokens), reusing any still-cached prefix blocks.
  If the request ends up preempting itself, the running loop stops.
- **A waiting request cannot get blocks:** admission stops (SCH:1516-1523); a running request is **never** evicted to admit a waiting one; and a step that preempted anything admits no waiting requests.
- **Mitigations before that point:** whole-prompt admission check (`scheduler_reserve_full_isl`) and watermark headroom.

Seen in the benchmark **[SIM]**: B at concurrency 64 needs ~176k tokens against a 139.6k pool; the pool sits at **100%**, with **64 preemptions per run** (baseline) and TTFT of 7 s (mean), against 0 preemptions and 0.6 s at
concurrency 16 (REPORT §3). Remedies and their costs: more pool (`util` 0.92: 56 preemptions, TTFT -50%), more memory (TP=2: 2.9x capacity), cheaper KV (fp8), or fewer concurrent sequences (caps: no preemption, but TTFT +114-295%).

## 8. Tensor parallelism and its communication requirements

```
MLP:  y = ( act(x A) ) B       A split by COLUMNS (A = [A1|A2]),  B split by ROWS ([B1;B2]);  partial outputs y_i = act(x A_i) B_i
      y = y_1 + y_2  ->  one AllReduce(sum)         attention: heads split across ranks, out-projection row-parallel -> one AllReduce
```

- Per layer: **2 AllReduces** (after attention and after the MLP). Per decode step: `2L` of them, each moving `T x h x 2 B` (fp16 activations).
  Ring AllReduce time `~ alpha + [2(n-1)/n] x bytes / busbw`.
- Llama-3-8B, 64 sequences: 64 AllReduces x 512 KiB per step; step time = `weight_read / n + comm`.
- **Requirement:** a fast, low-latency link. Analytic results **[assumed constants; multi_gpu.py]**: TP=2 speeds a decode step **1.64x on NVLink-class, 1.05x on PCIe-class, 0.88x (slower) over 100 GbE**.
  TP also shards weights and KV heads, so group KV capacity grows faster than `n` (2.88x at TP=2). Keep TP inside a node; use PP or DP across nodes.
- Implemented: `05-tensor-parallel/tensor_parallel.py` (`ColumnParallelLinear`, `RowParallelLinear`, `ParallelMLP`, verified against an unsharded reference on Gloo/CPU), `bench_model.py` (alpha-beta fit), and
  `07-capacity-study/multi_gpu.py`. Not run: NCCL on GPUs (`05-tensor-parallel` RUNBOOK, REPORT §5).

## 9. Identifying the dominant bottleneck in an inference workload

A decision procedure, cheapest signal first (`bottleneck.py` encodes steps 1-4 as named-threshold rules):

1. **KV pool at ~100% and preemptions > 0?** -> **capacity-bound** (`vllm:kv_cache_usage_perc`, `vllm:num_preemptions`). Fix: memory, KV dtype, concurrency.
2. **Running batch pinned at `max_num_seqs` with a growing waiting queue?** -> **scheduler-bound** (admission cap).
3. **TPOT vs the weight-read floor** `weights / HBM_BW` (10.3 ms here): within ~1.5x -> **bandwidth-bound decode** (only batching, quantization or more bandwidth helps).
4. **TPOT well above the floor** -> **prefill interference / compute**: look at `max_num_batched_tokens` and chunking.
5. **Wall time vs GPU busy time** (nsys gaps; CPU enqueue time vs step time): large gaps -> **host/launch-bound**. `06-profiling` **[measured-Mac]**: CPU enqueue 3.0 ms of a 15.9 ms step, so *device-bound*; `01-inference-loop`'s launch-overhead guess was refuted.

Example from the matrix **[SIM]**: B@64 -> step 1 fires (100%, 64 preemptions); B@16 -> TPOT 23 ms is 2.2x the floor -> step 4; A@4 -> TPOT 11.1 ms -> step 3 (`results/sim/analysis.json`). Rules are heuristics:
they label, they do not prove; confirm with a profile.

## 10. Improving throughput without violating a P95 latency objective

**Fix the constraint first, from the hardware**, then search for the highest-throughput point inside it. I set `L = 3 x` the baseline P95 at the lowest concurrency (REPORT §8): L = 5.3 s (chat), 19.9 s (long-context).
Then, in order of expected payoff:

```
maximize tok/s   s.t.   P95_e2e <= L        tok/s = N_conc x (1 / TPOT)      P95_e2e ~ TTFT + out_len x TPOT_tail
```

1. **Find the knee:** raise concurrency until P95 reaches `L`. Chat: 64 is feasible (2,614 tok/s, P95 4.1 s). Long-context: only 16 is (629-638 tok/s, P95 12.4 s); 64 gives 841-907 tok/s but P95 ~36 s.
2. **Add capacity where the bottleneck is KV** (TP=2 on NVLink: pool 2.9x so B@64 stops thrashing; fp8 KV; higher `util`). This shifts the whole curve; the others only slide along it.
3. **Trade one latency for another with chunk size** (`max_num_batched_tokens` 8192 -> 2048: TPOT -5%, tok/s +5% at B@64) only if it helps the *binding* metric; here P95 e2e did not move.
4. **Do not "fix" overload with admission caps:** they replaced preemption with queueing (TTFT +114-295%, tok/s -3 to -24%).
5. **Validate with repeats:** adopt a change only if it clears the A/A noise floor (1.05% for tok/s here); `tuned`'s +1.5% at the best feasible cell did not, so the answer was *keep baseline*.

All numbers are **SIMULATED**; the procedure, not the magnitudes, is the transferable part. Implemented: `objective.py` (threshold, selection, before/after), `stability.py` (noise rules), `run_matrix_sim.py --ablation`.
