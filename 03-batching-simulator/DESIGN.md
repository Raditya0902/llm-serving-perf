# Design: continuous-batching scheduler simulator

A discrete-event simulator of one LLM serving engine. It decides *which sequences run in each
forward pass* and charges each pass a simulated time from a documented cost model. Nothing here
touches a GPU; all times are simulated milliseconds.

```
 Incoming requests (arrival_time)
        |
        v
  Waiting queue  --(ranked: FCFS or priority)-->  Scheduler.plan()
        ^                                              |  admit while: slots, KV blocks, token budget
        | preempt (KV freed, recompute later)          v
        +------------------------------------   Running batch  <-->  KV block manager (counts blocks)
                                                       |
                                                       v
                                  IterationPlan(prefill chunks, decode set)
                                                       |
                                  CostModel.iteration_time()  -> advance clock
                                                       |
                                                       v
                                      Completed requests (KV freed, timestamps recorded)
```

| Module | Role |
|---|---|
| `request.py` | `Request` lifecycle: status, prefill/decode progress, timestamps (the only mutable class) |
| `cost_model.py` | per-iteration time model |
| `kv_blocks.py` | paged-KV block accounting, all-or-nothing allocation |
| `scheduler.py` | `ContinuousScheduler` (FCFS / priority, chunked prefill, preemption) and `StaticScheduler` |
| `simulator.py` | event loop, per-iteration records, event log |
| `metrics.py` | throughput, latency percentiles, TTFT, inter-token gaps, utilization proxy |

## Event loop

Per iteration, with clock `t`: (1) requests with `arrival_time <= t` join the waiting queue; (2) if
nothing is running or waiting, jump `t` to the next arrival; (3) the scheduler returns an
`IterationPlan`; (4) the cost model gives the duration; (5) at `t + duration` prefill chunks add KV,
a finished prefill emits the first token (that instant is TTFT), each decoder adds one KV entry and
emits a token, and finished requests free their KV blocks.

A request always generates exactly `max_output_tokens` (no EOS model). Its KV peaks at
`prompt + max_output - 1` tokens, because the last token's KV is never stored.

## Compute-cost model

```
iter_ms    = overhead + max(compute_ms, memory_ms)
compute_ms = 0.04 ms * (prefill_tokens + decode_tokens)
memory_ms  = 7.0 ms  +  0.0001 ms * (sum of KV context of decoding sequences)
overhead   = 0.3 ms
```

Compute is linear in tokens processed; memory time is dominated by streaming the weights once per
iteration regardless of batch size, plus each decoder's KV reads. The numbers are illustrative
(roughly an 8B fp16 model on an A100-class GPU), not measured. This one formula is what produces the
trade-offs a constant iteration time would hide:

* **Decode batches are memory-bound**: 1 sequence costs ~7.3 ms/token, 32 sequences ~8.9 ms for 32
  tokens, so batching multiplies throughput.
* **Long prefills are compute-bound**: a 2048-token prompt costs ~82 ms, over 10x a decode step, so
  an unchunked long prompt stalls every running decoder.
* **Small prefill chunks are almost free**: up to ~175 tokens hide under the weight-read time.

## Scheduling rules

Every iteration the continuous scheduler does, in order:

1. **KV for decoders.** Each decoder needs room for its next token. In `on_demand` mode a new block
   is needed every `block_size` tokens; if the pool is empty, the lowest-ranked running request is
   **preempted by recompute** (blocks freed, progress reset, re-queued; the tokens it already
   streamed are kept, so the recompute prefill covers prompt + generated tokens).
2. **Admission, in rank order**, while a running slot, KV blocks, and token budget remain. Strict
   head-of-line: if the best-ranked waiting request cannot be admitted, nothing behind it is.
3. **Token budget.** Each decoder costs 1 token; the rest of the budget goes to prefill. Unchunked
   mode must fit a whole prompt in the leftover budget. Chunked mode feeds prompts in whatever
   remains, so a long prompt is spread over several iterations.

Ranking is `(arrival, id)` for FCFS and `(priority, arrival, id)` for priority (lower number first).
The same key, reversed, picks the preemption victim. With `preempt_for_priority`, a blocked
higher-priority request evicts the lowest-ranked running request of strictly lower priority.

**KV modes.** `on_demand` allocates the prompt at admission and grows by blocks as tokens are
generated (admits more, can preempt). `full_reserve` reserves prompt + max output up front (never
preempts, admits fewer).

**Static batching** (baseline): when idle, form a batch of up to N requests (full reservation),
prefill it, then decode in lockstep. A finished sequence's slot stays occupied and padded until the
whole batch drains; nothing is admitted mid-batch. Padding costs time and counts no useful tokens.
Requests are returned when *their* last token is emitted, which flatters static batching; a
non-streaming server would answer only at the end of the batch.

**Sequential** is the continuous engine with `max_running_seqs = 1`.

## Timeline: requests entering and leaving the active batch

Demo scenario: 4 batch slots, chunked prefill (token budget 256), a deliberately small KV pool
(75 blocks of 16 tokens), FCFS. Request 2 has a 1024-token prompt that arrives while 0, 1 and 3 are
decoding; request 4 is high priority (p0) but FCFS ignores that. Generated by
`run_experiments.py --exp T` into `results/timeline.txt`; the Gantt version is
`results/timeline.png`.

```
time: 0 ms -> 749 ms, 1 column = 7.5 ms
P=prefill  D=decode  -=in batch, idle  .=waiting  X=preempted  (blank = not arrived / finished)
req 0   p1 P=96   O=40  |PDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD                                                           |
req 1   p1 P=64   O=24  |PPDDDDDDDDDDDDDDDDDDDDDDD                                                                           |
req 2   p1 P=1024 O=16  | .......................PPPPPPPPDDDDDDDDDDDDDD                                                      |
req 3   p1 P=48   O=60  |  ............................PX........PPDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD|
req 4   p0 P=200  O=20  |        .....................................PPPDDDDDDDDDDDDDDDDDDD                                 |
req 5   p1 P=32   O=12  |            .................................PPPDDDDDDDDDDD                                         |
req 6   p1 P=512  O=30  |            .................................PPPPPDDDDDDDDDDDDDDDDDDDDDDDDDDDDD                     |
req 7   p1 P=64   O=16  |             .............................................PPDDDDDDDDDDDDDDD                         |
```

Reading it:

* Requests 0 and 1 are admitted the moment they arrive and decode together; the batch is continuous, so
  later requests join without waiting for them to finish.
* Request 2's long prompt cannot be admitted until KV frees up (`.`), then prefills in 8 chunks (`P`)
  while the other decoders keep generating.
* Request 3 queues behind request 2 (strict arrival order). It is admitted just after request 2, but one
  iteration later the pool runs out; as the youngest running request it is the victim and is **preempted
  (`X`)**. Its KV is freed, it waits again, then re-prefills its prompt plus the tokens it had generated.
* Requests 4 to 7 wait while request 2 holds 65 of the 75 blocks, even though batch slots are free:
  admission is limited by KV and strict FCFS order (head-of-line blocking). Request 4 is high priority but
  FCFS ignores that; they are admitted together once request 2 finishes.

## What is simplified, and what resembles production systems

| Mechanism | In this simulator | In production engines (e.g. vLLM) |
|---|---|---|
| Iteration-level scheduling | Yes: the batch is re-formed every iteration | Same idea (continuous batching) |
| Token budget per iteration | Yes, decode tokens count first | Same (`max_num_batched_tokens`) |
| Chunked prefill | Yes, chunk = leftover budget | Same idea, plus extra heuristics |
| Paged KV blocks | Block *counts* only | Real block tables, physical GPU memory |
| Preemption by recompute | Yes, generated tokens kept | Same; swap-to-CPU is also available |
| Priority scheduling | Rank queue + optional preemption (only when the eviction lets the blocked request in) | Similar priority policies exist |
| Execution time | Analytic roofline-style formula | Real kernels: attention is quadratic in prefill, varies with batch shape, CUDA graphs, etc. |
| Output length | Fixed, known to the simulator | Unknown until EOS; schedulers cannot reserve exactly |
| Prefix caching, speculative decoding, LoRA, multi-modal | Not modelled | Present |
| Parallelism | One GPU, no TP/PP, no network | Multi-GPU, collectives, pipeline bubbles |
| Scheduler CPU cost, sampling, tokenisation | Zero | Real overhead, sometimes visible |
| Admission | Strict head-of-line, no bypass | Policy-dependent |
| KV reservation (`on_demand`) | Whole prefill goal reserved at admission, growth by block only during decode | Block-by-block, including during chunked prefill |
| Static batch formation | Starts with whatever is waiting at that instant | Real batchers use a size-or-timeout window |
| Tenancy / fairness | None | Usually external to the engine |
