# Session 2 — vLLM request path: annotated call-flow

> **Pinned to vLLM `v0.34.0`, commit `62d4bc3c7e33b1b6c906e233e6af2d3506c482ca`.**
> The checked-out source is the authority; where it disagrees with the task
> brief or older docs, this file follows the source (see [Drift from the brief](#drift-from-the-brief)).
>
> **How much to trust the line numbers.** Every `file:line` that points at a
> `def`/`class` is machine-checked: `python verify_trace.py --src vllm_src`
> resolves all 151 cited symbols with `ast` and compares line numbers
> (`trace_manifest.json`). Line numbers *inside* a function body (e.g.
> `scheduler.py:1010`) were read from the same commit but are not machine-checked;
> if you bump the vLLM version, re-run `fetch_vllm_source.sh`, then
> `verify_trace.py --write-lines`, then re-read the interior citations.

Paths below are relative to the vLLM repo root. Abbreviations: **ENG** =
`vllm/v1/engine/core.py`, **SCH** = `vllm/v1/core/sched/scheduler.py`, **KVM** =
`vllm/v1/core/kv_cache_manager.py`, **KVC** = `vllm/v1/core/kv_cache_coordinator.py`,
**STM** = `vllm/v1/core/single_type_kv_cache_manager.py`, **BP** =
`vllm/v1/core/block_pool.py`, **MPX** = `vllm/v1/executor/multiproc_executor.py`,
**GW** = `vllm/v1/worker/gpu_worker.py`, **GMR** = `vllm/v1/worker/gpu/model_runner.py`.

## Drift from the brief

| Brief says | Source at v0.34.0 |
|---|---|
| `vllm/v1/worker/gpu_model_runner.py` | **Does not exist.** The only GPU runner is `GPUModelRunner` in `vllm/v1/worker/gpu/model_runner.py:206`, with helpers `gpu/states.py`, `gpu/block_table.py`, `gpu/input_batch.py`, `gpu/buffer_utils.py`, `gpu/async_utils.py`. `Worker._make_model_runner` (GW:579) builds it; the only alternative is `MMEncoderModelRunner` for encoder-only multimodal models. `VLLM_USE_V2_MODEL_RUNNER` is no longer a switch (it only warns or raises in `vllm/platforms/interface.py`). |
| Scheduler has "skipped waiting" requests | There are four waiting-side containers: `waiting`, `kv_holding_waiting`, `async_kv_waiting`, `deferred_waiting` (SCH `Scheduler.__init__` :87). |
| `max_concurrent_batches` lives in the executor | It is `VllmConfig.max_concurrent_batches` (`vllm/config/vllm.py:560`) = `pp_size` (+1 with async scheduling), consumed by `EngineCore.__init__` (ENG:138). |
| One model call per step | Two RPCs: `execute_model` (forward) then `sample_tokens` (sampling). They are split so the CPU can compute the grammar bitmask while the GPU runs the forward pass. |

## 0. Two processes, three threads — where state lives

```text
 API server process                EngineCoreProc process (CPU: scheduling)            Worker process(es) (one per GPU rank)
 ───────────────────               ─────────────────────────────────────────           ───────────────────────────────────────
 AsyncLLM / client   ──ZMQ──►  [input thread] process_input_sockets  ENG:2161
                                    │ preprocess_add_request ENG:1258  (hash blocks,
                                    │   grammar init — off the busy loop)
                                    ▼  input_queue
                               [busy-loop thread] run_busy_loop ENG:1794
                                    │  _process_input_queue ENG:1820
                                    │  _process_engine_step ENG:1855 → step_fn()
                                    │      Scheduler  (SCH:86) ── owns: waiting/running queues,
                                    │      KVCacheManager (KVM:131) ── owns: BlockPool free list,
                                    │                                    prefix-cache hash map,
                                    │                                    req_id → block list
                                    │  rpc_broadcast_mq.enqueue ─shm ring─►  WorkerProc.worker_busy_loop MPX:1188
                                    │                                         Worker → GPUModelRunner (GMR:206) ── owns:
                                    │                                         RequestState (states.py:9), BlockTables
                                    │                                         (block_table.py:17), KV tensors, CUDA graphs
                               [output thread] process_output_sockets ENG:2270
 ◄─────────ZMQ────────────────  output_queue ◄── EngineCoreOutputs
```

The scheduler and block manager are **pure CPU bookkeeping** in the engine-core
process; the GPU worker holds the actual KV tensors and a *mirror* of each
request's block ids. Only diffs cross the process boundary each step.

## 1. Master call-flow (one request, one decode step in steady state)

```text
EngineCoreProc.run_busy_loop                                             ENG:1794
 ├─ _process_input_queue                                                 ENG:1820
 │    └─ _handle_client_request (ADD)                                    ENG:1957
 │         └─ EngineCore.add_request                                     ENG:564
 │              └─ Scheduler.add_request                                 SCH:3065
 │                   └─ _enqueue_waiting_request                        SCH:2776   → waiting / kv_holding_waiting / async_kv_waiting / deferred_waiting
 └─ _process_engine_step                                                 ENG:1855
      └─ step_fn  ==  EngineCore.step (ENG:713)  or  step_with_batch_queue (ENG:758)
           │
           ├─① Scheduler.schedule(throttle_prefills)                     SCH:716
           │    ├─ budgets: token_budget = max_num_scheduled_tokens; input_budget = max_num_batched_tokens   SCH:735,738
           │    ├─ RUNNING loop   while token_budget > 0                 SCH:811
           │    │    ├─ num_new_tokens = num_tokens_with_spec + placeholders − num_computed_tokens          SCH:855
           │    │    ├─ clamp: long_prefill_token_threshold, token_budget, input_budget, max_model_len      SCH:860-873
           │    │    ├─ KVCacheManager.allocate_slots(req, n, lookahead)  KVM:401   ◄── SCH:930
           │    │    │     ├─ KVCacheCoordinator.get_num_blocks_to_allocate        KVC:194
           │    │    │     │     └─ SingleTypeKVCacheManager.get_num_blocks_to_allocate   STM:181
           │    │    │     ├─ shortage?  required > free − reserved  → return None         KVM:605-609
           │    │    │     ├─ KVCacheCoordinator.allocate_new_blocks               KVC:315
           │    │    │     │     └─ SingleTypeKVCacheManager.allocate_new_blocks   STM:375
           │    │    │     │           └─ BlockPool.get_new_blocks                  BP:681  (lazy eviction of cached blocks)
           │    │    │     └─ KVCacheCoordinator.cache_blocks → BlockPool.cache_full_blocks   KVC:358 / BP:203
           │    │    ├─ None?  ──► victim = running[-1] (FCFS) | max(priority, arrival) (PRIORITY)      SCH:948-954
           │    │    │            Scheduler._preempt_request                           SCH:1853  → free blocks, num_computed_tokens=0, status=PREEMPTED, waiting.prepend
           │    │    │            retry allocate_slots; if the request preempted itself → break         SCH:994-1010
           │    │    └─ success: token_budget −= n; req_to_new_blocks[id] = blocks                      SCH:1013-1019
           │    ├─ WAITING loop   (skipped entirely if anything was preempted this step)               SCH:1066
           │    │    ├─ _handle_blocked_waiting_request (grammar / remote KV ready?)                   SCH:3668  → skip_request SCH:1076
           │    │    ├─ _get_local_prefix_cache_hit → KVCacheManager.get_computed_blocks               SCH:681 / KVM:288
           │    │    ├─ num_new_tokens = request.num_tokens − num_computed_tokens                      SCH:1356
           │    │    ├─ no chunked prefill and n > budget → break (head-of-line block)                 SCH:1387-1394
           │    │    ├─ KVCacheManager.allocate_slots(req, n, new_computed_blocks=…,                   SCH:1497
           │    │    │     full_sequence_must_fit=scheduler_reserve_full_isl[default True])          (checks whole prompt fits; allocates only n tokens)
           │    │    ├─ None → break (NO preemption to admit a waiting request)                        SCH:1516-1523
           │    │    └─ admit: waiting.pop → running.append, status=RUNNING, budgets −= n              SCH:1609-1637
           │    ├─ build SchedulerOutput(new_reqs, cached_reqs, num_scheduled_tokens, …)               SCH:~1715-1804 (output.py:220)
           │    └─ _update_after_schedule: num_computed_tokens += scheduled   (BEFORE the GPU runs!)   SCH:1899
           │
           ├─② Executor.execute_model(scheduler_output, non_block=True)  MPX:430 → collective_rpc MPX:471
           │       rpc_broadcast_mq.enqueue((method, args, kwargs, output_rank))   — one shm write, all ranks read
           │                                                                         ═══════ process boundary ═══════
           │       WorkerProc.worker_busy_loop (MPX:1188) → _execute_worker_rpc (MPX:1197)
           │         └─ Worker.execute_model (GW:1449) → GPUModelRunner.execute_model (GMR:1826)
           │              ├─ finish_requests GMR:1255 · add_requests GMR:1284 · update_requests GMR:1340   (apply scheduler diffs)
           │              ├─ BlockTables.apply_staged_writes                                              (one fused H2D write)
           │              ├─ prepare_inputs GMR:1458 · prepare_attn GMR:1665  (positions, slot_mapping — Triton kernels)
           │              └─ forward: CUDA graph (FULL / PIECEWISE) or eager; stashes ExecuteModelState
           │
           ├─③ Scheduler.get_grammar_bitmask  SCH:2277   (CPU work overlaps the GPU forward)
           │
           ├─④ Executor.sample_tokens(grammar_output)  MPX:443 → Worker.sample_tokens GW:1442 → GPUModelRunner.sample_tokens GMR:2216
           │       sample → AsyncOutput (async D2H copy, GMR:2311 / async_utils.py:117) → postprocess_sampled GMR:1782
           │       output rank only: handle_output MPX:1161 → enqueue_output MPX:1130 → response MQ
           │                                                                         ═══════ process boundary ═══════
           └─⑤ Scheduler.update_from_output(scheduler_output, model_runner_output)   SCH:2301
                ├─ _update_request_with_output → append tokens, check_stop           SCH:2879 / sched/utils.py:98
                ├─ spec-decode rejection: num_computed_tokens −= num_rejected
                ├─ stopped? → _free_request SCH:3170 → _free_blocks SCH:3204 → _free_request_blocks SCH:3224
                │                 └─ KVCacheManager.free KVM:650 → KVC:404 → STM:596 → BlockPool.free_blocks BP:798
                └─ EngineCoreOutputs → output_queue → [output thread] process_output_sockets ENG:2270
```

`step_with_batch_queue` (ENG:758) pipelines steps ①–② of batch *N+1* with ③–⑤ of batch
*N* when `max_concurrent_batches > 1` (pipeline parallel, or async scheduling).

## 2. Stage-by-stage: reads / modifies / bottleneck

Each table answers the three questions for every function: **state read**,
**state modified**, **bottleneck addressed**.

### Stage 1 — How requests enter the scheduler (Q1)

| Function | Reads | Modifies | Bottleneck addressed |
|---|---|---|---|
| `EngineCoreProc.__init__` ENG:1335 | config | `input_queue`, `output_queue`; starts I/O daemon threads | **GPU idle**: ZMQ I/O and (de)serialisation run on threads (ZMQ releases the GIL), off the busy-loop thread |
| `process_input_sockets` ENG:2161 | ZMQ DEALER sockets | `input_queue` (ABORTs also go to `aborts_queue`) | GPU idle (same) |
| `EngineCore.preprocess_add_request` ENG:1258 | `request_block_hasher`, watermark config | builds internal `Request` incl. prefix-cache block hashes; kicks off grammar compile | **GPU idle + recompute**: hashing/grammar compile overlap the forward pass; hashes feed prefix-cache lookup later |
| `_process_input_queue` ENG:1820 | `input_queue`, `has_work()` | dispatches into the scheduler; blocks when idle | avoids busy-spinning; batches intake before the next `schedule()` |
| `_handle_client_request` ENG:1957 | request type | calls `add_request` / `abort_requests` | — |
| `EngineCore.add_request` ENG:564 | request | → `scheduler.add_request` | — |
| `Scheduler.add_request` SCH:3065 | `requests` | `requests[id]`, one waiting container via `_enqueue_waiting_request` (SCH:2776) | **Head-of-line blocking**: requests blocked on grammar/remote KV go to `deferred_waiting`; KV holders to `kv_holding_waiting` (drained first) |

### Stage 2 — How running requests receive a token budget (Q2)

There is **no prefill/decode phase** in the scheduler (SCH:718-727). Each request
has `num_computed_tokens` and `num_tokens_with_spec`; the scheduler assigns tokens
so the first catches up with the second. A decode step is a distance of 1 (+spec
tokens); a prefill is a large distance. Chunked prefill, prefix caching and
speculative decoding all fall out of this one model.

| Step (in `Scheduler.schedule`, SCH:716) | Reads | Modifies | Bottleneck addressed |
|---|---|---|---|
| Budgets (SCH:735,738): `token_budget = max_num_scheduled_tokens`, `input_budget = max_num_batched_tokens` | `SchedulerConfig` | locals | bounds per-step compute → **inter-token latency** |
| `long_prefill_token_threshold` (SCH:775-785; adaptive floor `input_budget // num_eligible_reqs`) | request counts | local | **head-of-line blocking** by one long prefill (only when chunked prefill on and >1 eligible request) |
| `defer_prefills` (SCH:766, 796-807) | `prefill_capacity_bound`, step cadence | local | **prefill/decode interference** (DP prefill balancing) |
| Running loop `while req_index < len(running) and token_budget > 0` (SCH:811) | `running` (FCFS order) | — | running requests are served **before** any waiting request |
| `num_new_tokens = num_tokens_with_spec + num_output_placeholders − num_computed_tokens` (SCH:855-859), then `min(…, long_prefill_token_threshold, token_budget, input_budget − draft_slots, max_model_len − num_computed − 1)` (SCH:860-873) | request counters | local | one formula covers decode (=1), chunked prefill, spec decode |
| `num_new_tokens == 0` → `continue`, not `break` (SCH:924-925) | — | — | deliberately **not** strict FCFS, so a request with nothing to do does not stall those behind it |
| On success (SCH:1013-1019) | — | `scheduled_running_reqs`, `req_to_new_blocks`, `num_scheduled_tokens`, `token_budget −= n`, `input_budget −= n + draft_slots` | budget is charged **only after** KV allocation succeeded |

### Stage 3 — How waiting requests are admitted (Q3)

| Step | Reads | Modifies | Bottleneck addressed |
|---|---|---|---|
| Guard `if not preempted_reqs and not PAUSED_ALL` (SCH:1066) | `preempted_reqs` | — | **no admission in a step that had to preempt**: KV is exhausted, admitting would only thrash |
| `_reorder_waiting_by_cached_prefix` SCH:2790 (if `cache_aware_window`) | `kv_cache_manager.get_num_cached_tokens`, `usage` | reorders head window of `waiting` | **recompute**: cache-hot requests first |
| Loop `while True` SCH:1091; `loads_only` when `len(running) ≥ max_num_active_reqs` or `token_budget ≤ 0` (SCH:1095) | `running`, budgets | picks queue: `kv_holding_waiting` first, else `waiting`/`async_kv_waiting` | `max_num_seqs` is enforced **here** |
| `_handle_blocked_waiting_request` SCH:3668 → `skip_request` SCH:1076 | request status | moves request to `deferred_waiting` + a `step_skipped_*` deque, re-queued after the loop | **HOL blocking** on grammar compile / remote KV |
| `_get_local_prefix_cache_hit` SCH:681 → `KVCacheManager.get_computed_blocks` KVM:288 → `coordinator.find_longest_cache_hit` | `request.block_hashes`, block hash map | nothing (no ref-count change; lookup only) | **recompute** of shared prefixes |
| `num_new_tokens = request.num_tokens − num_computed_tokens` SCH:1356 (`num_tokens`, not `num_prompt_tokens`, so resumed requests replay their outputs) | | local | — |
| `not enable_chunked_prefill and num_new_tokens > budget` → `break` SCH:1387-1394 | `enable_chunked_prefill` | — | the remaining **HOL-blocking** case: without chunking, a prompt larger than the leftover budget stops admission |
| `allocate_slots(...)` SCH:1497; `None` → `break` SCH:1516-1523 | free blocks | block manager state on success | see Stage 4/5 |
| Admit SCH:1609-1637 | | `waiting.pop_request`, `running.append`, `scheduled_new_reqs`, `status = RUNNING`, `request.num_computed_tokens = num_computed_tokens`, budgets | a resumed (previously preempted) request is re-sent as a *new* request with full `prefill_token_ids` |

### Stage 4 — How KV blocks are allocated (Q4)

`KVCacheManager.allocate_slots` (KVM:401) is the **coupling point between the
token budget and KV memory**: the scheduler asks for KV for exactly the
`num_new_tokens` it is about to schedule *this step* — not the whole prompt and
not the worst-case output. For a *new* request there is one extra gate: with the
default `scheduler_reserve_full_isl = True` it first **checks** that the whole input
sequence would fit (KVM:552-569), then allocates only the first chunk. That is a
check, not a reservation — nothing is held for the blocks of later chunks.

| Function | Reads | Modifies | Bottleneck addressed |
|---|---|---|---|
| `KVCacheManager.__init__` KVM:132 | config | builds `coordinator`, `block_pool`, `watermark_blocks = int(watermark × num_blocks)`, `empty_kv_cache_blocks` | KV fragmentation (fixed-size blocks) |
| `get_computed_blocks` KVM:288 | `request.block_hashes` | none | recompute; caps hit so the **last token is always recomputed** to get logits |
| `allocate_slots` KVM:401 (layout `comp │ new_comp │ ext_comp │ new │ lookahead`) | request, free-block count, `watermark_blocks`, `reserved_blocks` | `req_to_blocks`, block `ref_cnt`/hash, free queue | admission accuracy; **fragmentation**; preemption trigger |
| – watermark KVM:539-546 | `has_scheduled_reqs`, `request.status` | — | applies **only to WAITING/PREEMPTED** requests and only when something is already scheduled → keeps headroom so running decodes do not immediately preempt |
| – `full_sequence_must_fit` KVM:552-569 (passed from SCH:1511 as `scheduler_reserve_full_isl`, default **True**, `vllm/config/scheduler.py:202`) | whole `request.num_tokens`, free blocks, watermark | — | **over-admission / KV thrash under chunked prefill**: a new request is refused unless its *entire* input sequence would fit, even though only the first chunk is then allocated |
| – `num_tokens_need_slot = min(total_computed + num_new_tokens + lookahead, max_model_len)` KVM:571-574 | | | lookahead slots (spec decode) are allocated but never hashed |
| – shortage test `num_blocks_to_allocate + watermark > free − reserved` → `return None` KVM:605-609 | `block_pool.get_num_free_blocks()` | — | **all-or-nothing**: a failed request leaves no partial state |
| – `allocate_new_computed_blocks` KVM:617 → `touch` hit blocks | | `ref_cnt`, free queue | prefix reuse without recompute |
| – `allocate_new_blocks` KVM:625 → `STM.allocate_new_blocks` STM:375 → `BlockPool.get_new_blocks` BP:681 | | pops blocks from the free queue | O(1) allocation |
| – `cache_blocks` KVM:646 → `STM.cache_blocks` STM:482 → `BlockPool.cache_full_blocks` BP:203 | `request.block_hashes` | block hash map | prefix-cache population; only **full** blocks, only up to `request.num_tokens` (drafts never hashed) |
| `KVCacheCoordinator` KVC:89 | | owns one shared `BlockPool` + one `SingleTypeKVCacheManager` per KV group; `get_kv_cache_coordinator` KVC:990 picks NoPrefixCache / Unitary (1 group) / Hybrid (several) | hybrid models (full + sliding-window + mamba layers) |
| `BlockPool.get_new_blocks` BP:681 | free queue | `popleft_n`; **evicts** a popped block's hash lazily (`_maybe_evict_cached_block`); sets `ref_cnt = 1` | eviction happens at allocation time, not free time |
| `BlockPool.touch` BP:774 | | `ref_cnt += 1`; removes a `ref_cnt == 0` block from the free queue (O(1)) | protects hit blocks from eviction |

**Block lifetime in one paragraph.** `ref_cnt` is 1 on allocation, +1 per sharer
(`touch`), −1 per `free_blocks`. At 0 the block returns to the free queue — but a
block that still has a hash stays *cached and evictable*: it is in the free queue
(so counted by `get_num_free_blocks`, BP:903) yet still hits in the prefix cache
until `get_new_blocks` pops it. `free_blocks` (BP:798-837) puts uncached blocks at
the **front** (LIFO reuse) and cached blocks at the **tail** (FIFO ⇒ LRU), and
callers pass a request's blocks in **reverse** order (STM:596 `free`), so the
tail of a sequence — the longest hash chain, least shareable — is evicted first
and prefix heads live longest. Block id 0 is a `null_block` placeholder, never
freed or hashed.

### Stage 5 — How resource exhaustion is handled (Q5)

| Situation | What vLLM does (source) | Cost / bottleneck |
|---|---|---|
| **Running** request cannot get blocks | loop `allocate_slots`→`None`→ pick victim → `_preempt_request` → retry (SCH:929-996). Victim: FCFS `running.pop()` = **last-admitted** (SCH:986); PRIORITY `max(running, key=(priority, arrival_time))` (SCH:949), with the victim's already-scheduled work rolled back (SCH:960-984) | **Recompute** preemption: no swap |
| `_preempt_request` SCH:1853 | frees blocks (`_free_request_blocks`), frees encoder cache, `status = PREEMPTED`, **`num_computed_tokens = 0`**, `num_preemptions += 1`, marks in-flight output stale, `waiting.prepend_request` (front of FCFS queue; heap push for PRIORITY), records id in `reset_preempted_req_ids` so workers drop their state | prompt **and** generated tokens are re-prefilled on resume (hashed blocks may survive in the cache and shortcut that) |
| Request preempts **itself** (no other victim) | `break` out of retry loop (SCH:994-996); under PRIORITY `continue` to later requests (SCH:998-1008), under FCFS `break` the whole running loop (SCH:1010) | |
| **Waiting** request cannot get blocks | `allocate_slots`→`None` → `break` (SCH:1516-1523). **No preemption to make room for a waiting request.** It stays at the queue head | strict HOL for waiting side |
| Anything preempted this step | waiting admission is skipped for the whole step (SCH:1066) | avoids admit/evict thrash |
| Async-scheduling hazard | `_request_blocks_can_be_freed` SCH:3217 / `_free_request_blocks` SCH:3224: if a GPU step may still write the blocks, free is **deferred** via `pop_blocks_for_free` + `deferred_frees` | correctness with CPU-ahead-of-GPU scheduling |
| Headroom | `watermark_blocks` (KVM:539-546) and `reserved_blocks` | fewer preemptions |

### Stage 6 — How scheduled work reaches the GPU workers (Q6)

| Function | Reads | Modifies | Bottleneck addressed |
|---|---|---|---|
| `Scheduler._update_after_schedule` SCH:1899 | `SchedulerOutput` | `num_computed_tokens += scheduled` **before** the step runs; `num_in_flight_tokens`; clears `finished_req_ids` | **GPU idle**: lets the next `schedule()` run before this step's tokens exist |
| `AsyncScheduler._update_after_schedule` async_scheduler.py:18 | | `num_output_placeholders`, `spec_token_ids = [-1]…`, `next_decode_eligible_step` | scheduling optimistically ahead of sampling |
| `SchedulerOutput` output.py:220 / `NewRequestData` :41 / `CachedRequestData` :139 | | | **IPC volume**: full request data is sent **once** (`NewRequestData`); afterwards only diffs (`CachedRequestData`: new block ids, `num_computed_tokens`) |
| `Executor.get_class` abstract.py:65 | `distributed_executor_backend` | | picks `MultiprocExecutor` ("mp"), `UniProcExecutor`, Ray, … |
| `MultiprocExecutor._init_executor` MPX:156 | world size | `rpc_broadcast_mq` (shared-memory ring buffer), `response_mqs`, worker processes | **IPC**: one shm write fans out to all ranks |
| `MultiprocExecutor.execute_model` MPX:430 / `sample_tokens` MPX:443 → `collective_rpc` MPX:471 | | `enqueue((method, args, kwargs, output_rank))`; returns a `FutureWrapper` (MPX:92) when `non_block` | non-blocking submit enables batch-queue overlap; **only `output_rank` replies** (`_get_output_rank` MPX:683) |
| `WorkerProc.worker_busy_loop` MPX:1188 → `_execute_worker_rpc` MPX:1197 | `rpc_broadcast_mq` | calls `getattr(worker, method)` | — |
| `Worker.execute_model` GW:1449 / `sample_tokens` GW:1442 | | → model runner | pipeline-parallel send/recv overlap |
| `GPUModelRunner.add_requests` GMR:1284 / `update_requests` GMR:1340 | `NewRequestData`, `CachedRequestData` | persistent `RequestState` (states.py:9), `BlockTables.append_block_ids` (block_table.py:125: appends only the new block ids), staged writes | **Persistent batch**: the batch is not rebuilt each step; **many small H2D copies → one** (`StagedWriteTensor`) |
| `prepare_inputs` GMR:1458 / `prepare_attn` GMR:1665 | `RequestState`, `BlockTables` | `input_ids`, `positions`, `seq_lens`, `slot_mappings` (`BlockTables.compute_slot_mappings` block_table.py:229) | **CPU launch overhead**: input prep as Triton kernels; previous step's sampled token consumed on the GPU |
| `GPUModelRunner.execute_model` GMR:1826 | | KV cache (by attention kernels), `ExecuteModelState` | CUDA graphs (FULL/PIECEWISE) cut launch overhead; returns `None` — sampling is a separate RPC |
| `GPUModelRunner.sample_tokens` GMR:2216 → `AsyncOutput` (async_utils.py:117) | | `req_states` (via `postprocess_sampled` GMR:1782: appends token, `num_computed_tokens += query_len − num_rejected`) | D2H copy on a side stream overlaps the next step |

### Stage 7 — How model outputs update request state (Q7)

`Scheduler.update_from_output` (SCH:2301), per scheduled request:

1. Drain in-flight/stale accounting (`num_in_flight_tokens`, `num_stale_output_tokens`); skip requests that finished or aborted mid-flight.
2. `generated_token_ids = sampled_token_ids[req_index]`; if spec decode: `num_rejected = num_draft − num_accepted`, then `num_computed_tokens −= num_rejected` and `num_output_placeholders −= num_rejected` (roll back the optimistic advance from `_update_after_schedule`).
3. `_update_request_with_output` SCH:2879 → `request.append_output_token_ids`, `check_stop` (`sched/utils.py:98`: EOS / `stop_token_ids` → `FINISHED_STOPPED`; `max_tokens` or `max_model_len` → `FINISHED_LENGTH_CAPPED`; honours `min_tokens`).
4. `kv_cache_manager.cache_blocks` publishes hashes for blocks that became full (SCH:~2468).
5. Build `EngineCoreOutput` per request, grouped by `client_index`.

Reads: `SchedulerOutput`, `ModelRunnerOutput` (`outputs.py:267`). Modifies: request token
state, `running`, waiting queues, `finished_req_ids`, KV manager hashes. Bottleneck: this
O(requests) CPU loop is a known potential CPU bottleneck at ≥1K requests (comment at SCH:2348-2350).

### Stage 8 — How completed requests free resources (Q8)

```text
update_from_output ─ stopped ─► _free_request SCH:3170
                                 ├─ connector hooks may delay the block free
                                 ├─ encoder_cache_manager.free(request)
                                 ├─ finished_req_ids.add(id)      ← sent to workers next step
                                 └─ _free_blocks SCH:3204 → _free_request_blocks SCH:3224
                                        └─ KVCacheManager.free KVM:650
                                              └─ KVCacheCoordinator.free KVC:404
                                                    └─ SingleTypeKVCacheManager.free STM:596
                                                          └─ BlockPool.free_blocks(reversed(blocks)) BP:798
                                                  del self.requests[id]
next schedule(): SchedulerOutput.finished_req_ids → GPUModelRunner.finish_requests GMR:1255
                 → _remove_request: frees the persistent RequestState slot
```

Aborts take the same road via `EngineCore.abort_requests` ENG:610 →
`Scheduler.finish_requests` SCH:3100 → `_free_request`.

### Stage 0 — Startup: how `gpu_memory_utilization` becomes the block pool

1. `Worker.init_device` (GW:451) → `request_memory` (`vllm/v1/worker/utils.py:734`): `requested = ceil(total_gpu_memory × gpu_memory_utilization)`.
2. `Worker.determine_available_memory` (GW:673) runs a profile forward at `max_num_batched_tokens` (`GPUModelRunner.profile_run` GMR:1070): `available_kv_cache_memory_bytes = requested − non_kv_cache_memory − cudagraph_estimate` (GW:776-780).
3. `EngineCore._initialize_kv_caches` (ENG:273) collects it, `get_kv_cache_configs` (`vllm/v1/core/kv_cache_planning.py:1565`) divides by bytes-per-block (`num_blocks = available // bytes_per_block`, minimum across ranks) and stores `cache_config.num_gpu_blocks`.
4. Scheduler side: `BlockPool(num_gpu_blocks)` in `KVCacheCoordinator.__init__` (KVC:94). Worker side: `GPUModelRunner.initialize_kv_cache` (GMR:644) allocates the KV tensors.

Consequence for Session 3: raising `--gpu-memory-utilization` buys **more blocks, not
more speed** — it changes preemption frequency only once the pool is the binding constraint.
`--max-num-batched-tokens` also changes memory: the profile run sizes activations at that
value, so a larger budget shrinks the KV pool.

## 3. Where each performance bottleneck is attacked

| Bottleneck | Mechanism (symbol) |
|---|---|
| GPU idle waiting on CPU | I/O threads (ENG:2161, 2270); `preprocess_add_request` on the input thread; `execute_model(non_block=True)` before bitmask (ENG:724); `step_with_batch_queue` ENG:758; `AsyncScheduler` placeholders; `_update_after_schedule` pre-advance SCH:1899; `AsyncOutput` D2H overlap |
| Head-of-line blocking | `long_prefill_token_threshold`; chunked prefill; `skip_request` + `deferred_waiting`; `kv_holding_waiting` first; `continue` for zero-token running requests (SCH:924). Still present: non-chunked `break` (SCH:1394) and allocation-failure `break` (SCH:1523) |
| Prefill/decode interference | unified `num_computed_tokens` model; running before waiting; `defer_prefills` cadence |
| KV fragmentation / waste | fixed-size blocks; allocation for scheduled tokens only; all-or-nothing `allocate_slots`; `null_block`; sliding-window block release (`remove_skipped_blocks`) |
| Recompute | prefix cache (`get_computed_blocks`); LRU/reverse-order free; `_reorder_waiting_by_cached_prefix`; preemption keeps hashed blocks evictable-but-hittable |
| IPC cost | shm ring buffer broadcast (MPX:202); one reply rank; `NewRequestData` once then `CachedRequestData` diffs; staged H2D writes |
| CPU launch overhead | CUDA graphs (`capture_model` GMR:1141); Triton input-prep kernels; persistent `InputBuffers` |

## 4. Reproduce / verify

```bash
./fetch_vllm_source.sh                                  # clone v0.34.0 into vllm_src/
.venv/bin/python verify_trace.py --src vllm_src         # every cited symbol + def line still exists
.venv/bin/python verify_trace.py --src vllm_src --write-lines   # refresh lines after a version bump
```
