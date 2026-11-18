# Knowledge checkpoint — answers

vLLM `v0.34.0` @ `62d4bc3`. Citations use the abbreviations in
[SOURCE_TRACE.md](SOURCE_TRACE.md); that file has the full call-flow and
reads/modifies/bottleneck tables behind each answer.

## 1. What is the responsibility of the vLLM engine core?

`EngineCore` (ENG:129) / `EngineCoreProc` (ENG:1328) is the **CPU-side control loop** that
sits between the API frontend and the GPU workers. It owns the `Scheduler` and its
`KVCacheManager`, and each iteration it:

1. takes requests off an input queue (filled by a ZMQ thread that also does request
   preprocessing and prefix-hash computation, ENG:2161, 1258),
2. asks the scheduler *what to run this step* (`schedule()`, SCH:716),
3. ships that decision to the workers via the executor and waits for sampled tokens
   (`step`, ENG:713; or the pipelined `step_with_batch_queue`, ENG:758),
4. feeds the results back into the scheduler (`update_from_output`, SCH:2301) and
   puts the outputs on a queue drained by an output thread (ENG:2270).

It also sizes the KV pool at startup (`_initialize_kv_caches`, ENG:273). It does **no**
tensor math and holds **no** GPU memory itself; keeping the scheduler in its own process
and the I/O on threads is what stops Python scheduling/serialisation from starving the GPU.

## 2. How does vLLM decide which requests execute next?

There is no prefill phase or decode phase — each request has `num_computed_tokens` and
`num_tokens_with_spec`, and every step the scheduler hands out tokens so the first
catches up with the second (SCH:718-727). Within a step, `schedule()` makes one pass with
two budgets (`token_budget`, `input_budget`):

1. **Running requests first**, in list order (SCH:811): `num_new_tokens` is the remaining distance,
   clamped by `long_prefill_token_threshold`, the budgets and `max_model_len` (SCH:855-873).
   A request with 0 tokens to do is skipped with `continue` (SCH:924), deliberately not strict FCFS.
2. **Then waiting requests**, only if nothing was preempted this step (SCH:1066), taken from
   the KV-holding queue first and then the policy queue (FCFS deque or priority heap,
   `request_queue.py`). Admission requires a free slot (`max_num_seqs`, SCH:1095), budget left,
   the prompt not blocked on a grammar/remote KV (SCH:3668), and a successful `allocate_slots`.

Order is therefore: *requests already running → KV holders → policy order of the waiting
queue*, with a hard stop at the first waiting request that cannot get tokens or blocks.

## 3. How does vLLM manage KV-cache blocks?

As fixed-size **paged blocks** in one `BlockPool` (BP:107) per engine, shared by all requests:

- **Allocation**: `KVCacheManager.allocate_slots` (KVM:401) allocates blocks for exactly the
  tokens being scheduled now (+ lookahead for spec decode); all-or-nothing. A new request is
  first *checked* against its whole prompt (`scheduler_reserve_full_isl`, KVM:552-569) but only
  the first chunk is allocated.
- **Sharing**: full blocks are hashed (chained hashes, so each fingerprints its whole prefix,
  `kv_cache_utils.py:703`) into a map; a prefix hit `touch`es (ref-count++) the existing blocks
  instead of recomputing (KVM:288, BP:774).
- **Free list**: a doubly-linked `FreeKVCacheBlockQueue`. Freed blocks with `ref_cnt == 0` that
  are still hashed stay *cached-but-evictable*; eviction is lazy, at allocation time
  (`get_new_blocks`, BP:681). Uncached blocks are reused first (LIFO), cached ones LRU, and a
  request's blocks are freed in reverse so tails evict before prefixes (BP:826-837, STM:596).
- **Structure**: `KVCacheCoordinator` (KVC:89) fronts one `SingleTypeKVCacheManager` per KV
  group (full attention, sliding window, mamba, …) over the shared pool.
- **Sizing**: the pool size comes from the worker's memory profile —
  `gpu_memory_utilization × total − weights/activations` (GW:673), not from a config number.
- The scheduler-side manager is pure bookkeeping; the worker mirrors each request's block ids
  (`BlockTables`, `worker/gpu/block_table.py:17`), updated with new-block diffs only.

## 4. What happens when a request cannot acquire enough KV memory?

Depends which side of the scheduler it is on:

- **A running request that needs one more block** (`allocate_slots → None`, SCH:930):
  vLLM preempts a victim and retries (SCH:929-996). Victim: the **last** request in `running`
  (FCFS, SCH:986) or the max `(priority, arrival_time)` (PRIORITY, SCH:949). `_preempt_request`
  (SCH:1853) frees its blocks, sets `num_computed_tokens = 0`, marks it `PREEMPTED`, and puts
  it back at the **front** of the waiting queue. It will be **recomputed** (re-prefilled with
  its prompt *and* generated tokens) when readmitted — hashed blocks that survive in the cache
  shorten that. If the request ends up preempting itself, scheduling of the running loop stops
  (`break`, SCH:1010; `continue` under PRIORITY, SCH:1008).
- **A waiting request** that cannot get blocks: the admission loop `break`s (SCH:1516-1523) and
  the request stays at the head of the queue. vLLM **never evicts a running request to admit a
  waiting one.**
- **Afterwards**: a step that preempted anything admits no waiting requests (SCH:1066).
- **Mitigations before it gets that far**: the whole-prompt admission check and the
  `watermark` headroom for waiting/preempted requests (KVM:539-546).

## 5. What does the GPU worker do?

A `Worker` (GW:216) per GPU rank, in its own process, driven by an RPC loop
(`WorkerProc.worker_busy_loop`, MPX:1188). It:

- at startup: claims its memory budget (`request_memory`), loads the model, profiles
  `determine_available_memory` (GW:673) so the engine can size the block pool, allocates the
  KV tensors (`initialize_kv_cache`, GMR:644), warms up and captures CUDA graphs (GW:1017, GMR:1141);
- every step: applies the scheduler's diff to a **persistent batch** (`add_requests` GMR:1284,
  `update_requests` GMR:1340, `finish_requests` GMR:1255), builds inputs/positions/slot-mappings
  with GPU kernels (`prepare_inputs` GMR:1458, `prepare_attn` GMR:1665), runs the forward
  (CUDA graph or eager, GMR:1826), then in a *separate* RPC samples (`sample_tokens`, GMR:2216),
  starts an async device→host copy, and returns a `ModelRunnerOutput` only from the output rank.

The worker never decides *what* to run — it executes the `SchedulerOutput` it is given
and keeps its own mirror of per-request state and block tables.

## 6. How do scheduling parameters influence latency and throughput?

What the **source** establishes (mechanism):

| Parameter | Mechanism | Expected direction |
|---|---|---|
| `--max-num-batched-tokens` | per-step token budget (SCH:735, 738); with chunked prefill, caps how much prefill rides along with decodes; also sizes the startup activation profile so a larger value *shrinks* the KV pool | larger → higher throughput and TTFT for long prompts, but higher inter-token latency (decodes share a bigger step) |
| `--max-num-seqs` | admission cap (SCH:1095); sizes persistent worker buffers | larger → more concurrency/throughput until the KV pool or token budget binds; only matters when offered load exceeds it |
| `--gpu-memory-utilization` | sets the KV pool size (GW:776) | larger → more blocks → fewer preemptions; no effect on speed until the pool is the binding constraint |
| request rate | offered load vs. the limits above | past saturation, queueing makes TTFT grow without bound while throughput plateaus |
| prompt length | prefill tokens per request; blocks per request | longer prompts → higher TTFT, more KV per request, more preemption risk |
| chunked prefill / `long_prefill_token_threshold` | bounds prefill's share of a step | smooths ITL at some cost to TTFT |

**Measured** (one NVIDIA A40, vLLM 0.34.0, Qwen2.5-0.5B, 3 repeats, fresh server per run, 0 failed requests; saturated numbers are from the
1000-prompt burst; full tables, noise estimates and caveats in [RESULTS.md](RESULTS.md)). Most "expected directions" above held, with one
exception, and these sizes:

- `--max-num-batched-tokens` is the strongest scheduling lever for latency: under saturated load, 512 vs 8192 gives 4.6x lower p99 ITL
  (24 vs 109 ms) for 46% less throughput and 2.0x higher p99 TTFT. Raising it to 16384 adds about 1% throughput (noise) and worsens p99 ITL by 64%.
- `--max-num-seqs` scales throughput up to 256 (1,124 / 3,513 / 6,926 / 9,445 tok/s for 4 / 16 / 64 / 256) but **1024 is 11% slower (8,399) with
  4x worse p99 ITL (442 vs 109 ms)**. The "larger is better until something binds" prediction is therefore wrong above 256 here; the cause was
  not fully explained: a CUDA-graph capture-range effect was tested and refuted (Day 6 report 9.3), and traces show prefill-carrying steps cost 42-135 ms vs 7 ms for pure decode, which explains the ITL tail but not the throughput loss. A first pass with only 100 prompts missed this because it
  could not use more than 100 sequences. At the brief 2 req/s workload no latency effect is visible.
- `--gpu-memory-utilization` sets the KV pool as predicted (1.75M to 3.50M tokens for 0.5 to 0.95) and changed no speed (spread 1.4%), because the
  pool was 3x to 19x larger than the computed peak demand. **No preemption occurred in any of 168 runs, so Q4's behaviour under real KV exhaustion remains unmeasured.**
- Throughput tracks the arrival rate until saturation; longer prompts raise TTFT and cut throughput (16,343 to 1,231 tok/s from 128 to 3,584 tokens).

Mechanism (Day 6 traces): a step that carries prefill costs 93-132 ms vs 7-26 ms for pure decode, and each decode riding in such a step costs about 4x more; this also explains why p50 ITL at `max_num_batched_tokens=1024` exceeds that at 512.
Not established: the cause of the `max_num_seqs=1024` throughput loss, the 234 ms p50 ITL at `input_len=3584`,
anything for models larger than 0.5B, and the saturation point between 16 req/s and unthrottled.
