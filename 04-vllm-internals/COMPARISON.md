# Scheduler comparison: `03-batching-simulator` vs vLLM v0.34.0

Compares `../03-batching-simulator/` (a discrete-event *simulator* of a scheduler) with
vLLM's real V1 scheduler, pinned to commit `62d4bc3`. vLLM citations use the
abbreviations from [SOURCE_TRACE.md](SOURCE_TRACE.md) (SCH = `core/sched/scheduler.py`,
KVM = `core/kv_cache_manager.py`, …). `03-batching-simulator` citations are `file:line` inside
`03-batching-simulator/`.

Everything on the vLLM side is read from source, not from memory or docs. Where this
file says what *would happen* (a prediction), it says so.

## 1. Mechanism table

### 1a. Implemented — same mechanism

| Mechanism | `03-batching-simulator` | vLLM | Notes |
|---|---|---|---|
| Iteration-level (continuous) batching | `ContinuousScheduler.plan` scheduler.py:174, called once per `Simulator.run` iteration | `Scheduler.schedule` SCH:716, called once per `EngineCore.step` ENG:713 | Both re-decide the batch every forward pass |
| Per-iteration token budget; a decoder costs 1 token | `_allocate` :197 — `budget = token_budget − len(decoders)` | `token_budget` SCH:735; decode `num_new_tokens = 1` via SCH:855 | `03-batching-simulator` `token_budget` ≙ vLLM `--max-num-batched-tokens` |
| Cap on concurrent sequences | `max_running_seqs`, checked in `_can_admit` :229 | `--max-num-seqs` → `max_num_running_reqs`/`max_num_active_reqs`; checked SCH:1095 | |
| Chunked prefill | `chunked_prefill` flag; `chunk = min(remaining, budget)` :204 | `enable_chunked_prefill` (default **True**, `config/scheduler.py:146`); `min(num_new_tokens, request_token_budget)` SCH:1396 | |
| …and its absence blocks the queue | `_admit`: `left < needs → return` :221 | `not enable_chunked_prefill and n > budget → break` SCH:1387-1394 | Both reproduce head-of-line blocking by an oversized prompt |
| Paged KV, block-granular, all-or-nothing | `KVBlockManager.try_ensure` kv_blocks.py:37 | `allocate_slots` returns `None` on shortage, no partial state KVM:605-609 | |
| Decode KV grows on demand | `_ensure_decode_kv` :184 (+1 token per decoder) | `allocate_slots(request, num_new_tokens)` per running request SCH:930 | |
| Recompute-style preemption | `EngineState.preempt` :144 → `Request.reset_for_recompute` | `_preempt_request` SCH:1853 (`num_computed_tokens = 0`, blocks freed, back to waiting) | Neither swaps KV to CPU |
| Preemption victim = lowest priority, latest arrival (PRIORITY) | `max(state.running, key=rank)` :192 | `max(running, key=(priority, arrival_time))` SCH:949 | |
| FCFS and priority policies | `rank_key` :103 | `SchedulingPolicy`, `FCFSRequestQueue`, `PriorityRequestQueue` (`request_queue.py:13,75,131`) | |
| Free KV when a request finishes | `EngineState.finish` :153 → `kv.free` | `_free_request` SCH:3170 → … → `BlockPool.free_blocks` BP:798 | |
| Preempted request resumes with its generated tokens | `prefill_goal = prompt + generated` | `request.num_tokens` (not `num_prompt_tokens`) SCH:1353-1356 | |

### 1b. Simplified

| Mechanism | `03-batching-simulator` (simplification) | vLLM (reality) |
|---|---|---|
| KV accounting | **Counts only**: `_owned: dict[req_id, int]`, `used_blocks` | `BlockPool` with `KVCacheBlock` objects, `ref_cnt`, a doubly-linked `FreeKVCacheBlockQueue`, a hash→block map, a `null_block` (BP:107, kv_cache_utils.py:166,254). (`02-kv-cache-memory`'s `PagedKVAllocator` has real ids/tables but `03-batching-simulator` does not use it.) |
| Prefill vs decode | Explicit partition: `is_prefilling = kv_tokens < prefill_goal`; `_allocate` handles the two sets separately | **No phases**: one `num_computed_tokens` vs `num_tokens_with_spec` distance per request (SCH:718-727) |
| Time | `CostModel.iteration_time` — a formula (`0.3 + max(0.04·tokens, 7.0 + 0.0001·ctx)`) in simulated ms | A real forward pass; plus scheduling, IPC and sampling that vary with batch shape |
| Scheduling latency | Zero; scheduler runs "between" iterations | Real CPU cost; hidden by async scheduling/batch queue (ENG:758) |
| Output length | Fixed `max_output_tokens`, no EOS | `check_stop` (`sched/utils.py:98`): EOS, stop ids, `max_tokens`, `max_model_len`, repetition |
| Waiting queue | Python list, `min(waiting, key=…)` per admission (O(n)) | `deque` (FCFS) / `heapq` (PRIORITY), O(1)/O(log n) |
| Block size | 16 tokens (default) | `CacheConfig.DEFAULT_BLOCK_SIZE = 64` (`config/cache.py:185`) |
| Process model | One Python loop | `EngineCoreProc` busy loop + 2 I/O threads + worker processes over a shm ring buffer (ENG:1794, MPX:156) |
| Memory sizing | `num_kv_blocks` is an input | Derived: `gpu_memory_utilization × total − non-KV (profile run)` ÷ bytes-per-block (GW:673, `kv_cache_planning.py:1565`) |

### 1c. vLLM does it differently (same goal, different design)

| Aspect | `03-batching-simulator` | vLLM | Why it matters |
|---|---|---|---|
| **Order of work within a step** | `_ensure_decode_kv` → `_admit` → `_allocate` (:174-178): admission is decided **before** the budget is spent | One pass: running requests first, then waiting; for each request `n = f(budget)` → `allocate_slots(n)` → charge budget (SCH:809-1023, 1066-1640) | See §2 — this is the budget↔KV coupling |
| **KV obtained at admission** | `on_demand`: `try_ensure(prefill_goal)` — the **whole prompt is reserved** up front (`_reserve_tokens` :181, `admit` :133), even with chunked prefill | **Checks** the whole prompt fits (`scheduler_reserve_full_isl`, default True, KVM:552-569) but **allocates only the first chunk** (`num_new_tokens`) | `03-batching-simulator` holds KV for tokens not yet computed; vLLM does not |
| **KV peak reservation** | `full_reserve`: reserve `prompt + max_output − 1` → preemption impossible | No equivalent (it never reserves for future *output*); the closest knobs are `watermark` (KVM:539-546) and `scheduler_reserve_full_isl` | |
| **Admission in a step that preempted** | `plan()` calls `_admit` right after `_ensure_decode_kv`, so a request can be admitted into the blocks a victim just freed | Skipped entirely: `if not preempted_reqs …` SCH:1066 | Prevents admit→evict→admit thrash |
| **Can a waiting request evict a running one?** | Optional: `preempt_for_priority` → `_preempt_lower_priority` :234 | **Never.** On `allocate_slots → None` the waiting loop just `break`s (SCH:1516-1523) | Only KV *growth* of running requests triggers preemption in vLLM |
| **FCFS victim** | Latest **arrival** (rank key) | `running.pop()` = last in the running list = latest **admitted** (SCH:986) | Differs when a preempted request was re-admitted |
| **Where preempted requests go** | Appended to `waiting`; ordering restored by `min(key)` | `waiting.prepend_request` (front of FCFS deque; heap push for PRIORITY) SCH:1896 | Equivalent for FCFS, by different means |
| **Headroom** | None | `watermark_blocks` applied only to WAITING/PREEMPTED requests and only when something is already scheduled (KVM:539-546) | Keeps running decodes from immediately preempting |
| **Zero-work requests** | Skips a prefill whose chunk exceeds the budget (`continue` :205-206) | `continue` instead of `break` "does not strictly follow FCFS" SCH:921-925 | Similar spirit |
| **Static batching** | Implemented as a baseline (`StaticScheduler` :251) | Does not exist — continuous batching only | `03-batching-simulator`'s baseline has no vLLM counterpart |
| **Request ids / state** | `int` ids, one mutable `Request` | `str` ids; the scheduler's `Request` is mirrored in the worker as `RequestState` (`worker/gpu/states.py:9`) fed by `NewRequestData` once, then `CachedRequestData` diffs | IPC volume |

### 1d. Not implemented in `03-batching-simulator`

| vLLM mechanism | Where | What it buys |
|---|---|---|
| **Prefix caching**: chained block hashes, `get_computed_blocks`, `cache_full_blocks`, LRU eviction | KVM:288; BP:203; `kv_cache_utils.py:703,949` | Skips recomputing shared prefixes |
| **Ref-counted shared blocks** (`touch`, `free_blocks`), reverse-order free so sequence tails evict first | BP:774, 798 | Safe sharing + sensible eviction order |
| **Hybrid KV groups** (full + sliding-window + mamba …) | `KVCacheCoordinator` KVC:89; `HybridKVCacheCoordinator` KVC:617 | Memory for models with mixed attention types |
| **Speculative decoding** (lookahead slots, optimistic advance, rejection rollback) | KVM:571-574; SCH `update_from_output` :2301 | More tokens per forward |
| **Async scheduling / batch queue / PP** | `step_with_batch_queue` ENG:758; `AsyncScheduler` | Hides CPU scheduling behind GPU time |
| **Unified waiting-side containers** (`kv_holding_waiting`, `async_kv_waiting`, `deferred_waiting`) and blocked-status handling (grammar compile, remote KV) | SCH:2776, 3668 | No HOL blocking on external waits |
| **`long_prefill_token_threshold`** (adaptive) | SCH:775-785 | Bounds how much of the budget one long prefill takes |
| **Cache-aware waiting reorder** | SCH:2790 | Cache-hot requests first |
| **Deferred block free** for in-flight GPU steps | SCH:3217-3260 | Correctness when the CPU runs ahead of the GPU |
| Encoder (multimodal) budget/cache, LoRA slot limits, structured-output grammar bitmask, KV connectors (P/D disaggregation), DP prefill balancing, pause states, streaming sessions | SCH (various) | Feature coverage |
| **The entire worker side**: persistent batch, staged H2D writes, slot-mapping kernels, CUDA graphs, async D2H | GMR:1284-1782, `block_table.py:229` | Making the GPU, not the CPU, the bottleneck |
| Abort/cancel | ENG:610 → SCH:3100 | |

**`03-batching-simulator` has, vLLM does not:** static batching, `preempt_for_priority`, `full_reserve` KV mode, the cost model/simulator itself.

## 2. Token budget ↔ KV-block coupling

This is the point where the two designs differ most.

**`03-batching-simulator` — decide in separate phases, tie them together by over-reserving:**

```text
_ensure_decode_kv   KV(+1 token) for each decoder        (budget not consulted)
_admit              1) budget check:  left >= needs ?     (budget not spent)
                    2) KV check:      can_fit(prefill_goal)
                    3) KV acquire:    try_ensure(prefill_goal)   ← whole prompt, now
_allocate           spend budget: decoders 1 each, then prefill chunks   (KV not consulted)
```

The invariant "KV owned ≥ tokens computed" holds because KV for the *entire* prompt is
acquired at admission, before any chunk is scheduled. The budget can then be spent without
ever touching KV again, but the cost is that a request admitted for a 4 000-token prompt
holds 4 000 tokens of blocks while only its first 512-token chunk has run.

**vLLM — one atomic step per request, budget and KV charged together:**

```text
for request in running (then waiting):
    n = min(tokens_still_needed, [long_prefill_threshold], token_budget, input_budget, max_model_len…)
    new_blocks = allocate_slots(request, n, …)       # KV for exactly n tokens (+ lookahead)
    if new_blocks is None:  preempt (running) / break (waiting)    # neither budget nor KV charged
    else:                   token_budget -= n ; request holds KV for n more tokens
```

Consequences read off the source:

1. **Failure is clean.** A failed `allocate_slots` charges neither resource; the budget is
   not burned on a request that cannot run (SCH:998-1010, 1516-1523).
2. **KV tracks computation, not intent.** Under chunked prefill a long prompt acquires blocks
   chunk by chunk, so more requests can be *running* in the same KV pool than under `03-batching-simulator`'s
   up-front reservation.
3. **…but that creates preemption risk, and vLLM spends three mechanisms to manage it:**
   the whole-prompt **check** at admission (`scheduler_reserve_full_isl`, KVM:552-569), the
   **watermark** headroom for waiting/preempted requests (KVM:539-546), and the rule that
   a step that preempted admits nobody (SCH:1066). `03-batching-simulator` has none of these, so its
   `on_demand` mode pays for the same risk with early, coarse reservation instead.
4. **The two failure paths are asymmetric.** A *running* request that cannot grow evicts the
   latest admitted request (recompute cost); a *waiting* request that cannot be admitted
   simply stops the loop. `03-batching-simulator`'s default is the same except under `preempt_for_priority`.
5. **Budget is per-step, KV is per-lifetime.** `token_budget` is rebuilt every `schedule()`;
   KV is held until the request finishes or is preempted. So a small `max_num_batched_tokens`
   limits per-step latency but does *not* limit KV pressure; `max_num_seqs` and the pool size do.
   Raising `--max-num-batched-tokens` additionally *shrinks* the pool, because the startup
   profile run sizes activations at that value (SOURCE_TRACE.md, Stage 0).

**Prediction (not yet tested).** Under a workload of long prompts + chunked prefill + a
tight pool, `03-batching-simulator`'s `on_demand` mode should admit *fewer* concurrent requests than vLLM
(it reserves whole prompts) and therefore under-estimate vLLM's throughput, while
over-estimating its stability (fewer preemptions). A natural follow-up for `03-batching-simulator` is a third
`kv_mode` that does *check-whole-prompt, allocate-per-chunk, watermark* and re-runs `03-batching-simulator`
experiment C; I have not implemented that here.

## 3. How the sweep knobs map onto both schedulers

| vLLM flag | `03-batching-simulator` parameter | Mechanism |
|---|---|---|
| `--max-num-seqs` | `max_running_seqs` | admission cap (SCH:1095) |
| `--max-num-batched-tokens` | `token_budget` | per-step budget (SCH:735, 738); also sizes the profile run |
| `--gpu-memory-utilization` | `num_kv_blocks` | pool size (derived, not given) |
| `--enable-chunked-prefill` (default on) | `chunked_prefill` | SCH:1387-1396 |
| `--block-size` | `block_size` | |
| request rate / prompt length | `WorkloadConfig.mean_gap_ms` / prompt distribution | offered load |
