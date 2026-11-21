# How vLLM implements tensor parallelism (source notes)

Source: vLLM **v0.34.0**, commit `62d4bc3c7e33` — the same checkout Day 4 used
(`04-vllm-internals/vllm_src/`, git-ignored). Paths below are relative to that tree.

**How this was checked.** I read the quoted ranges directly and confirmed every symbol below
exists with `grep -n`. Line numbers are what that grep returned on this checkout; they will
drift on other versions, and (as on Day 4) interior line numbers are not machine-checked.
I did **not** read the NCCL wrappers (`pynccl.py`, `custom_all_reduce.py`) — only the code that
chooses between them — and I did not run any of this (no GPU here).

## 1. The two layer types are the ones I built

| Idea | My code (`tensor_parallel.py`) | vLLM v0.34.0 |
|---|---|---|
| Column-parallel linear | `ColumnParallelLinear` | `ColumnParallelLinear` — `vllm/model_executor/layers/linear.py:439` |
| Row-parallel linear | `RowParallelLinear` | `RowParallelLinear` — `linear.py:1686` |
| Don't gather after a column layer | `gather_output=False` (my default is `True`) | `gather_output: bool = False` (`linear.py:476`) — vLLM defaults to *not* gathering |
| Row layer takes a sharded input | `input_is_parallel=True` | `input_is_parallel: bool = True` (`linear.py:1728`) |
| AllGather on the column output | `all_gather_cat(y, dim=-1)` | `if self.gather_output and self.tp_size > 1: tensor_model_parallel_all_gather(output_parallel)` (`linear.py:628-630`) |
| AllReduce on the row output | `all_reduce_sum(partial)` | `if self.reduce_results and self.tp_size > 1: tensor_model_parallel_all_reduce(output_parallel)` (`linear.py:1850-1851`) |
| Shard the weight by rank | `_shard(weight, dim, rank, n)` (`narrow` + clone) | `loaded_weight.narrow(output_dim, tp_rank * shard_size, shard_size)` at load time (`linear.py:599-601`) — each rank only ever loads its slice from disk |
| Bias in a row layer | added **after** the reduce | added **only on rank 0, before** the reduce: `bias_ = None if (self.tp_rank > 0 or self.skip_bias_add) else self.bias` (`linear.py:1845`) |
| Single GPU | collectives are no-ops when `world_size == 1` | guarded by `tp_size > 1`, and `GroupCoordinator.all_reduce` returns `input_` when `world_size == 1` (`parallel_state.py:746-747`) |

Both bias strategies give the same answer (a rank-0-only bias is summed once by the AllReduce;
mine is added once after it). vLLM's version saves a separate elementwise add by fusing the
bias into rank 0's GEMM.

Differences worth knowing: vLLM layers carry quantization (`quant_method.apply`), a
`reduce_results` flag (skip the AllReduce so a caller can fuse it with the next op) and
`skip_bias_add`; none of that exists in my minimal layers. Weights are sharded when loaded
(`weight_loader`), not by slicing a full tensor already in memory as my demo does.

## 2. A real model uses them exactly as in the Megatron pattern

`vllm/model_executor/models/qwen2.py` (Qwen2 — the family in the Session 3 `vllm serve` command):

- **MLP** (`Qwen2MLP`): `gate_up_proj = MergedColumnParallelLinear` (`qwen2.py:91`), then
  `SiluAndMul`, then `down_proj = RowParallelLinear` (`qwen2.py:98`). The forward is
  `gate_up, _ = self.gate_up_proj(x); x = self.act_fn(gate_up); x, _ = self.down_proj(x)`
  (`qwen2.py:112-114`). Column layer without gather → elementwise activation → row layer =
  **one AllReduce per MLP**, same as `ParallelMLP`. (`MergedColumnParallelLinear` is two
  column-parallel matrices — gate and up — fused into one GEMM. SwiGLU's `silu(gate) * up` is
  elementwise, so it commutes with sharding just like GELU.)
- **Attention**: `qkv_proj = QKVParallelLinear` (`qwen2.py:157`) and `o_proj = RowParallelLinear`
  (`qwen2.py:166`). `QKVParallelLinear` shards by **heads**:
  `self.num_heads = divide(self.total_num_heads, tp_size)` (`linear.py:1113`), with KV heads
  divided the same way, or replicated when `tp_size` exceeds the number of KV heads (the
  `num_kv_heads = 1` branch at `linear.py:1115` — I did not read the replication logic in full).
  Each rank runs attention locally for its heads, with no communication, and `o_proj`'s
  AllReduce combines them.
- So a transformer layer costs **two AllReduces** (attention output, MLP output) in the forward
  pass. For Qwen2.5-1.5B that is `2 × num_layers` AllReduces per generated token — the number the
  Session 3 TP=2 experiment's communication overhead is made of. (Layer count: check the model
  config on the GPU host; I haven't loaded it here.)

## 3. How the communication is dispatched

```
RowParallelLinear.forward           linear.py:1850
  tensor_model_parallel_all_reduce  vllm/distributed/communication_op.py:12
    get_tp_group().all_reduce       parallel_state.py:730   (GroupCoordinator, class at :427)
      device_communicator.all_reduce  vllm/distributed/device_communicators/cuda_communicator.py:389
```

`get_tp_group()` is defined at `parallel_state.py:1598`; `initialize_model_parallel(
tensor_model_parallel_size=...)` (`parallel_state.py:2085`) builds the process groups by
reshaping all ranks into rows of `tensor_model_parallel_size` (`parallel_state.py:2175`:
`all_ranks.view(-1, tensor_model_parallel_size).unbind(0)`), so **adjacent ranks form a TP group**
— with the usual rank-to-GPU mapping that keeps TP traffic on GPUs that sit close together.

`cuda_communicator.all_reduce` (read down to line ~434) is a **ladder** of backends, tried in order
by `should_*` predicates on the tensor: NCCL symmetric-memory all-reduce, "quick all-reduce"
(an AMD path), FlashInfer all-reduce variants, AITER (AMD), then `ca_comm`, vLLM's own
**custom all-reduce** kernel, and presumably PyNCCL as the fallback (I stopped reading before the
tail). The point for this lab: for the small messages decode produces, vLLM does not just call
`ncclAllReduce`; it has hand-written low-latency paths because **at small sizes the α (latency)
term dominates** — the same regime my Exp A fits measure on Gloo.

## 4. What this tells us about the Day 5 hypotheses

- Per-token TP cost = `2 × layers` small AllReduces, each of size `tokens × hidden × 2 bytes`
  (fp16). At decode that is KBs, so latency (α), not bandwidth, is the bill. This is why custom
  all-reduce kernels exist and why TP=2 on a 1.5B model is *not* assumed to be faster.
- Whether TP=2 actually wins for Qwen2.5-1.5B is an **open question to settle on a GPU host**
  (see `RUNBOOK.md`); nothing in this file is a measurement.
