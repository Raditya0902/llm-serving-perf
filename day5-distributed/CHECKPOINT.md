# Day 5 knowledge checkpoint

Numbers marked **[measured]** come from `results/` (CPU, Gloo, one Apple M4, 3-repeat medians).
Numbers or claims marked **[hypothesis]** are not measured here; no GPU or NCCL was available.

## 1. How does tensor parallelism work?

Each weight matrix is cut across `n` devices so each stores `1/n` of the weights and does `1/n`
of the multiply-adds. For `Y = X·W` there are two cuts. **Column-parallel** splits `W`'s columns:
every device sees all of `X` and produces a slice of `Y`'s columns. **Row-parallel** splits `W`'s
rows (and `X`'s columns): every device produces a full-size *partial sum* of `Y`. A transformer
block chains them (column → elementwise activation → row) so the sharded hidden activation
never has to be assembled: **one AllReduce per MLP**, and one per attention block (heads are
split across devices). I implemented and verified this in `tensor_parallel.py`: assembled outputs
match the unsharded layer to ~1e-6 (fp32) for 2 and 4 ranks (`tests/test_tensor_parallel.py`,
`tp_demo.py`). vLLM v0.34.0 does the same with `ColumnParallelLinear` / `RowParallelLinear`
(`TP_SOURCE_NOTES.md`).

## 2. What is the difference between tensor parallelism and data parallelism?

DP splits the **batch**: every device holds the whole model and handles different requests, so
inference DP needs no communication and the model must fit on one device. TP splits the
**model**: every device holds `1/n` of each matrix and all devices work on the *same* request, so
it needs communication in every layer but lets a model that doesn't fit on one GPU run, and
frees memory for KV cache. DP buys throughput; TP buys capacity (and, with a fast link, latency).
Table in `PARALLELISM.md` §2.

## 3. Why does tensor parallelism require communication?

Because after a row-parallel layer each device holds only a *partial* sum, and the true output
is the sum across devices — nobody has it until they communicate. (A column-parallel layer
needs communication only if a full output is required.) The cost is paid on the critical path
of every layer for every token: `2 × layers` AllReduces per forward pass. **[measured]** the
AllReduce for a single-token MLP at `d_model=256` took 0.19 ms on 2 ranks vs 0.017 ms for the
whole unsharded MLP.

## 4. How do AllReduce and AllGather differ?

**AllReduce**: every rank contributes a tensor of the same shape and every rank receives the
**elementwise sum** (output size = input size). **AllGather**: every rank contributes a shard
and every rank receives **all shards concatenated** (output size = `n ×` input). Sum vs
concatenate. In TP: AllReduce combines row-parallel *partials*; AllGather reassembles
column-parallel *slices*. By construction AllGather's output is `n×` its input, so the two are
not interchangeable; `tests/test_collectives.py` and `tp_demo.py` check each against a known
answer. (A ring AllReduce is equivalent to a
reduce-scatter followed by an AllGather.)

## 5. What is NCCL responsible for?

NCCL is NVIDIA's library that implements the collectives themselves *on GPUs*: detecting the
topology (NVLink, PCIe, NICs); choosing the algorithm (ring/tree) and protocol per message size;
moving the data over the best transport (NVLink P2P, PCIe, InfiniBand/RDMA, sockets);
and running it asynchronously on CUDA streams. It is **not** responsible for deciding *what* to
communicate (PyTorch / vLLM do), for overlapping communication with compute, or for process
rendezvous (`torchrun` + the c10d store). Gloo plays NCCL's role for CPU tensors, which is
what I ran; my code chooses NCCL only when each rank has its own CUDA device
(`dist_utils.resolve_backend`), and **that branch has never run on real hardware**
**[hypothesis that it works]**. For vLLM's choices above NCCL (custom all-reduce, etc.) see
`TP_SOURCE_NOTES.md` §3.

## 6. When does communication overhead outweigh the parallelism benefit?

TP over `n` devices wins only if the compute it saves exceeds the collective it adds:

```
T_single · (1 − 1/n)  >  α + β · message_bytes
```

with α the collective's fixed latency and β its per-byte cost. So it loses when the work is
small (small model, few tokens per step) or the link is slow.

**[measured, CPU/Gloo]** α(2-rank AllReduce) ≈ 151 µs, β ≈ 0.123 ns/B (fit good to ±40%). On
2 ranks TP beat one process in only 4 of 16 (d_model, tokens) shapes (>1.05×; best 1.76×, which
is partly a baseline-kernel artifact), and on 4 ranks in 4 of 16 (best 1.55×). For
`d_model ≤ 512` and `tokens ≤ 16` the AllReduce was 56–96% of TP's time (e.g. 0.08× at
`d_model=256`, 1 token on 2 ranks; 0.03× on 4 ranks).
Caveat: on this Mac the compute itself only sped up ~1.07–1.38× across processes at large
shapes, so the "TP stops paying off at large sizes" half of the picture is *not* demonstrated
here — only that communication share shrinks (4–13%). Collective timings include barrier-exit
skew between ranks and the 4-rank runs may mix performance/efficiency cores.

**[hypothesis, assumed hardware]** For a Qwen2.5-1.5B-sized MLP the model predicts TP=2 slightly
helps on an NVLink-class link (1.3–1.8×) and slightly hurts on a PCIe-class link (0.9×). Measured in round 5 on the same A100 GPU: Llama-3.1-8B TPOT speedup 1.65x (NVLink, 158 GB/s) vs 0.70x (PCIe VM at about 3 GB/s), Day 7 GPU_RESULTS.md.
Whether that holds for TP=1 vs TP=2 on real GPUs is the open experiment in `RUNBOOK.md`.

---

**Completion criterion.** Distributed implementation: `collectives.py` (init, rank, AllReduce,
Broadcast, AllGather, barrier, clean shutdown) and `tensor_parallel.py` run under real `torchrun`
with 2 Gloo processes and their self-checks pass (`./launch.sh 2 collectives.py`,
`./launch.sh 2 tp_demo.py`; `tests/test_launch.py`). Explanation of how communication drives
cost: §3 and §6 above, backed by Exp A–C, with the GPU-specific parts labelled as hypotheses.
