# Day 5 — distributed collectives and tensor parallelism

> **Update:** the NCCL path and the TP=1 vs TP=2 vLLM experiment have now run on 2x A100-80GB over NVLink; see
> [RESULTS.md](RESULTS.md); the PCIe case (2x A40) and an `nsys` measurement of the all-reduce share have run too. Round 5 (Lambda Cloud) ran the same 8B arms on a 2x A100 PCIe host and an 8x A100 SXM4 host: TP=2 is 0.70x of TP=1 on the PCIe host (link about 3 GB/s) and 1.65x on NVLink (158 GB/s); see [../07-capacity-study/GPU_RESULTS.md](../07-capacity-study/GPU_RESULTS.md). The "not run / empty stub" lines below describe the state before those runs.
>
> **What is and isn't done (before the GPU run).**
> - **Done and measured (CPU, Gloo backend, Apple M4, 2–4 processes):** `collectives.py`
>   (init, rank, AllReduce, Broadcast, AllGather, barrier, clean shutdown) running under real
>   `torchrun`; column- and row-parallel linear layers and a Megatron-style MLP that reproduce
>   the unsharded output; a communication benchmark and a latency/bandwidth model of it.
> - **Not run: anything on GPUs.** This Mac has no NVIDIA GPU, so the **NCCL path**
>   (written, `dist_utils.resolve_backend`) and the **TP=1 vs TP=2 `vllm serve` experiment**
>   have never executed. [RUNBOOK.md](RUNBOOK.md) is the procedure; [RESULTS.md](RESULTS.md) is an
>   empty stub. Statements about GPU behaviour are marked as hypotheses.
> - **Read this before trusting Exp B's speed-ups:** on this Mac, extra processes did *not*
>   parallelise the matmul well (about 1.07–1.38× compute speed-up at large shapes, even with no
>   communication). The benchmark is a clean measurement of **communication cost**, not of
>   how much TP speeds up compute on real GPUs. Details in [PARALLELISM.md](PARALLELISM.md) §3.

## Files

| File | Purpose |
|---|---|
| `collectives.py` | **Session 2 deliverable.** Functional wrappers (`all_reduce_sum`, `broadcast_from`, `all_gather_cat`, `synchronize`) + a torchrun demo with self-checks |
| `dist_utils.py` | Init from torchrun env, backend choice (NCCL iff one CUDA device per rank, else Gloo), clean shutdown, `spawn_group` for tests/benchmarks without torchrun |
| `tensor_parallel.py` | **Session 3 deliverable.** `ColumnParallelLinear`, `RowParallelLinear`, `ParallelMLP`, shard helpers |
| `tp_demo.py` | torchrun entry: checks column / row / MLP against a single-device reference |
| `launch.sh` | `torchrun` launcher; works around macOS hostname resolution (see Limitations) |
| `bench_model.py` / `bench_workers.py` / `run_experiments.py` / `plot_results.py` | Communication benchmark: pure maths / per-rank workers / CLI (`--exp A\|B\|C\|all --smoke`) / figures |
| `PARALLELISM.md` | **Session 4:** TP diagram, DP vs TP, benchmark results, GPU topology, decision guide |
| `CHECKPOINT.md` | Answers to the six knowledge-checkpoint questions |
| `TP_SOURCE_NOTES.md` | How vLLM v0.34.0 implements TP, mapped onto my code |
| `RUNBOOK.md`, `RESULTS.md` | The unrun two-GPU vLLM experiment: commands, metrics, hypotheses; empty results stub |
| `tests/` | pytest; multi-process tests run real Gloo groups |
| `results/` | `exp_a/b/c[_fits].{json,csv}`, `run_log.txt`, figures |

## Setup and run

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt

./launch.sh 2 collectives.py     # = torchrun --standalone --nnodes=1 --nproc-per-node=2 collectives.py on Linux
./launch.sh 2 tp_demo.py
.venv/bin/python run_experiments.py --exp all [--smoke] [--out DIR] [--seed N]
.venv/bin/python plot_results.py
.venv/bin/python -m pytest tests -q --cov=.
```

With two GPUs, the same commands pick NCCL automatically.

## Results in one paragraph

Gloo AllReduce on 2 ranks costs ~150–190 µs whether it moves 1 KB or 64 KB (a latency floor
α ≈ 151 µs), then grows linearly (bus bandwidth up to ≈ 6.3 GB/s). A tensor-parallel MLP
therefore pays that toll on every forward pass: for small shapes the AllReduce is 56–96% of TP's
time and TP is up to 12× (2 ranks) / 33× (4 ranks) *slower* than one process; it beat one process
in only 4 of 16 shapes on both 2 and 4 ranks (best 1.76× / 1.55×, partly artifacts). A
two-parameter α/β model fitted on the collective benchmark predicts the AllReduce time seen
inside the MLP to within 0.5–2× in 31 of 32 cases (median 0.76×).
Figures and tables: [PARALLELISM.md](PARALLELISM.md).

## Knowledge checkpoint

See [CHECKPOINT.md](CHECKPOINT.md).

## Limitations

- **CPU/Gloo only.** No NCCL, NVLink, PCIe or multi-node measurement. Rows with
  `source=assumption` in `exp_c.json` use made-up round hardware numbers.
- **Compute scaling across processes is poor on this Mac**, so Exp B's speed-ups are bounded by
  that, not just by communication (explained above). Some small shapes show super-linear
  compute speed-ups (probably cache effects — untested), and the best "speed-ups" are partly a
  baseline-kernel artifact. Repeat-to-repeat spread is recorded per row (`*_spread`) and reaches
  218% on the 4-rank baseline and 183% on one collective case; the robust findings are the large
  effects only.
- **α/β fits are rough** (worst point off by 19–41%).
- **Timing method:** each timed call starts after a barrier, so latencies include barrier-exit
  skew; the isolated AllReduce and the in-MLP total are different statistics. macOS has no CPU
  affinity and this chip has 4 performance + 6 efficiency cores (recorded in the result
  metadata), so 4-rank runs may mix core types.
- **macOS:** `torchrun --standalone` hung here because this machine's hostname does not resolve
  (Gloo tried to look it up). `launch.sh` rendezvous on `127.0.0.1` with `GLOO_SOCKET_IFNAME=lo0`
  instead; `spawn_group` does the same. On Linux `launch.sh` is plain `--standalone`.
- **One intermittent Gloo crash, not reproduced.** The first full benchmark run died with
  `gloo … Unexpected opcode` (SIGABRT) and could not be reproduced (the same sizes passed in
  isolation and in three subsequent full runs). `run_experiments.py` retries a rank-death once,
  logs it, and records retries in the result metadata (`spawn_retries`); Python exceptions are
  never retried. The final run needed no retry.
- **Forward pass only**; backward (the mirror-image collectives) is described, not implemented.
- **Tests:** 117 tests, 98% line coverage *with subprocess tracing* (`.coveragerc`; without it
  coverage reads ~84% because worker code runs in spawned processes). Untested: the NCCL
  branch, which needs GPUs. Coverage makes the suite take ~80 s instead of ~30 s.
