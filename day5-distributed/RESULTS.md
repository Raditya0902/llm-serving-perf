# Session 3 results — vLLM TP=1 vs TP=2

**No GPU run yet.** This repository was built on a Mac without an NVIDIA GPU, so `vllm serve
--tensor-parallel-size 2`, `vllm bench serve`, NCCL and `nvidia-smi` have not been run and there
are **no measurements** here. The CPU/Gloo measurements from Day 5 are in `results/exp_*.json`
and `PARALLELISM.md`; they are about communication *cost structure*, not about GPU speed.

Follow [RUNBOOK.md](RUNBOOK.md) on a two-GPU Linux host, save raw output under
`results/raw/tp_compare/` (git-ignored), then fill in:

| Concurrency | TP | out tok/s | mean TTFT ms | mean TPOT ms | GPU mem (dev 0 / dev 1) | % GPU time in all-reduce |
|---|---|---|---|---|---|---|
| 1 | 1 | – | – | – | – / n.a. | n.a. |
| 1 | 2 | – | – | – | – / – | – |
| 8, 32, 128 | 1, 2 | – | – | – | – | – |

Also record: `topo.txt` (NVLink vs PCIe), `nccl_all_reduce.txt` (α and bus bandwidth),
two-replica TP=1 throughput, and the outcome of each hypothesis in RUNBOOK §5.
