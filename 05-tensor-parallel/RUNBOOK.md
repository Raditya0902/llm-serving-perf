# Runbook: vLLM TP=1 vs TP=2 on two GPUs

> **Status (updated): run once on 2x A100-80GB over NVLink** via `run_tp_compare.sh`; results and verdicts are in [RESULTS.md](RESULTS.md).
> Also run on 2x A40 over PCIe, including 3a (`nsys`, via `run_nsys_tp2.sh`, concurrency 8 only); see the PCIe section of RESULTS.md. The text below is the original procedure, written on a Mac and kept as written:
> **Status: written, never run.** This Mac has no NVIDIA GPU, so every command below that
> touches `vllm serve`, `vllm bench serve`, `nvidia-smi`, `nsys` or NCCL is **unexecuted**.
> What I did verify, against the vLLM v0.34.0 source in `04-vllm-internals/vllm_src/`: the
> flag names `--tensor-parallel-size` / `-tp` (`vllm/engine/arg_utils.py:1224`),
> `--disable-custom-all-reduce` (`arg_utils.py:1382`), and the bench flags `--max-concurrency`,
> `--request-rate`, `--save-result`, `--result-filename`, `--percentile-metrics`
> (`vllm/benchmarks/serve.py`). If something fails on the GPU host, trust the host's
> `vllm … --help` over this file. Predictions in section 5 are **hypotheses to test**, not results.

Target: a Linux host with **two** NVIDIA GPUs, ideally on one NVLink/NVSwitch island *and* a
second configuration over PCIe if you can get one (the interconnect is the main variable).
Qwen2.5-1.5B-Instruct (~3 GB in bf16) fits on one GPU, so **both TP=1 and TP=2 are valid**
and the comparison is fair — that is the point of the experiment.

## 0. Setup and topology (record this first)

```bash
cd llm-serving-perf/04-vllm-internals && ./setup_gpu_host.sh && source .venv-gpu/bin/activate
mkdir -p ../05-tensor-parallel/results/raw/tp_compare
nvidia-smi topo -m | tee ../05-tensor-parallel/results/raw/tp_compare/topo.txt   # NV# = NVLink, PIX/PHB/SYS = PCIe paths
nvidia-smi --query-gpu=index,name,memory.total --format=csv | tee ../05-tensor-parallel/results/raw/tp_compare/gpus.txt
```

`topo.txt` is what lets you interpret every number below. Also run the NCCL baseline on the
same two GPUs (no vLLM involved):

```bash
git clone https://github.com/NVIDIA/nccl-tests && make -C nccl-tests
./nccl-tests/build/all_reduce_perf -b 1K -e 256M -f 2 -g 2 | tee ../05-tensor-parallel/results/raw/tp_compare/nccl_all_reduce.txt
```

This gives the real α (small-size latency) and bus bandwidth to compare with the Gloo-on-CPU
fits in Exp A and with the assumed numbers in Exp C.

## 1. Servers — one at a time, same everything else

```bash
# TP=1: pin to GPU 0 so the second GPU is idle (and observable)
CUDA_VISIBLE_DEVICES=0   vllm serve Qwen/Qwen2.5-1.5B-Instruct --max-model-len 4096 --gpu-memory-utilization 0.85 --tensor-parallel-size 1
# TP=2
CUDA_VISIBLE_DEVICES=0,1 vllm serve Qwen/Qwen2.5-1.5B-Instruct --max-model-len 4096 --gpu-memory-utilization 0.85 --tensor-parallel-size 2
```

Stop the server between runs (a fresh server per run avoids prefix-cache inflation, as in `04-vllm-internals`).
From each startup log, copy the `GPU KV cache size: N tokens` line — TP=2 frees weight memory on
each GPU, so the KV pool per GPU differs even though the model "fits both ways".

## 2. Load — closed loop at several concurrencies

Throughput, TTFT and TPOT depend on load, so sweep it; do not report a single point.

```bash
for C in 1 8 32 128; do
  vllm bench serve --backend vllm --model Qwen/Qwen2.5-1.5B-Instruct --endpoint /v1/completions \
    --dataset-name random --random-input-len 512 --random-output-len 128 \
    --num-prompts $((C * 8 > 200 ? C * 8 : 200)) --max-concurrency $C --request-rate inf \
    --percentile-metrics ttft,tpot,itl,e2el --save-result \
    --result-filename ../05-tensor-parallel/results/raw/tp_compare/tp${TP}_c${C}.json
done
```

(`TP=1` or `TP=2` set in the shell to match the running server.) Same prompts, same seed,
same lengths in both configurations.

## 3. What to measure, and where it comes from

| Metric | Source |
|---|---|
| Throughput (output tok/s) | `output_throughput` in the result JSON |
| TTFT | `mean_ttft_ms` (and `p99_ttft_ms`) |
| TPOT | `mean_tpot_ms` (and p99) |
| **GPU memory per device** | while the bench runs: `nvidia-smi --query-gpu=index,memory.used --format=csv -l 1 > results/raw/tp_compare/mem_tp${TP}_c${C}.csv`. Compare device 0 in TP=1 vs devices 0 and 1 in TP=2. |
| **Communication overhead** | see 3a–3c |

**3a. Share of time in collectives.** Profile a short steady-state window with Nsight Systems:
`nsys profile -t cuda,nvtx -o tp2_c8 --duration 20 <command that launches the server>`, then
look at the kernel summary for `ncclDevKernel_AllReduce*` (or vLLM's custom all-reduce kernel)
as a percentage of total GPU time per rank. Expected count per generated token:
`2 × num_layers` AllReduces (see `TP_SOURCE_NOTES.md` §2). Get `num_layers` from the model config.

**3b. Isolate the all-reduce implementation.** Repeat TP=2 at one concurrency with
`--disable-custom-all-reduce` (falls back to plain NCCL). The delta is what vLLM's low-latency
kernel buys at small message sizes.

**3c. Interconnect sensitivity.** If you have access to both NVLink and PCIe-only pairs, repeat
TP=2 on each. The topology file from §0 says which you have.

## 4. Optional: the other way to use two GPUs

TP=2 competes with *two independent TP=1 servers* (data parallelism, one per GPU, with
a load-balancer in front or two clients). For a model that fits on one GPU this is the real
baseline for "should I use the second GPU for TP?". Run two TP=1 servers on ports 8000/8001
(`CUDA_VISIBLE_DEVICES=0` / `=1`), split the prompts, and add the two throughputs.

## 5. Hypotheses to confirm or refute (write the outcome into RESULTS.md)

Derived from the source reading and the CPU benchmark, **not measured on a GPU**:

1. **TP=2 will not double throughput for a 1.5B model.** Per token it adds `2 × layers`
   small AllReduces whose cost is dominated by latency α, not bandwidth; Exp B showed the
   same shape on CPU (71–97% of the time in AllReduce for `d_model ≤ 512`, `tokens ≤ 16`).
2. **TP=2 may lower TPOT/TTFT at low concurrency on NVLink** (halved weight reads per token) and
   **lose on PCIe**. Exp C's assumed-hardware rows predict exactly that split — as assumptions.
3. **Two TP=1 replicas beat one TP=2 server on throughput** when the model fits one GPU.
4. **Per-GPU memory:** weights halve under TP=2, so more of each GPU's budget goes to KV cache.
5. **Disabling custom all-reduce hurts TP=2 at small batch** (small messages).

If any of these is refuted, that is the interesting result — say so in RESULTS.md, with the
topology file and `nccl_all_reduce.txt` next to it.
