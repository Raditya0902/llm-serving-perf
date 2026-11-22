# Runbook: serve and benchmark vLLM

> **Status: written and unit-tested on a Mac; never run against a real vLLM.**
> This laptop has no NVIDIA GPU, so everything below that touches `vllm serve` /
> `vllm bench serve` is **unexecuted**. What *was* verified: the flag names against the
> v0.34.0 source (`vllm/benchmarks/serve.py`, `vllm/config/scheduler.py`), the result-JSON key
> names the parser expects, and the orchestration logic (with a fake server). If something
> here fails on the GPU host, trust the host's `vllm … --help` over this file.

Target: a Linux host with an NVIDIA GPU (compute capability ≥ 7.5, per the v0.34.0 install
doc), `uv`, and enough disk for model weights. A 0.5B model needs little GPU memory; the
optional 1.5B–3B repeat needs more.

## 0. Get the code onto the host

```bash
git clone <this repo> && cd llm-serving-perf/04-vllm-internals
./setup_gpu_host.sh            # .venv-gpu with vllm==0.34.0 + results/env.txt
source .venv-gpu/bin/activate
```

`setup_gpu_host.sh` follows the official steps in the pinned docs
(`uv pip install vllm --torch-backend=auto`, then `vllm download-kernels`) and records the
GPU/driver/vLLM/torch versions to `results/env.txt`, which `make_report.py` embeds. Keep the
same model, workload and hardware across every comparison.

## 1. Smoke test — the exact commands from the task

Terminal A:

```bash
vllm serve Qwen/Qwen2.5-0.5B-Instruct \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.85
```

Check the startup log for the line that tells you the KV pool you actually got:

```text
GPU KV cache size: N tokens, Maximum concurrency for 4,096 tokens per request: X.XXx
```

(logged by `update_kv_cache_capacity`, `vllm/v1/core/kv_cache_utils.py:1157`). `N` is `num_gpu_blocks × block_size` — the number
that the source trace's Stage 0 derives from `--gpu-memory-utilization`.

Terminal B:

```bash
vllm bench serve \
    --backend vllm \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --endpoint /v1/completions \
    --dataset-name random \
    --random-input-len 512 \
    --random-output-len 128 \
    --num-prompts 100 \
    --request-rate 2 \
    --save-result
```

Stop the server (Ctrl-C) before moving on. **Do not repeat `vllm bench serve` against a
still-running server for the sweep**: the v0.34.0 docs warn that repeats can hit prompts left
in the prefix cache and inflate throughput (`docs/benchmarking/cli.md`). `run_sweep.py`
therefore starts a fresh server for every run.

## 2. The sweep (one knob at a time)

Preview exactly what will run — this works on any machine, no GPU:

```bash
python run_sweep.py --dry-run
```

`sweep_config.py` builds a **baseline** plus runs that each change **one** of
`max_num_seqs`, `max_num_batched_tokens`, `gpu_memory_utilization`, `request_rate`,
`input_len` (a test asserts each run differs from the baseline in exactly one field). The
baseline pins `--max-num-seqs 256 --max-num-batched-tokens 8192` explicitly so it is
reproducible; vLLM's own defaults depend on the GPU (`EngineArgs.get_batch_defaults`,
`vllm/engine/arg_utils.py:2977`: e.g. 2048 tokens / 256 seqs for the OpenAI server on GPUs
under 70 GiB, 8192 tokens / 1024 seqs on H100/H200-class GPUs).

Two load profiles — pick both, they answer different questions:

```bash
# (a) the task's paced workload: 2 req/s. Good for latency vs request rate / prompt length.
python run_sweep.py --load brief --repeat 3 --out results/raw/qwen0.5b-brief

# (b) saturated: all requests at once (rate=inf). Needed for scheduler limits to bite.
python run_sweep.py --load saturated --repeat 3 --out results/raw/qwen0.5b-saturated
```

Why (b): at 2 req/s with 128 output tokens a 0.5B model keeps only a handful of requests
in flight, far below `max_num_seqs=256` or any plausible KV pool, so (a) will likely show
`max_num_seqs`, `max_num_batched_tokens` (above ~1k) and `gpu_memory_utilization` as flat.
That is a prediction from the source reading, not a result — the sweep is how you find out.

Useful flags: `--only max_num_seqs input_len` (subset), `--force` (re-run existing results),
`--startup-timeout 900` (slow first download/compile), `--no-ignore-eos` (the baseline passes
`--ignore-eos` so every request generates exactly 128 tokens — a deliberate deviation from the
task's command, which makes output length constant across runs).

Output per run `<id>`, rep `<n>`, under the `--out` directory:

| File | Content |
|---|---|
| `<id>__r<n>.json` | `vllm bench serve --save-result` output, tagged with `sweep_*` / `p_*` metadata |
| `metrics/<id>__r<n>.metrics.txt` | the server's `/metrics` scraped right after the bench (source of **preemption counts**) |
| `logs/<id>__r<n>.server.log` | full `vllm serve` stdout/stderr (KV pool size, warnings, preemption messages) |

Safety properties (each is covered by a unit test, but only against stand-in processes):

- **Resume is checked, not assumed.** Every result carries a `sweep_config_id` fingerprint of
  its model, load profile, prompts, seed, EOS handling and knob values. Re-invoking the same
  command skips finished runs; pointing a *different* configuration at the same `--out`
  directory fails with "different configuration" instead of silently mixing results
  (`--force` overwrites, or use a new `--out`). `parse_results.py` likewise refuses to average
  repetitions whose fingerprints differ.
- **Stale data cannot masquerade as new data.** A run deletes its previous result/metrics
  files before starting.
- **No benchmarking of the wrong server.** The runner refuses to start if something already
  listens on `--port`, and binds the server to `127.0.0.1` only (it is unauthenticated).
- **Teardown is thorough.** The whole process group is signalled, even if the API-server
  process has already exited, because vLLM's engine/worker children can outlive it and hold the GPU.
  SIGTERM/SIGHUP to the sweep (kill, SSH disconnect) trigger the same teardown; exit code 130.
- Raw JSON includes every generated text and is git-ignored (`results/raw/`).

## 3. Summarise, plot, report

```bash
TAG=qwen0.5b-brief
python parse_results.py --raw results/raw/$TAG --out results/$TAG
python plot_results.py  --summary results/$TAG/sweep_summary.json --out results/$TAG
python make_report.py   --summary results/$TAG/sweep_summary.json --env results/env.txt --out results/$TAG/RESULTS.md
```

Repeat per `TAG`. `make_report.py` only formats the numbers; fill in the Observations section
yourself, then copy the findings into [CHECKPOINT.md](CHECKPOINT.md) Q6. It refuses to overwrite
an `--out` file whose Observations you have edited (`--force` overrides), and its default output is
`results/RESULTS.generated.md`, never the top-level `RESULTS.md`.

## 4. Larger model (optional; run: 1.5B, 51 runs, see RESULTS.md)

If memory allows, repeat with a 1.5B–3B model, same everything else:

```bash
python run_sweep.py --load saturated --repeat 3 --model Qwen/Qwen2.5-1.5B-Instruct \
    --out results/raw/qwen1.5b-saturated
```

## 5. How to read the results (what each knob should and should not do)

- **Throughput vs latency**: compare `output_throughput` with `p99_ttft_ms` and `p99_itl_ms`
  per knob (the plots show these together).
- **Preemptions** (`num_preemptions` column): non-zero means the KV pool ran out
  (CHECKPOINT Q4). If `gpu_memory_utilization` changes nothing and preemptions are 0 throughout,
  the pool was never the constraint for that workload.
- **Server-side vs client-side time**: TTFT includes queueing in the scheduler's waiting queue;
  a TTFT blow-up at rate=inf is queueing, not slow prefill.
- **Failed requests**: any `failed > 0` is flagged in the report; those rows' latencies cover only
  completed requests.
- Single-repetition rows carry no variance information; use `--repeat 3` or more.

## 6. Alternative

vLLM ships its own sweep driver, `vllm bench sweep serve`, which also resets caches between runs
(`docs/benchmarking/sweeps.md`). This repo's `run_sweep.py` exists to enforce the
*one-knob-at-a-time* rule mechanically, tag every result file with its parameters, and capture
preemption counts and server logs — not because the built-in one is inadequate.
