# Day 6 GPU runbook (NVIDIA host)

> **Status (updated): run on RunPod A40 pods.** Sections 1, 3, 4 and 5 produced the measurements in [performance-report.md](performance-report.md) section 9; section 2 (nsys) ran in a second round
> (nsys 2026.5.2 from the CUDA apt repo; `gpu/run_nsys.sh` now passes `--sample=none --cpuctxsw=none`, needed in a container). New in that round: `profile_run.py --model` and `--probe-only` (unprofiled launch probe),
> `gpu/step_costs.py` (per-step GPU time vs prefill tokens and decodes), `gpu/run_itl_study.sh`, `gpu/run_e6_cap.sh` (E6 with a pool the workload can exhaust). The text below is the original procedure.
> The vLLM flags were read out of the pinned v0.34.0 source in `../day4-vllm-internals/vllm_src/`. If the host's `--help` disagrees with this file, trust the host.

Hypotheses for every experiment are fixed in [experiments.py](experiments.py) *before* any run.
Experiments E4-E6 (max batched tokens, max seqs, KV capacity) are the ones that need this runbook;
the Day 3 simulator gives a labelled **SIMULATED** prediction in `results/sim/`, not a result.

## 0. Setup

```bash
# on the GPU host, from the repo root
bash day4-vllm-internals/setup_gpu_host.sh          # pins vllm==0.34.0, writes results/env.txt
cd day6-profiling && uv venv --python 3.12 .venv-gpu && uv pip install --python .venv-gpu/bin/python -r requirements.txt
```

Keep model, workload and hardware identical across every comparison; record `nvidia-smi` clocks
and whether the GPU is otherwise idle (the memory sampler reads whole-GPU memory).

## 1. Profile the Day 1 loop with PyTorch Profiler on CUDA

```bash
.venv-gpu/bin/python profile_run.py --device cuda --prompt-len 512 --new-tokens 33 --tag cuda-baseline
```

Same script as on the Mac, but on CUDA `ProfilerActivity.CUDA` is recorded, so
`profile_summary.json` gets real values for what the Mac cannot show:

| Question | Where to read it |
|---|---|
| Which ops take the most time | `phases.*.top_ops`, `ops_<phase>.csv` (device time on CUDA) |
| Matmul vs attention share | `phases.*.category_totals` (device time basis) |
| Gaps between GPU ops | `phases.*.gaps.busy_fraction`, `gap_p95_us`, `gap_max_us` (`available: true`) |
| Allocations during generation | `phases.*.alloc_events` and `alloc_probe` |
| Is the CPU delaying the GPU | `launch_probe.enqueue_fraction` (>= 0.9 means CPU-bound) |

Open `trace_decode.json` in <https://ui.perfetto.dev> to look at the timeline directly.
Caveat that applies here too: Day 1's loop syncs every step, so the GPU idles at the end of each
step by construction. Compare with the `deferred_sync` variant (`run_local_experiments.py --exp O1`).

## 2. nsys on the same window

```bash
bash gpu/run_nsys.sh --phase decode --prompt-len 512 --new-tokens 33
nsys stats results/raw/nsys/inference_trace.nsys-rep
```

`run_nsys.sh` passes `--nsys` to `profile_run.py`, which brackets its window with
`torch.cuda.profiler.start()/stop()` (the `cudaProfilerApi` that `--capture-range` waits for) and labels
attention/MLP with NVTX ranges, so only that window is recorded (warmup excluded). `--nsys` deliberately skips
`torch.profiler` and the probes, because Kineto's CUPTI subscription may clash with nsys (unverified). If no
`.nsys-rep` appears, check that the window opened: the script prints `cudaProfilerApi window done` per phase.
Look for: many tiny kernels with gaps between them (launch-bound), `cudaStreamSynchronize` / `cudaMemcpyAsync`
at each step end (the `.item()`), and the `d6::attn` / `d6::mlp` NVTX ranges.

## 3. Profile vLLM

From `vllm/config/profiler.py` (`ProfilerConfig`) and `vllm/entrypoints/serve/profile/api_router.py`:

```bash
mkdir -p "$PWD/results/raw/vllm-trace"
vllm serve Qwen/Qwen2.5-0.5B-Instruct --max-model-len 4096 --gpu-memory-utilization 0.85 \
  --profiler-config.profiler=torch \
  --profiler-config.torch_profiler_dir="$PWD/results/raw/vllm-trace" \
  --profiler-config.torch_profiler_with_stack=false \
  --profiler-config.delay_iterations=20 --profiler-config.max_iterations=20
```

- `torch_profiler_dir` **must be an absolute path**. `delay_iterations` skips warm-up iterations and
  `max_iterations` caps the window, which is the "limited profiling window" the task asks for.
- Start/stop either with `vllm bench serve ... --profile` (the server must have `--profiler-config`) or with
  `curl -X POST localhost:8000/start_profile` / `/stop_profile`.
- Offline alternative: `vllm_src/examples/features/profiling/run_one_batch_offline.py --profile decode --profile-dir ...`.
- Profiling slows the server; never use a profiled run for throughput numbers.

## 4. Baseline vs modified comparison (the report table)

Day 4's `run_sweep.py` requests percentiles `50,99` and does **not** pass `--save-detailed`, and vLLM
v0.34.0 deletes per-request arrays from the saved JSON without it (`vllm/benchmarks/serve.py`
~L2428). So Day 4 sweep files carry no P95. For the two configurations in the report table, run the
bench directly, once per configuration, against a *fresh* server each time (prefix-cache warning in
Day 4's runbook):

```bash
# terminal A: server for configuration X (e.g. baseline: 256 seqs / 8192 tokens)
vllm serve Qwen/Qwen2.5-0.5B-Instruct --max-model-len 4096 --gpu-memory-utilization 0.85 \
  --max-num-seqs 256 --max-num-batched-tokens 8192

# terminal B: bench wrapped by the memory sampler
python gpu/gpu_mem_sampler.py --out results/raw/vllm/baseline_mem.json -- \
  vllm bench serve --backend vllm --model Qwen/Qwen2.5-0.5B-Instruct --endpoint /v1/completions \
    --dataset-name random --random-input-len 512 --random-output-len 128 --num-prompts 200 \
    --request-rate inf --max-concurrency 64 --ignore-eos --seed 0 \
    --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,95,99 \
    --save-result --save-detailed --result-dir results/raw/vllm --result-filename baseline.json
```

Repeat for the modified configuration (`modified.json`), 3 times each, fresh server every time.
Then read both with `gpu/report_from_vllm.py` (`row_from_result(path, mem_path)`) and fill the table.
The sampler reports whole-GPU memory, which for vLLM is dominated by the pre-allocated KV pool, so it
tracks `--gpu-memory-utilization` and not live tensor use; say so when reporting it.

## 5. E4-E6 sweeps (Day 4 tooling)

```bash
cd ../day4-vllm-internals
python run_sweep.py --load saturated --repeat 3 \
  --only max_num_batched_tokens max_num_seqs gpu_memory_utilization \
  --out results/raw/day6-sweep
python parse_results.py --raw results/raw/day6-sweep --out results/day6-sweep
```

This covers E4 (`max_num_batched_tokens`), E5 (`max_num_seqs`) and E6 (`gpu_memory_utilization`).
Limits to keep in mind: it reports p99 not P95; it has no peak-memory metric; preemption counts come
from the scraped `/metrics` file (`vllm:num_preemptions`). E6 only shows preemptions if the KV pool
is smaller than the workload's peak demand: check the `GPU KV cache size: N tokens` startup line and
size the workload against it, otherwise E6 will look flat for the wrong reason.
For concurrency (E1) and output length (E3) on vLLM, `--max-concurrency` and `--random-output-len`
exist on `vllm bench serve` but are not in Day 4's sweep knobs; use the direct command in section 4.

## 6. Reading the results against the hypotheses

| Exp | Hypothesis (abridged, full text in experiments.py) | Simulator direction (SIMULATED) |
|---|---|---|
| E4 | Small budget: lower inter-token P95, higher TTFT for long prompts; throughput plateaus ~2048+ | `results/sim/sim_e4.csv` |
| E5 | More seqs: throughput up and TTFT down until compute or KV binds | `results/sim/sim_e5.csv` |
| E6 | Preemptions appear only below peak KV demand; above it no change | `results/sim/sim_e6.csv` |

Write each outcome as hypothesis -> result -> evidence, and say "refuted" when it is.
Do not report a speedup unless output tokens match the baseline, the workload (seed, lengths,
request count) is identical, and the repeat ranges do not overlap.
