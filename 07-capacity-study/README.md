# Capacity study: 8B serving benchmark, optimization decision, and report

> **Update (A40 run).** The matrix has now been run on a real GPU (one NVIDIA A40, vLLM 0.34.0, Llama-3.1-8B, KV pool capped to the
> modeled A100-40GB size): 49 runs, 0 failures. **Results and verdicts are in [GPU_RESULTS.md](GPU_RESULTS.md)**: H1, H3, H5, H6 held,
> H2 half refuted (the simulator had the wrong sign for the token-budget knob), H4 borderline. The decision is unchanged (keep baseline).
> **Two-GPU update.** The tensor-parallel predictions were also run: on 2x A100 NVLink (TP=2 +62-77% tok/s, step 1.61-1.77x) and on 2x A40 over PCIe (+35-52%, step 1.29-1.53x), with the KV pool capped to the modeled size
> (TP=2 removes the preemptions: 62 -> 0, pool 2.88x), an 8B data-parallel baseline (replicas +31% over TP=2 at 64 concurrent, TP=2 +16% at 16) and the uncapped A40 (the simulator's no-pressure prediction fails because the real pool is ~12% smaller than modeled);
> see GPU_RESULTS.md. Round 5 (Lambda Cloud) ran the NVLink data-parallel arm and the same-GPU PCIe/NVLink pair (GPU_RESULTS.md, last section; `run_round5_a100.sh`). Still not done: 8B step-level profiling. The paragraph below is the **pre-run** status and is kept for the record.
>
> **What was and wasn't done before the GPU run.** The 12-case benchmark matrix, the optimization decision and the report are **done on the [`03-batching-simulator`](../03-batching-simulator/)
> simulator and are SIMULATED**: no GPU produced any number in `results/sim/`. The vLLM runner that would produce real numbers is
> built and tested only with stand-in server/bench/`nvidia-smi` processes; **it has never run against vLLM**, because this was built
> on a Mac with no NVIDIA GPU (same limit as folders 04-06). The decision in [REPORT.md](REPORT.md) section 8 is a **hypothesis** until
> [RUNBOOK.md](RUNBOOK.md) is executed on a GPU host. Peak GPU memory is *not measured* anywhere. The two-GPU / multi-node numbers
> are an analytic model with **assumed** link constants.

## Files

| File | What it is |
|---|---|
| [REPORT.md](REPORT.md) | The main report: eight sections, the before/after comparison, and the biggest tradeoff |
| [ANSWERS.md](ANSWERS.md) | The ten closed-book questions, each answered with an equation, diagram, example or pointer to code |
| [RUNBOOK.md](RUNBOOK.md) | How to run the matrix on a Linux GPU host, and the predictions the real run will confirm or refute |
| `matrix.py` | The 12-case matrix (2 workloads x 3 concurrencies x 2 configs), the A/A control, and the exact `vllm serve` / `vllm bench serve` commands |
| `kv_sizing.py` | [`02-kv-cache-memory`](../02-kv-cache-memory/) byte counts -> KV pool size, a hardware-derived cost model, and the analytic capacity-pressure check |
| `closed_loop_sim.py`, `run_matrix_sim.py` | Fixed-concurrency client on the [`03-batching-simulator`](../03-batching-simulator/) engine; runs the matrix and an ablation (**SIMULATED**) |
| `run_matrix_gpu.py`, `prom_metrics.py`, `gpu_mem.py`, `periodic.py` | Real vLLM runner: fresh server per run, rotating case order, background `/metrics` poller (peak KV use, queue, preemptions) and `nvidia-smi` sampler |
| `ingest_gpu.py` | vLLM bench JSON (+ sidecar) -> the same row schema the simulator emits; rejects failed, partial or mismatched runs |
| `verify_flags.py` | Checks every CLI flag the matrix emits against the pinned vLLM v0.34.0 source (all 26 present) |
| `stability.py`, `objective.py`, `bottleneck.py` | Repeat aggregation + A/A noise rules; the pre-registered objective and before/after; rule-based bottleneck labels |
| `multi_gpu.py` | Analytic tensor-parallel model: decode step time and group KV capacity vs TP degree (**ASSUMED** link constants) |
| `analyze.py`, `make_report.py`, `plot_matrix.py`, `diagrams.py` | Results -> `analysis.json`, generated tables, figures, architecture diagram |
| `results/sim/` | Simulated results: `matrix*`, `ablation*`, `analysis.json`, `multi_gpu.*`, `tables.generated.md`, `figures/` |
| `results/architecture.png` | Inference architecture diagram |
| `tests/` | pytest suite (CPU only, no model download, no GPU); `tests/helpers.py` builds **synthetic** vLLM-shaped files that mean nothing |

## Quick start (any machine, no GPU)

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -m pytest tests -q --cov=.                 # unit + integration tests
.venv/bin/python run_matrix_sim.py --ablation               # SIMULATED matrix + ablation -> results/sim/
.venv/bin/python analyze.py --dir results/sim               # objective, before/after, bottlenecks, multi-GPU model
.venv/bin/python plot_matrix.py --dir results/sim           # figures
.venv/bin/python make_report.py --dir results/sim           # results/sim/tables.generated.md
.venv/bin/python diagrams.py                                # results/architecture.png
.venv/bin/python run_matrix_gpu.py --dry-run                # print every vLLM command; starts nothing
.venv/bin/python verify_flags.py                            # needs 04-vllm-internals/vllm_src (fetch_vllm_source.sh)
```

Earlier folders (02-05) are imported by `_prior.py`. **[`06-profiling`](../06-profiling/) is not imported** (it was developed on a separate branch and is now merged); its GPU-memory sampler was re-implemented in
`gpu_mem.py` and its report is cited as [../06-profiling/performance-report.md](../06-profiling/performance-report.md).

## Experiment design in one paragraph

Workload A (chat, ~256 in / ~128 out) and B (long context, ~2560 in / ~384 out, lengths varying +-25%), concurrency 4 / 16 / 64 in a closed
loop, baseline vs `tuned` (`gpu_memory_utilization` 0.85 -> 0.92, `max_num_batched_tokens` 8192 -> 2048). Each case runs 3 repeats (5 for
concurrency 64), plus an A/A control. The objective and its latency limit were fixed before any run: maximize output tok/s subject to
P95 end-to-end <= 3 x the baseline's P95 at concurrency 4. Details and results: [REPORT.md](REPORT.md).

## Limitations

- **No GPU numbers.** Everything in `results/sim/` comes from the [`03-batching-simulator`](../03-batching-simulator/) roofline cost model with constants derived from *illustrative*
  A100-40GB peaks (`kv_sizing.HARDWARE`). Magnitudes will differ on real hardware; the report names which predictions are structural
  (where KV pressure appears) and which are magnitude-sensitive.
- **Simulator simplifications:** no attention-quadratic prefill cost, no prefix cache, no CUDA-graph or host overhead beyond a fixed 0.3 ms,
  vLLM's whole-prompt admission check is not modelled, and the closed-loop client has zero think time.
- **`tuned` was chosen a priori, not searched**, and the objective factor `k = 3.0` is a judgment call, not derived. Both are fixed in
  `matrix.py` / `objective.py` before the runs; the ablation attributes `tuned` but does not tune it.
- **Two-GPU / multi-node is analytic only.** `matrix.py` has no `--tensor-parallel-size` knob, so a TP run is not automated (RUNBOOK section 5
  gives the manual procedure). Link constants are assumptions; [`05-tensor-parallel`](../05-tensor-parallel/) measured only Gloo/CPU AllReduce.
- **Server-side metrics depend on vLLM v0.34.0 names** (`vllm:kv_cache_usage_perc`, `vllm:num_preemptions`, `vllm:num_requests_waiting`).
  `verify_flags.py` checks CLI flags only, not metric names; a mismatch shows up as "not measured", never as zero.
- **Reviewed (code + security) and fixed** for the findings that were this folder's: untrusted result/sidecar files (types, sizes, symlinks, filename vs
  metadata, duplicate reps), CSV formula injection, atomic writes, telemetry threads that could die silently, resume reusing incomplete results,
  `select` comparing against an infeasible baseline, stale-analysis and mislabelled-provenance reports. **Not fixed:** [`04-vllm-internals`](../04-vllm-internals/)'s teardown under a second
  signal and unbounded `/metrics` read ([`04-vllm-internals`](../04-vllm-internals/) code), and the A/A noise floor below.
- **The A/A noise floor is one draw.** It comes from a single control case (B@16 baseline, 5 repeats) and is widened per comparison by each case's own CV;
  with 3-5 repeats both estimates are noisy, and the "ranges do not overlap" test is weak at that size. Treat a "real" verdict near the floor as
  suggestive, and treat "within noise" as "not shown", not "equal".
