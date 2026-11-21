# Profiling and optimization experiments

> **Status.** Measured on an Apple M4 (MPS) and, since 2026-11-18, on one NVIDIA A40 (report section 9): profiling, experiments
> E1-E3, O1, O2, a vLLM torch-profiler study, and (second A40 pod) nsys captures, models up to 7B, the 3584-token ITL trace and a KV-pool-capped E6
> sweep (report 9.5-9.8), and (third A40 pod) a kernel-level cause for the mixed-step decode cost, an isolation of the `max_num_seqs=1024` loss and 8B step costs (report 9.9-9.11). The simulated E4-E6 rows are compared against [`04-vllm-internals`](../04-vllm-internals/)'s real sweep in report section 9.4. See [performance-report.md](performance-report.md).

## Headline findings (details and caveats in the report)

**On the A40 the Mac conclusions change:**
- The [`01-inference-loop`](../01-inference-loop/) loop is **CPU-launch-bound on CUDA** (unprofiled: enqueue 97-98% of a 15.4-18.8 ms step; a loop with no syncs takes the same time as its enqueue; GPU device work is at most about 27% of the step), so [`01-inference-loop`](../01-inference-loop/)'s original launch-overhead guess was right there and the "device-bound" finding below is MPS-specific. Decode is 16.2-16.5 ms/token at every batch size, E1 throughput scales 7.6x at batch 8, TTFT is flat (17.5 ms) up to 1024 prompt tokens, and deferring syncs gains only 2.6%.
- In vLLM, a step that carries prefill costs 93-132 ms vs 7.3 ms (256 seqs) or 26 ms (999 seqs) for a pure decode step, and each decode riding in a mixed step costs about 4x more; that is the mechanism behind [`04-vllm-internals`](../04-vllm-internals/)'s ITL tails and its non-monotonic p50 ITL.
- **The loop stops being launch-bound between 3B and 7B**: enqueue/step is 0.96-0.98 up to 3B and 0.63 at 7B (30 ms step, GPU 64-69% busy); CPU cost is about 0.7 ms per layer (9.5).
- **The 234 ms p50 ITL at 3584-token prompts is explained**: 89% of steps are mixed, each ~231 ms, because a decode riding in a mixed step costs ~0.5 ms at 3584 context vs 0.11 ms at 512 (9.6).
- `max_num_seqs=1024` being slower than 256 is **not** a CUDA-graph capture-range effect (tested and refuted). It is the next finding's consequence: a **mixed step costs a decode about 4-7x more because FlashAttention 2 (A40, vLLM 0.34.0) loses its grouped-query decode path as soon as any sequence in the batch has more than one query token** (kernel-only microbenchmark: ratio 6.6 / 3.65 / 1.00 for GQA groups 7 / 4 / 1; step function at query length 2; 9.9). With `--attention-backend TRITON_ATTN` the 1024-seq loss goes from -9.4% to +0.4% (9.10) and the 3584-token workload runs **+60%** faster (9.9). Nsight Compute counters were blocked by the container, so the byte-level story is inferred.

**Apple M4 (MPS) findings:**

- Decode (Qwen2.5-0.5B, fp16, MPS) is **device-bound on weight reads**: matmuls 74% of a 15.9 ms step, effective 51-93 GB/s. The CPU enqueues a step in 3.0 ms, so [`01-inference-loop`](../01-inference-loop/)'s "launch overhead" guess was wrong.
- Removing [`01-inference-loop`](../01-inference-loop/)'s per-step host syncs: **+6.6% throughput / -7% median TPOT**, identical tokens, validated against an A/A control (-0.1%).
- `eager` attention is 33% slower than `sdpa` at decode (attention core 6.7 vs 3.1 ms, 32% more ops).
- Batch 1 -> 8 raises output tok/s only 2.8-3.5x (not the hoped 5x) because prefill is compute-bound and stops amortizing.
- Two of my own early "results" were harness artifacts (case-order bias; a wrong sync-overhead calibration); both are fixed and described in the report.

## Files

| File | Role |
|---|---|
| [performance-report.md](performance-report.md) | The deliverable: baseline table, profiling answers, hypothesis -> result -> evidence, interpretation |
| [CHECKPOINT.md](CHECKPOINT.md) | The six knowledge-checkpoint answers, tagged measured / mechanism / GPU pending |
| [RUNBOOK.md](RUNBOOK.md) | Unrun steps for an NVIDIA host: torch profiler on CUDA, nsys, vLLM profiler, E4-E6 sweeps |
| [experiments.py](experiments.py) | Registry of the six experiments + 2 optimizations; hypotheses fixed before any run |
| [profile_run.py](profile_run.py) | Profiles one [`01-inference-loop`](../01-inference-loop/) generation (prefill and decode windows); writes trace, op table, summary |
| [module_timing.py](module_timing.py) | Device-inclusive attribution where the profiler has no device timeline (matmul replay + sync bracketing) |
| [launch_probe.py](launch_probe.py), [alloc_probe.py](alloc_probe.py) | CPU-enqueue vs step time; per-step allocator growth |
| [trace_analysis.py](trace_analysis.py), [op_classify.py](op_classify.py) | Gap/alloc analysis of Chrome traces; operator categories |
| [variants.py](variants.py), [measure.py](measure.py) | Decode variants (day1, deferred sync, batched); interleaved repeats, request-level metrics, baseline deltas |
| [run_local_experiments.py](run_local_experiments.py) | Runs E1-E3, O1, O2 with controls and token validation |
| [sim_predictions.py](sim_predictions.py) | SIMULATED E4-E6 from [`03-batching-simulator`](../03-batching-simulator/) |
| [make_performance_report.py](make_performance_report.py), [plot_results.py](plot_results.py) | Generated tables and figures |
| `gpu/` | `run_nsys.sh`, `gpu_mem_sampler.py`, `report_from_vllm.py`; round 2: `step_costs.py`, `run_itl_study.sh`, `run_e6_cap.sh`; round 3: `kernel_costs.py` (per-kernel-family attribution of vLLM profiler steps, with a coverage check), `attn_microbench.py` (vLLM's attention call on pure / mixed batches), `run_backend_ab.sh`, `run_seqs_isolation.sh` + `seqs_isolation.py`, `run_ncu.sh` + `ncu_summary.py` (counters blocked in the container, untested on real output); round 4: `quality_run.py` + `quality_compare.py` (backend output equivalence), `attn_sweep.py` (context x decodes x prefill sweep, FA2/FA3, block-launch model), `backend_ab_summary.py`, `run_models_matrix.sh`, `run_e123_vllm.sh` + `e123_summary.py`, `run_round4_a40.sh` / `run_round4_h100.sh`, `preflight_round4.sh` |
| [PREREGISTRATION_ROUND3.md](PREREGISTRATION_ROUND3.md) | Round 3 hypotheses and decision rules, written before the runs, with the outcomes appended |
| [PREREGISTRATION_ROUND4.md](PREREGISTRATION_ROUND4.md) | Round 4 (output check, other models, Hopper, the ratio of 18, loose ends): predictions and rules written before the runs, outcomes appended |
| [PREREGISTRATION_ROUND5.md](PREREGISTRATION_ROUND5.md) | Round 5 (Lambda Cloud: Nsight Compute counters, busy fraction, output check at 1.5B-8B, H100, A100 PCIe vs NVLink): predictions and rules written before the runs, outcomes appended (report 9.14) |
| `results/measured/` | Real result JSON/CSV (includes the two archived biased runs); `results/sim/` simulated; `results/raw/` gitignored traces |

[`01-inference-loop`](../01-inference-loop/) is used unchanged via [_day1.py](_day1.py) (appends `../01-inference-loop` to `sys.path`).

## Run

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -m pytest --cov=.                       # 139 tests, CPU only, no model download
.venv/bin/python profile_run.py --smoke --device cpu --out /tmp/p   # tiny-model smoke run
.venv/bin/python profile_run.py --device mps --tag mps-baseline     # real profile (needs the cached Qwen model)
.venv/bin/python run_local_experiments.py --exp O1 --device mps --tag mps
```

## Limitations

See section 6 of the report. In short: one model on one Apple-silicon machine; MPS profiler gives CPU-side ops only, so device
attribution is indirect (the attention-core figure is an estimate); no NVIDIA measurements; E4-E6 are simulated; Mac thermals/background load were not controlled.
