# Day 6: Profiling and optimization experiments

> **Status.** Measured on an Apple M4 (MPS) for the Day 1 engine: profiling, experiments E1-E3, and
> two optimizations (O1, O2). **Not run: anything needing an NVIDIA GPU** (nsys, vLLM profiling, GPU peak memory).
> **E4-E6 are SIMULATED** (Day 3 simulator, illustrative constants), not measurements. See [performance-report.md](performance-report.md).

## Headline findings (details and caveats in the report)

- Decode (Qwen2.5-0.5B, fp16, MPS) is **device-bound on weight reads**: matmuls 74% of a 15.9 ms step, effective 51-93 GB/s. The CPU enqueues a step in 3.0 ms, so Day 1's "launch overhead" guess was wrong.
- Removing Day 1's per-step host syncs: **+6.6% throughput / -7% median TPOT**, identical tokens, validated against an A/A control (-0.1%).
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
| [profile_run.py](profile_run.py) | Profiles one Day 1 generation (prefill and decode windows); writes trace, op table, summary |
| [module_timing.py](module_timing.py) | Device-inclusive attribution where the profiler has no device timeline (matmul replay + sync bracketing) |
| [launch_probe.py](launch_probe.py), [alloc_probe.py](alloc_probe.py) | CPU-enqueue vs step time; per-step allocator growth |
| [trace_analysis.py](trace_analysis.py), [op_classify.py](op_classify.py) | Gap/alloc analysis of Chrome traces; operator categories |
| [variants.py](variants.py), [measure.py](measure.py) | Decode variants (day1, deferred sync, batched); interleaved repeats, request-level metrics, baseline deltas |
| [run_local_experiments.py](run_local_experiments.py) | Runs E1-E3, O1, O2 with controls and token validation |
| [sim_predictions.py](sim_predictions.py) | SIMULATED E4-E6 from the Day 3 simulator |
| [make_performance_report.py](make_performance_report.py), [plot_results.py](plot_results.py) | Generated tables and figures |
| `gpu/` | `run_nsys.sh`, `gpu_mem_sampler.py`, `report_from_vllm.py`: written for a GPU host, tested with stand-ins only |
| `results/measured/` | Real result JSON/CSV (includes the two archived biased runs); `results/sim/` simulated; `results/raw/` gitignored traces |

Day 1 is used unchanged via [_day1.py](_day1.py) (appends `../day1-inference` to `sys.path`).

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
