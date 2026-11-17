# RUNBOOK: run the Day 7 matrix on a Linux + NVIDIA host

**Status: never executed.** Written on a Mac with no GPU. The orchestration is tested with stand-ins (`tests/test_run_matrix_gpu.py`);
the real `vllm serve` / `vllm bench serve` path reuses Day 4's `default_backend()` and was not exercised. Expect to fix small things on
the first run, and treat any surprise as a finding to record, not to hide.

## 0. Prerequisites

- Linux, one idle NVIDIA GPU (the sampler reports whole-GPU memory), CUDA drivers, `nvidia-smi` on `PATH`.
- ~16 GB of disk for the model; a Hugging Face token if you keep the default gated model (`meta-llama/Llama-3.1-8B-Instruct`). Any model
  with the same architecture works via `--model`; **`--model-preset` must match its architecture** (it sizes KV and the simulated cost).
- vLLM `0.34.0`, as in Day 4: `day4-vllm-internals/setup_gpu_host.sh` creates `.venv-gpu` and writes `results/env.txt`.

```bash
cd day7-capstone
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt
nvidia-smi                                   # confirm the GPU is idle; note its name and memory
```

## 1. Match the hardware profile

`--hardware` selects the **simulated** comparison profile (`a100-40gb`, `a10g-24gb`). If your GPU is neither, add a `HardwareProfile` to
`kv_sizing.HARDWARE` (memory bytes, bandwidth, peak FLOPS) *before* running, so the predictions in section 4 are for your GPU. The real
run does not depend on this; the comparison does.

## 2. Check the commands without running anything

```bash
.venv/bin/python verify_flags.py                                  # every flag exists in vLLM v0.34.0 source (needs vllm_src)
.venv/bin/python run_matrix_gpu.py --dry-run --out results/raw/gpu | less
```

A full run is 12 cases x 3 repeats, with 5 for the four concurrency-64 cases, plus a 5-run A/A control: **49 fresh servers**. Budget a few
hours (model load dominates the short cases).

## 3. Run, ingest, analyze

```bash
.venv/bin/python run_matrix_gpu.py --out results/raw/<tag> --reps 3 --key-reps 5 --model <hub-id> --model-preset <preset>
.venv/bin/python ingest_gpu.py   --raw results/raw/<tag> --out results/gpu/<tag> --model <hub-id> --model-preset <preset>
.venv/bin/python analyze.py      --dir results/gpu/<tag> --model-preset <preset> --hardware <profile>
.venv/bin/python plot_matrix.py  --dir results/gpu/<tag>
.venv/bin/python make_report.py  --dir results/gpu/<tag>
```

- The runner resumes: finished runs are skipped only if their `matrix_config_id` matches; `--force` redoes them; `--only <case_id>...` limits it.
- `ingest_gpu.py` **rejects** failed or partial runs, observed concurrency above the target, and results from a different configuration, and
  exits non-zero listing them. Fix or re-run those before trusting any average.
- `results/raw/` is git-ignored (bench JSON with `--save-detailed` is large). Commit `results/gpu/<tag>/` after checking it.
- If `kv_peak_frac` / `preemptions` / `peak_waiting` come back "not measured", the server did not expose the v0.34.0 metric names: check
  `curl -s localhost:8000/metrics | grep -E 'kv_cache|preempt|requests_waiting'` and update `prom_metrics.py`.

## 4. What the real run should confirm or refute

These are the simulator's predictions, written down before any GPU data exists. **Structural** ones (where pressure appears) should survive a
change of hardware; **magnitude** ones may not.

| # | Prediction (from `results/sim/`) | Kind | Refuted if |
|---|---|---|---|
| H1 | Only workload B at concurrency 64 reaches >= 90% KV use with preemptions > 0, in both configs (the gauge is sampled every 0.5 s, so it may read a little under 100%); A never does | structural | any other case preempts, or B@64 does not |
| H2 | `tuned` cuts B@64 mean TTFT by >= 25% and raises output tok/s by >= 3%, with mean TPOT up | magnitude | TTFT change < 25%, or tok/s down |
| H3 | `tuned` vs baseline on output tok/s is within the A/A noise floor at every cell that meets the latency limit | structural | a real (non-overlapping, above-floor) gain or loss at a feasible cell |
| H4 | At concurrency 4 the decode step (TPOT) is within 1.5x of one full weight read (`weights / HBM bandwidth`), i.e. bandwidth-bound | magnitude | TPOT > 1.5x the weight-read floor at c=4 |
| H5 | B@64 P95 e2e exceeds the limit `L` for both configs, so the feasible optimum is at lower concurrency | structural | B@64 meets `L` |
| H6 | Admission caps trade preemption for queueing: TTFT up, preemptions to 0, throughput roughly flat to lower | structural | caps raise throughput materially |

H6 is not part of the 12-case matrix; reproduce it by starting a baseline server with `--max-num-seqs 16|24|32` and benchmarking B at
`--max-concurrency 64` using the commands from `--dry-run`.

Record, for every prediction: held / refuted / inconclusive, with the cases and the noise floor behind it. A refuted prediction is the
most useful output of this run: say which simulator assumption it points at.

## 5. Two-GPU follow-up (not automated)

`matrix.py` has no tensor-parallel knob. To test the multi-GPU claims in REPORT.md section 5, take the B@64 baseline serve command from
`--dry-run`, add `--tensor-parallel-size 2`, and run B at concurrency 64 and 16 against it. Capture `nvidia-smi topo -m` first (NVLink vs
PCIe decides the result) and see `day5-distributed/RUNBOOK.md` for NCCL checks and `--disable-custom-all-reduce` comparison. Predictions
(analytic, link constants assumed): TP=2 over NVLink speeds a 64-sequence decode step ~1.6x and raises group KV capacity ~2.9x so B@64
stops preempting; over PCIe the step speedup is ~1.05x (capacity gain unchanged); across a 100 GbE link TP is slower than one GPU.

## 6. Teardown and hygiene

The runner stops each server's whole process group (vLLM's engine and workers can outlive the API process). After an interruption run
`nvidia-smi` and kill any leftover `VLLM::EngineCore` / `VLLM::Worker` processes before re-running; the runner refuses a port that is
already in use rather than benchmark a server it did not start.

Known gaps, found in review and **not fixed** (the teardown code is Day 4's):

- **A second Ctrl-C / SIGTERM during teardown** can interrupt the 30 s graceful wait and skip the SIGKILL stage, leaving the server's process
  group holding the GPU. Press Ctrl-C once and wait; check `nvidia-smi` afterwards.
- **SIGKILL or an OOM-kill of the runner itself** orphans the server (it runs in its own session). The next run's port check then refuses to start.
- **Port race on shared hosts.** The port is checked, then the server takes seconds to bind; another local user could answer `/health` first. The runner
  re-checks that its own server is still alive after `/health` answers, which catches the common case; do not run this on a shared GPU host.
- **Proxy variables.** The runner adds loopback to `NO_PROXY` for itself and its children so `vllm bench serve` never routes through an HTTP proxy, but
  it leaves the proxy settings in place (the server may need them to download weights).
- **Run order** is a fixed-seed shuffle per round (reproducible), not a counterbalanced ABBA design; with 3 repeats it reduces but does not remove order effects.
