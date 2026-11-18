# Day 7 matrix on a real GPU: results and verdicts

Measured on one NVIDIA A40 (RunPod), vLLM 0.34.0, `Llama-3.1-8B-Instruct` (ungated mirror `NousResearch/Meta-Llama-3.1-8B-Instruct`,
`--model-preset llama3-8b`). Generated tables: [results/gpu/a40-capped/tables.generated.md](results/gpu/a40-capped/tables.generated.md).
This replaces the "never executed" status of [RUNBOOK.md](RUNBOOK.md). [REPORT.md](REPORT.md) is still the **simulated** analysis; this file
scores its predictions.

## What was run

| | |
|---|---|
| Matrix | 12 cases (workload A/B x concurrency 4/16/64 x baseline/tuned), 3 repeats, 5 for the four concurrency-64 cases, plus a 5-run A/A control: **49 fresh servers, 0 failed requests, 0 runs rejected by `ingest_gpu.py`** |
| Hardware handling | The A40 has 46 GiB, the simulator modeled an A100-40GB. Runs used `--hardware a100-40gb --cap-kv-pool`, so the server got `--kv-cache-memory-bytes` equal to the modeled pool (17.0 GiB baseline, 19.8 GiB tuned) and the memory-pressure regime is reproduced. **Timings are A40 timings** (0.696 TB/s, not the modeled 1.555 TB/s), so compare relative deltas, not absolute milliseconds |
| Follow-up arms | B at concurrency 64, one knob at a time (`run_followup_gpu.py`): `pool092` (pool 17.0 -> 19.8 GiB only), `budget2048` (`max_num_batched_tokens` 8192 -> 2048 only), admission caps `cap16/24/32`. 3 repeats for the first two, **1 repeat** for the caps |
| Noise floor (A/A) | output tok/s 0.81%, P95 e2e 0.81%, mean TTFT 3.43%, mean TPOT 0.44% |

## Verdicts on the pre-registered predictions (RUNBOOK section 4)

| # | Prediction | Verdict | Evidence |
|---|---|---|---|
| H1 | Only B@64 reaches >= 90% KV with preemptions, both configs; A never | **Held** | B@64 baseline: KV 100%, 64.2 preemptions (5 reps); tuned: 100%, 56.6. Every one of the other 10 cases: 0 preemptions, KV peak 1-35%. |
| H2 | `tuned` cuts B@64 mean TTFT >= 25% and raises tok/s >= 3%, TPOT up | **Half refuted** | TTFT -43.1% and TPOT +21.8% as predicted, but **tok/s -4.8%** (CV 0.4-0.5%, so real). The decomposition below shows why. |
| H3 | `tuned` = baseline on tok/s at every feasible cell | **Held** | A-c4 127/127, A-c16 427/428, A-c64 1,065/1,065, B-c4 110/110, B-c16 278/276 (-0.7%, inside the noise floor). Small real TTFT differences of opposite sign exist (A-c64 -6.3%, B-c16 +5.1%; floor 3.4%), peak GPU memory is +6% under tuned (a bigger pool). |
| H4 | At c=4, TPOT <= 1.5x the weight-read floor | **Borderline; depends on the hardware used for the floor** | Against the A40's floor (23.1 ms): A-c4 30.5 ms = 1.32x (pass), B-c4 34.7 ms = 1.50x (34.65 ms threshold: on the line). Against the modeled A100 floor (10.3 ms) it fails at 3x, but that floor does not apply to this GPU. TPOT at c=4 includes other requests' prefill chunks, which the check does not separate. |
| H5 | B@64 P95 e2e exceeds the limit `L` in both configs | **Held** | L = 50,595 ms; B@64 P95 e2e 73,935 (baseline), 79,200 (tuned). The best feasible B cell is B-c16-baseline (27,194 ms). No tested B@64 variant meets L either (cap32 72,683 ms). |
| H6 | Admission caps: TTFT up, preemptions to 0, throughput flat to lower | **Held** (1 rep per cap) | Preemptions 0 at caps 16/24/32. TTFT +330% / +210% / +121%. tok/s -31% / -20% / -8%. KV peak 35% / 52% / 68%. The throughput cost grows as the cap tightens. |

Decision output (`analyze.py`): both workloads **keep the baseline** (A: `tuned` +0.0% tok/s, within noise; B: the best feasible cell is already baseline at c=16).

## Why H2 failed: `tuned` is two knobs, and they pull in opposite directions

B@64, mean of repeats, change versus the baseline (5 reps; baseline 409 tok/s, TTFT 14.4 s, TPOT 114.7 ms):

| arm | n | tok/s | TTFT | TPOT | preemptions |
|---|---|---|---|---|---|
| `tuned` (both knobs) | 5 | -4.8% | -43.1% | +21.8% | 56.6 |
| `pool092` (bigger pool only) | 3 | **+2.4%** | **-48.4%** | +13.4% | 63.3 |
| `budget2048` (token budget only) | 3 | **-6.2%** | **+8.6%** | +6.3% | 59.7 |

The pool change gives the TTFT win; the token-budget change costs throughput and gives nothing on TTFT. The two effects roughly add up to `tuned`
(tok/s +2.4 - 6.2 = -3.8% vs -4.8% measured; TTFT -48.4 + 8.6 = -39.8% vs -43.1%), though their interaction was not tested separately.
The pool arm barely changes the preemption count (63 vs 64) while halving TTFT, so the TTFT gain is **not** explained by fewer preemptions;
a larger resident batch fits (TPOT +13%), but batch size was not measured, so that is a hypothesis.

## Simulator vs measurement (relative change versus each side's baseline, B@64)

| arm | tok/s sim | tok/s real | TTFT sim | TTFT real | TPOT sim | TPOT real | preempt sim | preempt real |
|---|---|---|---|---|---|---|---|---|
| tuned | +8.2% | **-4.8%** | -51.6% | -43.1% | +6.6% | +21.8% | 55 | 57 |
| pool092 (sim: util-only) | +2.8% | +2.4% | -50.2% | -48.4% | +12.6% | +13.4% | 66 | 63 |
| budget2048 (sim: tokens-only) | +5.0% | **-6.2%** | -4.1% | **+8.6%** | -5.0% | **+6.3%** | 56 | 60 |
| cap32 | -2.7% | -8.3% | +113.6% | +120.5% | -34.3% | -28.9% | 0 | 0 |
| cap24 | -11.0% | -19.5% | +185.5% | +210.3% | -46.0% | -38.7% | 0 | 0 |
| cap16 | -23.6% | -31.1% | +294.7% | +330.5% | -57.9% | -52.2% | 0 | 0 |

- **What the simulator got right** (structure): where KV pressure appears (B@64 only), the preemption counts (64.4 simulated vs 64.2 measured for the baseline), peak KV fractions of the non-pressured cases (for example B@16 34.9% vs 35%), the pool-size effect, and the direction and rough size of the admission-cap trade (it underestimated the throughput cost by about 1.3-3x).
- **What it got wrong**: the **token-budget knob has the wrong sign** on throughput (+5.0% vs -6.2%), TTFT and TPOT. Absolute times are also far off (for example A-c64 2,614 simulated vs 1,065 measured tok/s), as expected from modeling an A100 with illustrative constants.
- **Implicated assumption**: the Day 3 cost model charges a fixed `overhead_ms = 0.3` per iteration, and a smaller token budget therefore looks free: it only reduces per-step interference. Day 6 measured per-step costs on this A40 of 7-16 ms for a 0.5B model, so a fixed per-step cost more than 20x larger than 0.3 ms is real, and a 2,048 budget splits each 2,560-token prompt into two chunks. That fits the measured loss and Day 4's independent finding on the same GPU (a smaller token budget cost throughput at saturation), but **this was not isolated here** (no step-level trace of the 8B runs, batch size not measured), so treat it as the leading hypothesis.

## Two-GPU follow-up: the 8B model with `--tensor-parallel-size 2` (RUNBOOK section 5)

Run on a different pod: **2x A100-SXM4-80GB over NVLink (NV12)**, same vLLM 0.34.0 and `Llama-3.1-8B-Instruct` mirror, workload B, `--gpu-memory-utilization 0.85`,
**no KV cap** (`--hardware a100-80gb`), 3 repeats per arm, fresh server each, 0 failed requests. Summary: [results/gpu/a100-tp/summary.md](results/gpu/a100-tp/summary.md).
These are not comparable in absolute terms with the A40 tables above (different GPU, no pool cap); only TP=2 against TP=1 on the same pod is.

| arm | out tok/s (CV) | vs TP=1 | mean TPOT ms | vs TP=1 | mean TTFT ms | vs TP=1 | KV peak | preemptions |
|---|---|---|---|---|---|---|---|---|
| B@16 TP=1 | 721 (0.8%) | - | 20.5 | - | 404 | - | 12% | 0 |
| B@16 TP=2 | 1,166 (0.8%) | **+61.7%** | 12.7 | -38.3% | 251 | -37.8% | 5% | 0 |
| B@64 TP=1 | 1,044 (0.1%) | - | 57.3 | - | 1,050 | - | 45% | 0 |
| B@64 TP=2 | 1,844 (0.2%) | **+76.5%** | 32.3 | -43.6% | 638 | -39.3% | 20% | 0 |
| B@64 TP=2, `--disable-custom-all-reduce` | 1,796 (0.3%) | +71.9% | 33.2 | -42.1% | 630 | -40.0% | 20% | 0 |

Against the pre-run analytic predictions (REPORT section 5; link constants were assumed):

| Prediction | Measured | Verdict |
|---|---|---|
| NVLink speeds a 64-sequence decode step ~1.6x | TPOT 57.3 -> 32.3 ms = **1.77x** at B@64 (1.61x at B@16) | Held (somewhat better than predicted) |
| Group KV capacity ~2.9x | 413,184 -> 952,320 tokens = **2.30x** | Smaller than predicted. The 2.9x was computed for 40 GB cards, where the 15 GiB of weights are a bigger share; on 80 GB the weights are a smaller share, so a smaller ratio is expected (this explanation is arithmetic, not tested on 40 GB). |
| TP=2 stops B@64 from preempting | Neither TP=1 nor TP=2 preempts on 80 GB (KV peak 45% vs 20%) | **Not testable here.** The A40 result (100% KV, 64 preemptions) came from capping the pool to the modeled 40 GB size; this pod ran uncapped. |
| PCIe: step speedup ~1.05x | not run | Not tested |

TP=2 also lowers TTFT by ~39% and the custom all-reduce kernel is worth about +2.7% throughput at B@64 (1,844 vs 1,796 tok/s; small but outside the 0.2-0.3% CV).
The Day 5 1.5B experiment on the same pod (TP=2 +20 to +38% throughput; two TP=1 replicas beat one TP=2 server by 19.5% at concurrency 128) is in
[../day5-distributed/RESULTS.md](../day5-distributed/RESULTS.md); a data-parallel baseline was not run for the 8B model, so "is TP=2 the right use of the second GPU for 8B?" is open.

## Limits and what was not done

- **One GPU, one model, one capped-pool configuration.** The capped pool is a modeling choice that reproduces the simulator's regime; uncapped, the 46 GiB A40 would have a larger pool and (per the simulator) much less pressure. That was not run.
- **Caps are single runs** (n=1); the gaps are 8-31% against a baseline CV of 0.4%, so the direction is safe but the sizes are soft. `budget2048` and `pool092` have 3 repeats.
- The bottleneck labels in the generated tables are rule-based and the tool notes that batch size was not measured, so the "compute / prefill-interference bound" labels at c>=16 are unverified. The A100-floor labels in `results/gpu/a40-capped/` are the wrong floor for this GPU; use `results/gpu/a40-capped-a40bw/` (A40 floor) for H4.
- **Not done**: the PCIe two-GPU case, a TP=2 run with a KV pool capped to the modeled size (the preemption prediction), an 8B data-parallel baseline, 8B step-level profiling, and a single-GPU run with an uncapped pool. (The NVLink TP=2 follow-up is the section above.)
- **Found and fixed while running**: vLLM's "Peak concurrent requests" buckets requests by whole second and read 68 against a client cap of 64, so `ingest_gpu.py` rejected a valid run. The guard now computes exact overlap from `start_times` and `latencies` (2 new tests).

## Reproduce

```bash
python run_matrix_gpu.py --out results/raw/a40-capped --reps 3 --key-reps 5 --hardware a100-40gb --cap-kv-pool \
    --model NousResearch/Meta-Llama-3.1-8B-Instruct --model-preset llama3-8b
python ingest_gpu.py --raw results/raw/a40-capped --out results/gpu/a40-capped --hardware a100-40gb --cap-kv-pool \
    --model NousResearch/Meta-Llama-3.1-8B-Instruct --model-preset llama3-8b
python analyze.py --dir results/gpu/a40-capped --hardware a100-40gb --model-preset llama3-8b      # simulator's hardware
python analyze.py --dir results/gpu/a40-capped-a40bw --hardware a40-48gb --model-preset llama3-8b  # H4 on the A40 floor
python run_followup_gpu.py --out results/raw/followup --hardware a100-40gb --cap-kv-pool --model ... --model-preset llama3-8b
python summarize_followup.py --matrix-raw results/raw/a40-capped --followup-raw results/raw/followup

# 2-GPU pod, no --cap-kv-pool (tensor-parallel arms):
python run_followup_gpu.py --out results/raw/followup-tp --arms tp1_c16 tp2_c16 tp1_c64 tp2_c64 tp2nc_c64 --hardware a100-80gb --model NousResearch/Meta-Llama-3.1-8B-Instruct --model-preset llama3-8b
```

Raw runs (`results/raw/`) are git-ignored; ingested summaries are in `results/gpu/`.
