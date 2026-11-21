# Generated tables (do not hand-edit; interpretation lives in performance-report.md)

## Baseline vs modified

| Metric | Baseline (O1:day1) | Modified (O1:deferred_sync) | Difference |
|---|---|---|---|
| Requests/sec | 0.880 | 0.938 | +0.05786 (+6.6%) |
| Output tokens/sec | 56.3 | 60.0 | +3.703 (+6.6%) |
| P95 TTFT (ms) | 161.0 | 159.7 | -1.377 (-0.9%) |
| P95 TPOT (ms) | 15.83 | 14.58 | -1.247 (-7.9%) |
| Peak memory (MB) | 967.2 | 967.2 | +0.01538 (+0.0%) |

Requests/sec ranges across repeats do not overlap with the baseline.
Token match vs baseline: 1.00.

## O1

| Value | req/s | out tok/s | P95 TTFT ms | P95 TPOT ms | peak MB | tokens match | req/s vs base |
|---|---|---|---|---|---|---|---|
| day1 (baseline) | 0.880 | 56.3 | 161.0 | 15.83 | 967.2 | 1.00 | +0.0% |
| day1_control | 0.879 | 56.2 | 159.7 | 16.63 | 967.2 | 1.00 | -0.1% |
| deferred_sync | 0.938 | 60.0 | 159.7 | 14.58 | 967.2 | 1.00 | +6.6% |

A/A control `day1_control` (identical work) differs from the baseline by -0.1% requests/s and +5.1% P95 TPOT: that is the noise floor, so smaller differences between the baseline and the modified row are not evidence of an effect.
