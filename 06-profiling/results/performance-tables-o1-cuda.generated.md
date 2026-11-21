# Generated tables (do not hand-edit; interpretation lives in performance-report.md)

## Baseline vs modified

| Metric | Baseline (O1:day1) | Modified (O1:deferred_sync) | Difference |
|---|---|---|---|
| Requests/sec | 0.959 | 0.984 | +0.0249 (+2.6%) |
| Output tokens/sec | 61.4 | 63.0 | +1.593 (+2.6%) |
| P95 TTFT (ms) | 17.8 | 17.8 | +0.05505 (+0.3%) |
| P95 TPOT (ms) | 16.41 | 15.92 | -0.4838 (-2.9%) |
| Peak memory (MB) | 982.2 | 982.2 | +0 (+0.0%) |

Requests/sec ranges across repeats do not overlap with the baseline.
Token match vs baseline: 1.00.

## O1

| Value | req/s | out tok/s | P95 TTFT ms | P95 TPOT ms | peak MB | tokens match | req/s vs base |
|---|---|---|---|---|---|---|---|
| day1 (baseline) | 0.959 | 61.4 | 17.8 | 16.41 | 982.2 | 1.00 | +0.0% |
| day1_control | 0.953 | 61.0 | 17.8 | 16.85 | 982.2 | 1.00 | -0.6% |
| deferred_sync | 0.984 | 63.0 | 17.8 | 15.92 | 982.2 | 1.00 | +2.6% |

A/A control `day1_control` (identical work) differs from the baseline by -0.6% requests/s and +2.7% P95 TPOT: that is the noise floor, so smaller differences between the baseline and the modified row are not evidence of an effect.
