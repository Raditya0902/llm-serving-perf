# Generated tables (do not hand-edit; interpretation lives in performance-report.md)

## Baseline vs modified

| Metric | Baseline (O2:sdpa) | Modified (O2:eager) | Difference |
|---|---|---|---|
| Requests/sec | 0.882 | 0.667 | -0.2152 (-24.4%) |
| Output tokens/sec | 56.5 | 42.7 | -13.77 (-24.4%) |
| P95 TTFT (ms) | 161.5 | 199.7 | +38.19 (+23.7%) |
| P95 TPOT (ms) | 16.13 | 21.14 | +5.016 (+31.1%) |
| Peak memory (MB) | 1947.2 | 1947.2 | +0 (+0.0%) |

Requests/sec ranges across repeats do not overlap with the baseline.
Token match vs baseline: 1.00.

## O2

| Value | req/s | out tok/s | P95 TTFT ms | P95 TPOT ms | peak MB | tokens match | req/s vs base |
|---|---|---|---|---|---|---|---|
| sdpa (baseline) | 0.882 | 56.5 | 161.5 | 16.13 | 1947.2 | 1.00 | +0.0% |
| eager | 0.667 | 42.7 | 199.7 | 21.14 | 1947.2 | 1.00 | -24.4% |
| sdpa_control | 0.882 | 56.4 | 160.0 | 15.99 | 1947.2 | 1.00 | -0.1% |

A/A control `sdpa_control` (identical work) differs from the baseline by -0.1% requests/s and -0.8% P95 TPOT: that is the noise floor, so smaller differences between the baseline and the modified row are not evidence of an effect.
