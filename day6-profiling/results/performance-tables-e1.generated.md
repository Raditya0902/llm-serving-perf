# Generated tables (do not hand-edit; interpretation lives in performance-report.md)

## Baseline vs modified

| Metric | Baseline (E1:1) | Modified (E1:8) | Difference |
|---|---|---|---|
| Requests/sec | 0.858 | 2.676 | +1.818 (+211.8%) |
| Output tokens/sec | 54.9 | 171.3 | +116.4 (+211.8%) |
| P95 TTFT (ms) | 9503.2 | 2735.9 | -6767 (-71.2%) |
| P95 TPOT (ms) | 22.86 | 24.98 | +2.121 (+9.3%) |
| Peak memory (MB) | 967.2 | 1024.1 | +56.88 (+5.9%) |

Requests/sec ranges across repeats do not overlap with the baseline.
Token match vs baseline: 1.00.

## E1

| Value | req/s | out tok/s | P95 TTFT ms | P95 TPOT ms | peak MB | tokens match | req/s vs base |
|---|---|---|---|---|---|---|---|
| 1 (baseline) | 0.858 | 54.9 | 9503.2 | 22.86 | 967.2 | 1.00 | +0.0% |
| 2 | 1.284 | 82.2 | 6205.8 | 29.70 | 974.2 | 1.00 | +49.6% |
| 4 | 1.997 | 127.8 | 3948.7 | 23.66 | 988.4 | 1.00 | +132.6% |
| 8 | 2.676 | 171.3 | 2735.9 | 24.98 | 1024.1 | 1.00 | +211.8% |

WARNING: wall-time spread across repeats exceeds 20% for 1 (52%), 2 (48%), 4 (43%), 8 (54%). Treat P95 values and small differences from these rows as unreliable; repeat the run.
