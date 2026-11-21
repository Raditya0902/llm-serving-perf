# Generated tables (do not hand-edit; interpretation lives in performance-report.md)

## Baseline vs modified

| Metric | Baseline (E1:1) | Modified (E1:8) | Difference |
|---|---|---|---|
| Requests/sec | 0.950 | 7.248 | +6.298 (+662.7%) |
| Output tokens/sec | 60.8 | 463.9 | +403 (+662.7%) |
| P95 TTFT (ms) | 7382.1 | 60.3 | -7322 (-99.2%) |
| P95 TPOT (ms) | 19.37 | 17.07 | -2.3 (-11.9%) |
| Peak memory (MB) | 982.2 | 1148.5 | +166.2 (+16.9%) |

Requests/sec ranges across repeats do not overlap with the baseline.
Token match vs baseline: 1.00.

## E1

| Value | req/s | out tok/s | P95 TTFT ms | P95 TPOT ms | peak MB | tokens match | req/s vs base |
|---|---|---|---|---|---|---|---|
| 1 (baseline) | 0.950 | 60.8 | 7382.1 | 19.37 | 982.2 | 1.00 | +0.0% |
| 2 | 1.892 | 121.1 | 3226.3 | 16.83 | 1007.0 | 1.00 | +99.1% |
| 4 | 3.688 | 236.0 | 1135.6 | 17.13 | 1056.5 | 1.00 | +288.1% |
| 8 | 7.248 | 463.9 | 60.3 | 17.07 | 1148.5 | 1.00 | +662.7% |
