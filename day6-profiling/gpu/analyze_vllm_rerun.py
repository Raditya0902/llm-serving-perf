"""Per-token ITL distribution + live /metrics peaks for the saved `--save-detailed` vLLM bench runs.

Usage: python gpu/analyze_vllm_rerun.py <dir with *_rN.json and *.metrics_ts.txt>
Per-step GPU costs come from trace_step_costs.py.
"""
import gzip, json, glob, re, sys
import numpy as np
R = sys.argv[1].rstrip("/") + "/"
def metrics(id):
    rows = []
    for l in open(R + id + ".metrics_ts.txt"):
        t = l.split()
        if len(t) < 9: continue
        rows.append((float(t[0]), float(t[2]), float(t[4]), float(t[6]), float(t[8])))
    a = np.array(rows); return a
print("== per-run summary (1000-prompt burst)")
for f in sorted(glob.glob(R + "*_r?.json*")):
    id = f.split("/")[-1].split(".json")[0]
    d = json.load(gzip.open(f) if f.endswith(".gz") else open(f)); itls = np.array([x for r in d["itls"] for x in r]) * 1000
    m = metrics(id)
    print(f"{id:11s} tok/s={d['output_throughput']:7.0f} dur={d['duration']:5.1f}s failed={d['failed']} "
          f"ITL p50/p90/p95/p99={np.percentile(itls,50):5.1f}/{np.percentile(itls,90):5.1f}/{np.percentile(itls,95):5.1f}/{np.percentile(itls,99):6.1f} "
          f"peak running={m[:,1].max():.0f} peak waiting={m[:,2].max():.0f} peak KV={m[:,3].max()*100:.1f}% preempt={m[:,4].max():.0f}")
    print("            ITL>15ms %.1f%%  >50ms %.1f%%  >100ms %.1f%%  share-of-ITL-time in >30ms samples %.0f%%" % (
        np.mean(itls > 15) * 100, np.mean(itls > 50) * 100, np.mean(itls > 100) * 100, itls[itls > 30].sum() / itls.sum() * 100))
