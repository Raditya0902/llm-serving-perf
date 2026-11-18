"""Per-step GPU time vs prefill tokens from vLLM torch-profiler Chrome traces.

Usage: python gpu/trace_step_costs.py <dir containing trace_seq256/ and trace_seq1024/>
Reads the `execute_context_<n>(<prefill tokens>)_generation_<m>(<decodes>)` GPU annotations of rank 0.
Written on the A40 run; no unit tests (it is an analysis script for one-off traces).
"""
import gzip, json, re, glob, sys
import numpy as np
R = sys.argv[1].rstrip("/") + "/"
out = {}
for cfg in (256, 1024):
    f = glob.glob(R + f"trace_seq{cfg}/rank0.*.pt.trace.json.gz")[0]
    ev = json.load(gzip.open(f))["traceEvents"]
    steps = []
    for e in ev:
        n = e.get("name", "")
        m = re.match(r"execute_context_(\d+)\((\d+)\)_generation_(\d+)\((\d+)\)", n)
        if m and e.get("ph") == "X" and e.get("cat") in ("gpu_user_annotation", "user_annotation"):
            steps.append((e["ts"], e["dur"] / 1000, int(m.group(2)), int(m.group(4)), e["cat"]))
    cats = {s[4] for s in steps}
    steps = [s for s in steps if s[4] == ("gpu_user_annotation" if "gpu_user_annotation" in cats else s[4])]
    steps.sort()
    arr = np.array([(s[1], s[2], s[3]) for s in steps])
    pure = arr[arr[:, 1] == 0]; mixed = arr[arr[:, 1] > 0]
    print(f"seq{cfg}: {len(arr)} steps | pure-decode n={len(pure)} mean {pure[:,0].mean() if len(pure) else float('nan'):.1f} ms (decodes {pure[:,2].mean() if len(pure) else 0:.0f}) | "
          f"mixed n={len(mixed)} mean {mixed[:,0].mean() if len(mixed) else float('nan'):.1f} ms")
    if len(mixed) >= 4:
        A = np.c_[np.ones(len(mixed)), mixed[:, 1], mixed[:, 2]]
        c = np.linalg.lstsq(A, mixed[:, 0], rcond=None)[0]
        print(f"   mixed fit: ms = {c[0]:.1f} + {c[1]*1000:.2f}*(prefill tokens/1000) + {c[2]*1000:.2f}*(decodes/1000)")
        for lo, hi in ((1, 1024), (1024, 4096), (4096, 6144), (6144, 9000)):
            sel = mixed[(mixed[:, 1] >= lo) & (mixed[:, 1] < hi)]
            if len(sel): print(f"   prefill tok {lo}-{hi}: n={len(sel)} mean {sel[:,0].mean():.1f} ms, decodes {sel[:,2].mean():.0f}")
    if len(pure) >= 3:
        for lo, hi in ((1, 128), (128, 256), (256, 512), (512, 2000)):
            sel = pure[(pure[:, 2] >= lo) & (pure[:, 2] < hi)]
            if len(sel): print(f"   pure decode batch {lo}-{hi}: n={len(sel)} mean {sel[:,0].mean():.1f} ms")
