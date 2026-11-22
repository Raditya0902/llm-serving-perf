#!/usr/bin/env bash
# Capture an nsys trace of ONE profiled window of the `01-inference-loop` decode loop (NVIDIA + Linux only).
#
# STATUS: run on RunPod (A40, nsys 2026.5.2 installed with `apt-get install nsight-systems-2026.5.2` from the CUDA repo).
# `--sample=none --cpuctxsw=none` are needed in a container: CPU sampling and context-switch tracing want perf privileges.
#
# profile_run.py --nsys brackets its window with torch.cuda.profiler.start/stop (cudaProfilerApi) and
# labels attention/MLP with NVTX ranges, so --capture-range=cudaProfilerApi records only that window
# (warmup excluded). --nsys skips torch.profiler so Kineto/CUPTI cannot clash with nsys.
#
#   bash gpu/run_nsys.sh [extra profile_run.py args]
#   e.g. bash gpu/run_nsys.sh --phase decode --prompt-len 512 --new-tokens 33
set -euo pipefail

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "refusing: nsys capture needs a Linux NVIDIA host (this is $(uname -s))" >&2; exit 2
fi
command -v nvidia-smi >/dev/null || { echo "refusing: nvidia-smi not found" >&2; exit 2; }
command -v nsys >/dev/null || { echo "refusing: nsys not found (install Nsight Systems)" >&2; exit 2; }

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${NSYS_OUT:-$HERE/results/raw/nsys/inference_trace}"
mkdir -p "$(dirname "$OUT")"

nsys profile \
  --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  --force-overwrite=true \
  -o "$OUT" \
  python "$HERE/profile_run.py" --device cuda --nsys --tag nsys "$@"

echo "wrote $OUT.nsys-rep ; inspect with: nsys stats $OUT.nsys-rep"
