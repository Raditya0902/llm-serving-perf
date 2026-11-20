#!/usr/bin/env bash
# Nsight Compute counters for the attention kernel on a pure-decode batch and a mixed batch (gpu/attn_microbench.py).
# Needs `ncu` and permission to read GPU performance counters; without it ncu prints ERR_NVGPUCTRPERM and this script says so.
#
#   bash gpu/run_ncu.sh [OUT_DIR]        # MODEL=qwen2.5-0.5b CTX=3584 DECODES=256 PREFILL=7600
#   python gpu/ncu_summary.py OUT_DIR/pure_decode.csv OUT_DIR/mixed.csv
#
# Each range is profiled in its own ncu run with one warm-up and one measured launch, so replay cost stays small (ncu re-runs every
# kernel once per counter pass). The filter `-k regex:flash` keeps only the attention kernel.
set -uo pipefail
cd "$(dirname "$0")/.."
command -v ncu >/dev/null || { echo "refusing: ncu not found (install nsight-compute from the CUDA apt repo)" >&2; exit 2; }
OUT="$(mkdir -p "${1:-results/raw/ncu}" && cd "${1:-results/raw/ncu}" && pwd)"
FA_VERSION="${FA_VERSION:-2}"; MODEL="${MODEL:-qwen2.5-0.5b}"; CTX="${CTX:-3584}"; DECODES="${DECODES:-256}"; PREFILL="${PREFILL:-7600}"
SECTIONS=(--section SpeedOfLight --section MemoryWorkloadAnalysis --section Occupancy --section LaunchStats --section WarpStateStats)
# Named counters that ncu_summary.py reads, requested explicitly so a missing section cannot silently drop them.
METRICS="gpu__time_duration.sum,dram__bytes_read.sum,launch__grid_size,launch__block_size,launch__waves_per_multiprocessor,sm__warps_active.avg.pct_of_peak_sustained_active,gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed"
NCU=(ncu); [[ -n "${NCU_SUDO:-}" ]] && NCU=(sudo -E env "PATH=$PATH" ncu)   # counters are admin-only on Lambda VMs (RmProfilingAdminOnly=1)
status=0
for batch in pure_decode mixed prefill_only; do
  "${NCU[@]}" "${SECTIONS[@]}" --metrics "$METRICS" --nvtx --nvtx-include "${batch}/" --kernel-name-base demangled -k "regex:flash_fwd|FlashAttn" --launch-skip 2 --launch-count 2 \
      --csv --page raw --log-file "$OUT/$batch.csv" \
      python gpu/attn_microbench.py --model "$MODEL" --ctx "$CTX" --decodes "$DECODES" --prefill "$PREFILL" \
      --fa-version "$FA_VERSION" --batches "$batch" --iters 3 --warmup 1 --out "$OUT/$batch.timing.json" > "$OUT/$batch.stdout" 2>&1 || status=1
  grep -l ERR_NVGPUCTRPERM "$OUT/$batch.csv" "$OUT/$batch.stdout" 2>/dev/null && { echo "counters blocked (ERR_NVGPUCTRPERM)" >&2; exit 3; }
done
exit $status
