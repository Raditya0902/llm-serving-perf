#!/usr/bin/env bash
# Round 4, single-Hopper pod (Q3 of PREREGISTRATION_ROUND4.md): does FlashAttention 3 have the mixed-step penalty?
#   bash gpu/run_round4_h100.sh [OUT]
set -uo pipefail
cd "$(dirname "$0")/.."
OUT="$(mkdir -p "${1:-results/raw/round4-h100}" && cd "${1:-results/raw/round4-h100}" && pwd)"
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$OUT/driver.log"; }
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv > "$OUT/gpu.txt" 2>&1
mkdir -p "$OUT/sweep"
log "microbench: ratio table for FA2 and FA3, 0.5B and 8B geometry"
for fa in 2 3; do
  [[ -s "$OUT/sweep/fa${fa}_0.5b.json" ]] || python -m gpu.attn_sweep --fa-version "$fa" --out "$OUT/sweep/fa${fa}_0.5b.json" > /dev/null 2> "$OUT/sweep/fa${fa}_err.log"
  [[ -s "$OUT/sweep/fa${fa}_8b.json" ]] || python -m gpu.attn_sweep --fa-version "$fa" --model llama3.1-8b --ctx 512 3584 --decodes 64 256 \
    --out "$OUT/sweep/fa${fa}_8b.json" > /dev/null 2>> "$OUT/sweep/fa${fa}_err.log"
done
log "end to end, 0.5B: FA3, FA2 and Triton; 256 seqs at two workloads, 1024 seqs at 512x128 (18 runs, trimmed to fit the budget)"
BACKENDS="FLASH_ATTN:3 FLASH_ATTN:2 TRITON_ATTN" REPS=2 SEQS=256 WORKLOADS="512x128 3584x128" bash gpu/run_backend_ab.sh "$OUT/ab/qwen2.5-0.5b/s256"
BACKENDS="FLASH_ATTN:3 FLASH_ATTN:2 TRITON_ATTN" REPS=2 SEQS=1024 WORKLOADS="512x128" bash gpu/run_backend_ab.sh "$OUT/ab/qwen2.5-0.5b/s1024"
touch "$OUT/DONE"; log "DONE"
