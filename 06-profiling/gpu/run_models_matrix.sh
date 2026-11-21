#!/usr/bin/env bash
# Q2 of PREREGISTRATION_ROUND4.md: does the FlashAttention-2 mixed-step penalty and the TRITON_ATTN gain hold at other GQA ratios?
# FLASH_ATTN vs TRITON_ATTN on Qwen2.5-1.5B / 3B and Llama-3.1-8B, unprofiled, fresh server per run (gpu/run_backend_ab.sh).
#
#   bash gpu/run_models_matrix.sh [OUT_ROOT]     # REPS=2 ONLY="qwen2.5-3b" to run one model; analysis: python -m gpu.backend_ab_summary OUT_ROOT
#
# Groups are "slug|hf id|max-num-batched-tokens|max-num-seqs|workloads". 3584-token prompts run at a seq limit whose demand fits the KV pool
# (3B: 128 x 3,712 tokens); the 8B's pool (about 175k tokens) is why it runs 64 and 192 seqs, not 256 and 1024.
set -uo pipefail
cd "$(dirname "$0")/.."
OUT="$(mkdir -p "${1:-results/raw/models-matrix}" && cd "${1:-results/raw/models-matrix}" && pwd)"
REPS="${REPS:-2}"
ONLY="${ONLY:-}"
GROUPS_LIST=(
  "qwen2.5-1.5b|Qwen/Qwen2.5-1.5B-Instruct|8192|256|512x128 3584x128"
  "qwen2.5-1.5b|Qwen/Qwen2.5-1.5B-Instruct|8192|1024|512x128"
  "qwen2.5-3b|Qwen/Qwen2.5-3B-Instruct|8192|256|512x128"
  "qwen2.5-3b|Qwen/Qwen2.5-3B-Instruct|8192|128|3584x128"
  "qwen2.5-3b|Qwen/Qwen2.5-3B-Instruct|8192|1024|512x128"
  "llama3.1-8b|NousResearch/Meta-Llama-3.1-8B-Instruct|2048|64|512x128"
  "llama3.1-8b|NousResearch/Meta-Llama-3.1-8B-Instruct|2048|192|512x128"
)
# PROFILE=h100: the groups for an 80 GB Hopper pod (round 5): a larger 8B KV pool allows 192 and 512 seqs; set BACKENDS="FLASH_ATTN:3 FLASH_ATTN:2 TRITON_ATTN".
if [[ "${PROFILE:-}" == "h100" ]]; then
  GROUPS_LIST=(
    "qwen2.5-1.5b|Qwen/Qwen2.5-1.5B-Instruct|8192|256|512x128 3584x128"
    "qwen2.5-1.5b|Qwen/Qwen2.5-1.5B-Instruct|8192|1024|512x128"
    "qwen2.5-3b|Qwen/Qwen2.5-3B-Instruct|8192|1024|512x128"
    "llama3.1-8b|NousResearch/Meta-Llama-3.1-8B-Instruct|2048|192|512x128"
    "llama3.1-8b|NousResearch/Meta-Llama-3.1-8B-Instruct|2048|512|512x128"
  )
fi
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$OUT/driver.log"; }
for group in "${GROUPS_LIST[@]}"; do
  IFS='|' read -r slug model bt seqs workloads <<< "$group"
  [[ -n "$ONLY" && "$slug" != "$ONLY" ]] && continue
  log "GROUP $slug seqs=$seqs workloads=$workloads"
  MODEL="$model" BT="$bt" SEQS="$seqs" WORKLOADS="$workloads" REPS="$REPS" bash gpu/run_backend_ab.sh "$OUT/$slug/s$seqs" \
    || log "group failed: $slug s$seqs"
done
touch "$OUT/DONE"
