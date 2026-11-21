#!/usr/bin/env bash
# Round 4, single-A40 pod: everything in PREREGISTRATION_ROUND4.md except the Hopper items, in value order so that a stop at any point
# leaves complete results (sweep2 = the Q4b follow-up, run last so it cannot disturb a benchmark). Run inside tmux from 06-profiling with the vLLM venv active; results go to results/raw/round4-a40/.
#   bash gpu/run_round4_a40.sh [OUT]            # STEPS="quality sweep noise models e123" to choose; each step skips finished work
set -uo pipefail
cd "$(dirname "$0")/.."
OUT="$(mkdir -p "${1:-results/raw/round4-a40}" && cd "${1:-results/raw/round4-a40}" && pwd)"
STEPS="${STEPS:-quality sweep noise models e123 sweep2}"
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$OUT/driver.log"; }
have() { [[ " $STEPS " == *" $1 "* ]]; }

if have quality; then   # Q1: FlashAttention 2 vs Triton output equivalence, with the same-kernel noise floor and an fp32 reference
  log "quality"; q="$OUT/quality"; mkdir -p "$q"
  [[ -s "$q/fa2.json" ]] || python -m gpu.quality_run --backend FLASH_ATTN --bt 2048 --out "$q/fa2.json" > "$q/fa2.log" 2>&1
  [[ -s "$q/triton.json" ]] || python -m gpu.quality_run --backend TRITON_ATTN --bt 2048 --out "$q/triton.json" > "$q/triton.log" 2>&1
  [[ -s "$q/fa2_bt8192.json" ]] || python -m gpu.quality_run --backend FLASH_ATTN --bt 8192 --out "$q/fa2_bt8192.json" > "$q/fa2_bt8192.log" 2>&1
  [[ -s "$q/reference.json" ]] || python -m gpu.quality_run --reference --out "$q/reference.json" > "$q/reference.log" 2>&1
  python -m gpu.quality_compare "$q" > "$q/verdict.json" 2> "$q/verdict.err"; log "quality verdict written"
fi
if have sweep; then     # Q4: the ratio of 18 (FlashAttention 2, 0.5B geometry); then the same grid for the 8B geometry's head size
  log "sweep"; mkdir -p "$OUT/sweep"
  [[ -s "$OUT/sweep/fa2_0.5b.json" ]] || python -m gpu.attn_sweep --out "$OUT/sweep/fa2_0.5b.json" > /dev/null 2> "$OUT/sweep/err.log"
  [[ -s "$OUT/sweep/fa2_8b.json" ]] || python -m gpu.attn_sweep --model llama3.1-8b --ctx 512 3584 --decodes 64 256 --out "$OUT/sweep/fa2_8b.json" > /dev/null 2>> "$OUT/sweep/err.log"
fi
if have noise; then     # Q5a: the 128x128 cell with 4,000 prompts per run, 6 repeats, both backends, both seq limits
  log "noise"
  for seqs in 256 1024; do
    NUM_PROMPTS=4000 REPS=6 SEQS="$seqs" WORKLOADS="128x128" bash gpu/run_backend_ab.sh "$OUT/noise/qwen2.5-0.5b/s$seqs"
  done
fi
if have models; then log "models"; bash gpu/run_models_matrix.sh "$OUT/models"; fi   # Q2
if have e123; then log "e123"; bash gpu/run_e123_vllm.sh "$OUT/e123"; fi              # Q5b
if have sweep2; then    # Q4b (pre-registered after the first sweep): fit on ctx 512, predict ctx 1024 and 2048 at decode counts 128 and 384
  log "sweep2"
  [[ -s "$OUT/sweep/fa2_0.5b_q4b.json" ]] || python -m gpu.attn_sweep --ctx 512 1024 2048 --decodes 64 128 256 384 512 --out "$OUT/sweep/fa2_0.5b_q4b.json" > /dev/null 2>> "$OUT/sweep/err.log"
fi
touch "$OUT/DONE"; log "DONE"
