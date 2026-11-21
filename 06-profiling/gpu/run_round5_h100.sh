#!/usr/bin/env bash
# Round 5, one H100 80GB (SXM5) pod: counters on FA3 vs FA2, busy fraction at 256 vs 1,024 seqs for FA3 / FA2 / Triton, the Triton output
# check (default = FlashAttention 3 on Hopper) at 0.5B-8B, an untraced A/B bracket, and the backend matrix at 1.5B / 3B / 8B.
# Run inside tmux from 06-profiling with the vLLM venv active and PATH containing nsys 2026.5.2 (gpu/setup_lambda.sh prints how).
#   bash gpu/run_round5_h100.sh [OUT]     # STEPS="ncu quality busyfrac ab models" to choose; each step skips finished work
set -uo pipefail
cd "$(dirname "$0")/.."
OUT="$(mkdir -p "${1:-results/raw/round5-h100}" && cd "${1:-results/raw/round5-h100}" && pwd)"
STEPS="${STEPS:-ncu quality busyfrac ab models}"
HOPPER_BACKENDS="FLASH_ATTN:3 FLASH_ATTN:2 TRITON_ATTN"
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$OUT/driver.log"; }
have() { [[ " $STEPS " == *" $1 "* ]]; }
{ nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
  python -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__, torch.version.cuda)"; ncu --version | tail -1; nsys --version; } > "$OUT/env.txt" 2>&1

if have ncu; then      # FA3 and FA2 (forced) kernel counters on pure decode / mixed / prefill-only
  log "ncu"; mkdir -p "$OUT/ncu"
  sudo -E env "PATH=$PATH" ncu --section SpeedOfLight --launch-count 1 python -c "import torch; a=torch.randn(2048,2048,device='cuda'); (a@a).sum().item()" > "$OUT/ncu_gate.txt" 2>&1
  if grep -q ERR_NVGPUCTRPERM "$OUT/ncu_gate.txt"; then log "counters BLOCKED"; else
    for fa in 3 2; do for model in qwen2.5-0.5b llama3.1-8b; do for ctx in 512 3584; do
      d="$OUT/ncu/fa${fa}_${model}_ctx${ctx}"; [[ -s "$d/mixed.csv" ]] && continue
      FA_VERSION="$fa" NCU_SUDO=1 MODEL="$model" CTX="$ctx" DECODES=256 PREFILL=7600 bash gpu/run_ncu.sh "$d" > "$d.log" 2>&1 || log "ncu failed fa$fa $model ctx$ctx"
      python -m gpu.ncu_summary "$d/pure_decode.csv" "$d/mixed.csv" "$d/prefill_only.csv" > "$d/summary.json" 2> "$d/summary.err"
    done; done; done
  fi
fi
if have quality; then
  for entry in "qwen2.5-0.5b|Qwen/Qwen2.5-0.5B-Instruct" "qwen2.5-1.5b|Qwen/Qwen2.5-1.5B-Instruct" "qwen2.5-3b|Qwen/Qwen2.5-3B-Instruct" "llama3.1-8b|NousResearch/Meta-Llama-3.1-8B-Instruct"; do
    slug="${entry%%|*}"; model="${entry#*|}"; q="$OUT/quality/$slug"; mkdir -p "$q"; log "quality $slug"
    [[ -s "$q/fa2.json" ]] || python -m gpu.quality_run --model "$model" --backend FLASH_ATTN --bt 2048 --out "$q/fa2.json" > "$q/fa2.log" 2>&1   # default FA generation (FA3); file name kept for quality_compare
    [[ -s "$q/triton.json" ]] || python -m gpu.quality_run --model "$model" --backend TRITON_ATTN --bt 2048 --out "$q/triton.json" > "$q/triton.log" 2>&1
    [[ -s "$q/fa2_bt8192.json" ]] || python -m gpu.quality_run --model "$model" --backend FLASH_ATTN --bt 8192 --out "$q/fa2_bt8192.json" > "$q/fa2_bt8192.log" 2>&1
    [[ -s "$q/reference.json" ]] || python -m gpu.quality_run --model "$model" --reference --out "$q/reference.json" > "$q/reference.log" 2>&1
    python -m gpu.quality_compare "$q" > "$q/verdict.json" 2> "$q/verdict.err"
  done
fi
if have busyfrac; then log "busyfrac"; BACKENDS="$HOPPER_BACKENDS" bash gpu/run_busyfrac.sh "$OUT/busyfrac"; python -m gpu.busyfrac_summary "$OUT/busyfrac" > "$OUT/busyfrac/summary.json"; fi
if have ab; then
  log "ab"
  for seqs in 256 1024; do
    BACKENDS="$HOPPER_BACKENDS" NUM_PROMPTS=4000 REPS=2 SEQS="$seqs" WORKLOADS="128x128 512x128" bash gpu/run_backend_ab.sh "$OUT/ab/qwen2.5-0.5b/s$seqs"
  done
fi
if have models; then log "models"; PROFILE=h100 BACKENDS="$HOPPER_BACKENDS" REPS=2 bash gpu/run_models_matrix.sh "$OUT/models"; fi
touch "$OUT/DONE"; log "DONE"
