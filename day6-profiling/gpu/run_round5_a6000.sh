#!/usr/bin/env bash
# Round 5, single-GPU Ampere pod (A6000: GA102 like the A40): items A (Nsight Compute counters), B (busy fraction) and C (Triton output
# check at 1.5B / 3B / 8B) plus an untraced A/B bracket, in value order, each step skipping finished work. Run inside tmux from day6-profiling, vLLM venv active.
#   bash gpu/run_round5_a6000.sh [OUT]       # STEPS="ncu busyfrac quality" to choose; counter step is a gate: it stops the step on ERR_NVGPUCTRPERM
set -uo pipefail
cd "$(dirname "$0")/.."
OUT="$(mkdir -p "${1:-results/raw/round5-a6000}" && cd "${1:-results/raw/round5-a6000}" && pwd)"
STEPS="${STEPS:-ncu ab busyfrac quality}"
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$OUT/driver.log"; }
have() { [[ " $STEPS " == *" $1 "* ]]; }
{ nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader; nvidia-smi topo -m | head -5
  python -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__, torch.version.cuda)"; ncu --version | tail -1; nsys --version; } > "$OUT/env.txt" 2>&1

mkdir -p "$OUT/ncu"
if have ncu; then      # A: counters. Gate first (a tiny matmul), then the attention kernel on pure decode / mixed / prefill-only at the 0.5B and 8B geometries
  log "ncu"
  if [[ ! -s "$OUT/ncu_gate.txt" ]]; then
    sudo -E env "PATH=$PATH" ncu --section SpeedOfLight --launch-count 1 python -c "import torch; a=torch.randn(2048,2048,device='cuda'); (a@a).sum().item()" > "$OUT/ncu_gate.txt" 2>&1
  fi
  if grep -q ERR_NVGPUCTRPERM "$OUT/ncu_gate.txt"; then
    log "counters BLOCKED (ERR_NVGPUCTRPERM): try 'sudo -E env PATH=\$PATH ncu ...' or reload the driver with NVreg_RestrictProfilingToAdminUsers=0"
  else
    for model in qwen2.5-0.5b llama3.1-8b; do
      for ctx in 512 3584; do
        d="$OUT/ncu/${model}_ctx${ctx}"; [[ -s "$d/mixed.csv" ]] && continue
        NCU_SUDO=1 MODEL="$model" CTX="$ctx" DECODES=256 PREFILL=7600 bash gpu/run_ncu.sh "$d" > "$d.log" 2>&1 || log "ncu failed for $model ctx$ctx"
        python -m gpu.ncu_summary "$d/pure_decode.csv" "$d/mixed.csv" "$d/prefill_only.csv" > "$d/summary.json" 2> "$d/summary.err"
      done
    done
  fi
fi
if have ab; then       # does the A6000 reproduce the A40 backend A/B at 0.5B? (untraced bracket for the traced busyfrac runs)
  log "ab"
  for seqs in 256 1024; do
    NUM_PROMPTS=4000 REPS=3 SEQS="$seqs" WORKLOADS="128x128 512x128" bash gpu/run_backend_ab.sh "$OUT/ab/qwen2.5-0.5b/s$seqs"
  done
fi
if have busyfrac; then log "busyfrac"; bash gpu/run_busyfrac.sh "$OUT/busyfrac"; python -m gpu.busyfrac_summary "$OUT/busyfrac" > "$OUT/busyfrac/summary.json"; fi
if have quality; then  # C: Triton vs FA2 vs same-kernel noise floor vs fp32 reference, per model (reference needs the GPU alone, so it runs last per model)
  for entry in "qwen2.5-1.5b|Qwen/Qwen2.5-1.5B-Instruct" "qwen2.5-3b|Qwen/Qwen2.5-3B-Instruct" "llama3.1-8b|NousResearch/Meta-Llama-3.1-8B-Instruct"; do
    slug="${entry%%|*}"; model="${entry#*|}"; q="$OUT/quality/$slug"; mkdir -p "$q"; log "quality $slug"
    [[ -s "$q/fa2.json" ]] || python -m gpu.quality_run --model "$model" --backend FLASH_ATTN --bt 2048 --out "$q/fa2.json" > "$q/fa2.log" 2>&1
    [[ -s "$q/triton.json" ]] || python -m gpu.quality_run --model "$model" --backend TRITON_ATTN --bt 2048 --out "$q/triton.json" > "$q/triton.log" 2>&1
    [[ -s "$q/fa2_bt8192.json" ]] || python -m gpu.quality_run --model "$model" --backend FLASH_ATTN --bt 8192 --out "$q/fa2_bt8192.json" > "$q/fa2_bt8192.log" 2>&1
    [[ -s "$q/reference.json" ]] || python -m gpu.quality_run --model "$model" --reference --out "$q/reference.json" > "$q/reference.log" 2>&1
    python -m gpu.quality_compare "$q" > "$q/verdict.json" 2> "$q/verdict.err"
  done
fi
touch "$OUT/DONE"; log "DONE"
