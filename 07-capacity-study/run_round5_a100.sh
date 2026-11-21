#!/usr/bin/env bash
# Round 5, one multi-A100 host (2x A100 40GB PCIe, or 8x A100 40GB SXM4 NVLink; only GPUs 0 and 1 are used): the same 8B arms on both hosts,
# so the interconnect is the variable. Llama-3.1-8B (ungated mirror), workload B, uncapped real 40 GB pools.
#   STEPS="probe tp dp p2poff"   probe = topology + all-reduce latency/bandwidth; tp = TP=1 / TP=2 / TP=2 without custom all-reduce at c16 and c64;
#                                dp = two TP=1 replicas; p2poff = the TP=2 no-custom arms with NCCL_P2P_DISABLE=1 (the NVLink host through host memory)
#   REPS=2  OUT=results/raw/round5-a100/<host tag>      run from 07-capacity-study with the vLLM venv active, in tmux
#   bash run_round5_a100.sh results/raw/round5-a100/sxm4
set -uo pipefail
cd "$(dirname "$0")"
OUT="$(mkdir -p "${1:-results/raw/round5-a100}" && cd "${1:-results/raw/round5-a100}" && pwd)"
STEPS="${STEPS:-probe tp dp}"
REPS="${REPS:-2}"; HW="${HW:-a100-40gb}"
MODEL="${MODEL:-NousResearch/Meta-Llama-3.1-8B-Instruct}"; PRESET="llama3-8b"; PORT=18000
export CUDA_VISIBLE_DEVICES=0,1
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$OUT/driver.log"; }
have() { [[ " $STEPS " == *" $1 "* ]]; }
COMMON=(--hardware "$HW" --model "$MODEL" --model-preset "$PRESET" --port "$PORT" --reps "$REPS")
{ nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader | head -2; nvidia-smi topo -m | head -4
  python -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__)"; } > "$OUT/env.txt" 2>&1
nvidia-smi topo -m > "$OUT/topo.txt" 2>&1; nvidia-smi nvlink --status > "$OUT/nvlink_status.txt" 2>&1 || true

if have probe; then
  log "probe"
  torchrun --nproc_per_node 2 allreduce_probe.py --out "$OUT/probe_default.json" > "$OUT/probe_default.log" 2>&1 || log "probe failed"
  NCCL_P2P_DISABLE=1 torchrun --nproc_per_node 2 allreduce_probe.py --out "$OUT/probe_p2poff.json" > "$OUT/probe_p2poff.log" 2>&1 || log "probe p2poff failed"
fi
if have tp; then
  log "tp"
  python run_followup_gpu.py --out "$OUT/followup" --arms tp1_c16 tp1_c64 tp2_c16 tp2_c64 tp2nc_c16 tp2nc_c64 "${COMMON[@]}" > "$OUT/tp.log" 2>&1 \
    || log "tp step reported failures (see tp.log)"
fi
if have dp; then
  log "dp"
  python run_dp_gpu.py --out "$OUT/dp" "${COMMON[@]}" > "$OUT/dp.log" 2>&1 || log "dp step reported failures (see dp.log)"
fi
if have p2poff; then
  log "p2poff"
  NCCL_P2P_DISABLE=1 python run_followup_gpu.py --out "$OUT/followup_p2poff" --arms tp2nc_c16 tp2nc_c64 "${COMMON[@]}" > "$OUT/p2poff.log" 2>&1 \
    || log "p2poff step reported failures (see p2poff.log)"
fi
touch "$OUT/DONE"; log "DONE"
