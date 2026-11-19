#!/usr/bin/env bash
# RUNBOOK 3a: share of GPU time in all-reduce for TP=2 serving, from an Nsight Systems capture of a steady-state window.
#
#   source ../day4-vllm-internals/.venv-gpu/bin/activate
#   ./run_nsys_tp2.sh [OUT_DIR]             # ARMS="custom nocustom" C=8 WINDOW_S=6 by default
#
# vLLM runs with `--profiler-config.profiler=cuda`, so POST /start_profile and /stop_profile call cudaProfilerStart/Stop in each
# worker; nsys (`--capture-range=cudaProfilerApi`) records only that window while `vllm bench serve` keeps the server busy.
# Needs two GPUs and nsys. Never use these runs for throughput (profiling overhead); they only attribute GPU time.
set -uo pipefail
cd "$(dirname "$0")"
MODEL="${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
OUT="${1:-results/raw/nsys_tp2}"
ARMS="${ARMS:-custom nocustom}"
C="${C:-8}"
WINDOW_S="${WINDOW_S:-6}"
WARM_S="${WARM_S:-25}"
PORT=18000        # not 8001: RunPod's nginx owns it
HEALTH_TIMEOUT_S="${HEALTH_TIMEOUT_S:-600}"
mkdir -p "$OUT"
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$OUT/driver.log"; }
command -v nsys >/dev/null || { echo "nsys not found" >&2; exit 2; }

for arm in $ARMS; do
  tag="tp2_${arm}_c${C}"
  extra=()
  [[ "$arm" == "nocustom" ]] && extra=(--disable-custom-all-reduce)
  if (echo > "/dev/tcp/127.0.0.1/${PORT}") 2>/dev/null; then log "port $PORT already in use"; exit 1; fi
  log "ARM $tag"
  setsid nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none --cuda-graph-trace=node \
    --capture-range=cudaProfilerApi --capture-range-end=stop --force-overwrite=true -o "$OUT/$tag" \
    vllm serve "$MODEL" --max-model-len 4096 --gpu-memory-utilization 0.85 --tensor-parallel-size 2 \
    --host 127.0.0.1 --port "$PORT" --profiler-config.profiler=cuda "${extra[@]}" > "$OUT/server_${tag}.log" 2>&1 &
  server=$!
  waited=0
  until curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 && curl -sf "http://127.0.0.1:${PORT}/v1/models" | grep -q "$MODEL"; do
    if ! kill -0 "$server" 2>/dev/null; then log "server died during startup"; break; fi
    if (( waited >= HEALTH_TIMEOUT_S )); then log "not healthy after ${HEALTH_TIMEOUT_S}s"; break; fi
    sleep 3; waited=$((waited + 3))
  done
  vllm bench serve --backend vllm --model "$MODEL" --endpoint /v1/completions --host 127.0.0.1 --port "$PORT" \
    --dataset-name random --random-input-len 512 --random-output-len 128 --ignore-eos --num-prompts 3000 \
    --num-warmups "$C" --max-concurrency "$C" --request-rate inf --seed 100 > "$OUT/bench_${tag}.log" 2>&1 &
  bench=$!
  sleep "$WARM_S"
  curl -s -X POST "http://127.0.0.1:${PORT}/start_profile" > "$OUT/start_${tag}.txt" 2>&1
  sleep "$WINDOW_S"
  curl -s -X POST "http://127.0.0.1:${PORT}/stop_profile" > "$OUT/stop_${tag}.txt" 2>&1
  kill -TERM "$bench" 2>/dev/null; wait "$bench" 2>/dev/null
  kill -TERM -- "-$server" 2>/dev/null
  waited=0; while kill -0 "$server" 2>/dev/null && (( waited < 120 )); do sleep 1; waited=$((waited + 1)); done  # nsys finalizes the report
  kill -KILL -- "-$server" 2>/dev/null; sleep 5
  if [[ -f "$OUT/$tag.nsys-rep" ]]; then
    nsys export --type=sqlite --force-overwrite=true -o "$OUT/$tag.sqlite" "$OUT/$tag.nsys-rep" > "$OUT/export_${tag}.log" 2>&1 \
      && python analyze_nsys.py "$OUT/$tag.sqlite" > "$OUT/${tag}_summary.json" \
      && log "$tag: all-reduce share of busy GPU time = $(python -c "import json;print(json.load(open('$OUT/${tag}_summary.json'))['allreduce_share_of_busy'])")"
  else
    log "$tag: no .nsys-rep produced (see server_${tag}.log)"
  fi
done
touch "$OUT/DONE"
