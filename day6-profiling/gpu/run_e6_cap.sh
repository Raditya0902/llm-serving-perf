#!/usr/bin/env bash
# RUNBOOK 5 / E6 with a pool the workload can actually exhaust. Day 4's gpu_memory_utilization knob never got the KV pool below the
# workload's peak demand (pool 3-19x demand), so preemptions were never triggered. Here `--kv-cache-memory-bytes` shrinks the pool directly.
# Baseline load: 1000 prompts, 512 in / 128 out, all at once, max_num_seqs 256 => peak demand ~164k tokens x 12 KiB/token ~ 2.0 GB of KV.
#
#   bash gpu/run_e6_cap.sh [OUT_DIR]          # CAPS_MIB="4096 2560 2048 1536 1024 512" REPS=1
set -uo pipefail
cd "$(dirname "$0")/.."
MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
OUT="${1:-results/raw/e6-cap}"
CAPS_MIB="${CAPS_MIB:-4096 2560 2048 1536 1024 512}"
REPS="${REPS:-1}"
PORT=18000
mkdir -p "$OUT"
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$OUT/driver.log"; }

for rep in $(seq 0 $((REPS - 1))); do
  for mib in $CAPS_MIB; do
    tag="cap${mib}MiB_r${rep}"
    log "RUN $tag"
    if (echo > "/dev/tcp/127.0.0.1/${PORT}") 2>/dev/null; then log "port $PORT in use"; exit 1; fi
    setsid vllm serve "$MODEL" --max-model-len 4096 --max-num-seqs 256 --max-num-batched-tokens 8192 --seed 0 \
      --kv-cache-memory-bytes $((mib * 1024 * 1024)) --host 127.0.0.1 --port "$PORT" > "$OUT/server_$tag.log" 2>&1 &
    server=$!
    waited=0
    until curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; do
      kill -0 "$server" 2>/dev/null || { log "server died"; break; }
      (( waited >= 900 )) && { log "not healthy"; break; }
      sleep 3; waited=$((waited + 3))
    done
    ( while true; do echo "$(date +%s.%N) $(curl -s "http://127.0.0.1:${PORT}/metrics" | grep -E '^vllm:(kv_cache_usage_perc|num_preemptions_total|num_requests_running|num_requests_waiting)' | tr '\n' ' ')"; sleep 0.2; done ) > "$OUT/metrics_ts_$tag.txt" &
    sampler=$!
    vllm bench serve --backend vllm --model "$MODEL" --endpoint /v1/completions --host 127.0.0.1 --port "$PORT" \
      --dataset-name random --random-input-len 512 --random-output-len 128 --num-prompts 1000 --request-rate inf \
      --ignore-eos --seed 0 --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,95,99 \
      --save-result --result-dir "$OUT" --result-filename "$tag.json" > "$OUT/bench_$tag.log" 2>&1 || log "bench failed for $tag"
    curl -s "http://127.0.0.1:${PORT}/metrics" > "$OUT/metrics_end_$tag.txt"
    kill "$sampler" 2>/dev/null; wait "$sampler" 2>/dev/null
    grep -m1 "GPU KV cache size" "$OUT/server_$tag.log" | tee -a "$OUT/driver.log"
    kill -TERM -- "-$server" 2>/dev/null
    waited=0; while kill -0 "$server" 2>/dev/null && (( waited < 90 )); do sleep 1; waited=$((waited + 1)); done
    kill -KILL -- "-$server" 2>/dev/null; sleep 5
  done
done
touch "$OUT/DONE"
