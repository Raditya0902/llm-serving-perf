#!/usr/bin/env bash
# Isolate the max_num_seqs=1024 throughput loss (performance-report 9.3, 9.8): a 2x4 grid of seqs {256, 1024} x workloads, plus seqs 512
# at the baseline workload as a dose-response point. Fresh server per run, repeat-major order so slow drift does not line up with one arm.
# 0.5B, 1000 prompts at once (rate inf), max_num_batched_tokens 8192, --ignore-eos, seed 0. A40 / vLLM 0.34.0.
#
#   source ../day4-vllm-internals/.venv-gpu/bin/activate
#   bash gpu/run_seqs_isolation.sh [OUT_DIR]      # REPS=3; analysis: python -m gpu.seqs_isolation OUT_DIR
#   ARMS="256:512x128 1024:512x128" REPS=1 bash gpu/run_seqs_isolation.sh /tmp/smoke      # quick check first
#
# ARMS is a space-separated list of SEQS:INxOUT. Result files are seqs<S>_<IN>x<OUT>_r<R>.json (what gpu/seqs_isolation.py reads).
set -uo pipefail
cd "$(dirname "$0")/.."
ulimit -n 65536 2>/dev/null || true
MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
OUT="$(mkdir -p "${1:-results/raw/seqs-isolation}" && cd "${1:-results/raw/seqs-isolation}" && pwd)"
REPS="${REPS:-3}"
ARMS="${ARMS:-256:512x128 1024:512x128 256:128x128 1024:128x128 256:512x512 1024:512x512 256:128x512 1024:128x512 512:512x128}"
NUM_PROMPTS="${NUM_PROMPTS:-1000}"
PORT=18000
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$OUT/driver.log"; }

for rep in $(seq 0 $((REPS - 1))); do
  for arm in $ARMS; do
    seqs="${arm%%:*}"; workload="${arm#*:}"; in_len="${workload%x*}"; out_len="${workload#*x}"
    tag="seqs${seqs}_${workload}_r${rep}"
    [[ -s "$OUT/$tag.json" ]] && { log "SKIP $tag (done)"; continue; }
    log "RUN $tag"
    if (echo > "/dev/tcp/127.0.0.1/${PORT}") 2>/dev/null; then log "port $PORT in use"; exit 1; fi
    setsid vllm serve "$MODEL" --max-model-len 4096 --gpu-memory-utilization 0.85 --max-num-seqs "$seqs" --max-num-batched-tokens 8192 \
      --seed 0 --host 127.0.0.1 --port "$PORT" > "$OUT/server_$tag.log" 2>&1 &
    server=$!
    waited=0; healthy=1
    until curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; do
      kill -0 "$server" 2>/dev/null || { log "server died"; healthy=0; break; }
      (( waited >= 900 )) && { log "not healthy"; healthy=0; break; }
      sleep 3; waited=$((waited + 3))
    done
    if (( healthy )); then
      ( while true; do echo "$(date +%s.%N) $(curl -s "http://127.0.0.1:${PORT}/metrics" | grep -E '^vllm:(kv_cache_usage_perc|num_preemptions_total|num_requests_running|num_requests_waiting)' | tr '\n' ' ')"; sleep 0.2; done ) > "$OUT/metrics_ts_$tag.txt" &
      sampler=$!
      vllm bench serve --backend vllm --model "$MODEL" --endpoint /v1/completions --host 127.0.0.1 --port "$PORT" \
        --dataset-name random --random-input-len "$in_len" --random-output-len "$out_len" --num-prompts "$NUM_PROMPTS" --request-rate inf \
        --ignore-eos --seed 0 --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,99 \
        --save-result --result-dir "$OUT" --result-filename "$tag.json" > "$OUT/bench_$tag.log" 2>&1 || log "bench failed for $tag"
      kill "$sampler" 2>/dev/null; wait "$sampler" 2>/dev/null
    fi
    kill -TERM -- "-$server" 2>/dev/null
    waited=0; while kill -0 "$server" 2>/dev/null && (( waited < 90 )); do sleep 1; waited=$((waited + 1)); done
    kill -KILL -- "-$server" 2>/dev/null; sleep 5
  done
done
touch "$OUT/DONE"
