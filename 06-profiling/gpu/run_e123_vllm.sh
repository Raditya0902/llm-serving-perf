#!/usr/bin/env bash
# E1-E3 (Day 1 hypotheses, experiments.py) on vLLM at 1.5B / 3B / 8B: one default server per model, `vllm bench serve` at fixed concurrency.
#   E1  concurrency 1,2,4,8 (128 in / 128 out)       E2  TTFT at 128,512,1024,2048 input tokens (8 out, concurrency 1)
#   E3  TPOT at 32,128,256 output tokens (128 in, concurrency 1)
#
#   bash gpu/run_e123_vllm.sh [OUT_ROOT]       # MODELS="qwen2.5-1.5b|Qwen/Qwen2.5-1.5B-Instruct ..." to override; analysis: python -m gpu.e123_summary OUT_ROOT
set -uo pipefail
cd "$(dirname "$0")/.."
ulimit -n 65536 2>/dev/null || true
OUT="$(mkdir -p "${1:-results/raw/e123}" && cd "${1:-results/raw/e123}" && pwd)"
read -r -a MODEL_LIST <<< "${MODELS:-qwen2.5-0.5b|Qwen/Qwen2.5-0.5B-Instruct qwen2.5-1.5b|Qwen/Qwen2.5-1.5B-Instruct qwen2.5-3b|Qwen/Qwen2.5-3B-Instruct llama3.1-8b|NousResearch/Meta-Llama-3.1-8B-Instruct}"
PORT=18000
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$OUT/driver.log"; }

bench() {  # name model concurrency in_len out_len num_prompts
  local name="$1" model="$2" conc="$3" in_len="$4" out_len="$5" n="$6" dir="$7"
  [[ -s "$dir/$name.json" ]] && { log "SKIP $name"; return; }
  vllm bench serve --backend vllm --model "$model" --endpoint /v1/completions --host 127.0.0.1 --port "$PORT" \
    --dataset-name random --random-input-len "$in_len" --random-output-len "$out_len" --num-prompts "$n" --max-concurrency "$conc" \
    --request-rate inf --ignore-eos --seed 0 --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,99 \
    --save-result --result-dir "$dir" --result-filename "$name.json" > "$dir/bench_$name.log" 2>&1 || log "bench failed: $name"
}

for entry in "${MODEL_LIST[@]}"; do
  slug="${entry%%|*}"; model="${entry#*|}"; dir="$OUT/$slug"; mkdir -p "$dir"
  log "MODEL $slug"
  if (echo > "/dev/tcp/127.0.0.1/${PORT}") 2>/dev/null; then log "port $PORT in use"; exit 1; fi
  setsid vllm serve "$model" --max-model-len 4096 --gpu-memory-utilization 0.85 --seed 0 --host 127.0.0.1 --port "$PORT" > "$dir/server.log" 2>&1 &
  server=$!
  waited=0; healthy=1
  until curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; do
    kill -0 "$server" 2>/dev/null || { log "server died"; healthy=0; break; }
    (( waited >= 900 )) && { log "not healthy"; healthy=0; break; }
    sleep 3; waited=$((waited + 3))
  done
  if (( healthy )); then
    vllm bench serve --backend vllm --model "$model" --endpoint /v1/completions --host 127.0.0.1 --port "$PORT" --dataset-name random \
      --random-input-len 128 --random-output-len 32 --num-prompts 16 --max-concurrency 4 --request-rate inf --ignore-eos --seed 1 > "$dir/warmup.log" 2>&1
    for c in 1 2 4 8; do bench "e1_c$c" "$model" "$c" 128 128 $((c == 1 ? 24 : 6 * c)) "$dir"; done
    for n in 128 512 1024 2048; do bench "e2_in$n" "$model" 1 "$n" 8 20 "$dir"; done
    for n in 32 128 256; do bench "e3_out$n" "$model" 1 128 "$n" 12 "$dir"; done
  fi
  kill -TERM -- "-$server" 2>/dev/null
  waited=0; while kill -0 "$server" 2>/dev/null && (( waited < 90 )); do sleep 1; waited=$((waited + 1)); done
  kill -KILL -- "-$server" 2>/dev/null; sleep 5
done
touch "$OUT/DONE"
