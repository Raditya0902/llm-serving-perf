#!/usr/bin/env bash
# Why is p50 ITL 234 ms at input_len=3584 (Day 4) when the step-cost rule predicts ~100 ms?
# Fresh Day 4 baseline server per (input_len, window delay) with the torch profiler window, then a 1000-prompt burst with
# `--save-detailed` so per-token ITLs are available next to the per-step GPU costs. A40 / vLLM 0.34.0 / Qwen2.5-0.5B.
#
#   source ../day4-vllm-internals/.venv-gpu/bin/activate
#   bash gpu/run_itl_study.sh [OUT_DIR]       # LENS="3584 512" DELAYS="135 500" by default; analysis: python -m gpu.step_costs <trace dirs>
#
# Knobs (all optional; the defaults reproduce the round-2 runs and their tag names):
#   MODEL (0.5B)  SEQS=256 (--max-num-seqs)  BT=8192 (--max-num-batched-tokens)  OUTLEN=128  MAXLEN=4096 (--max-model-len)
#   EXTRA_SERVE="--attention-backend TRITON_ATTN"   extra `vllm serve` flags, e.g. --enforce-eager
#   e.g. 8B: MODEL=NousResearch/Meta-Llama-3.1-8B-Instruct SEQS=128 BT=2048 LENS=512 DELAYS="150 400" bash gpu/run_itl_study.sh ...
# A non-default MODEL / SEQS / EXTRA_SERVE is added to every tag so runs never overwrite each other.
set -uo pipefail
cd "$(dirname "$0")/.."
MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
OUT="$(mkdir -p "${1:-results/raw/itl-study}" && cd "${1:-results/raw/itl-study}" && pwd)"   # torch_profiler_dir must be absolute
LENS="${LENS:-3584 512}"
DELAYS="${DELAYS:-135 500}"
SEQS="${SEQS:-256}"
BT="${BT:-8192}"
OUTLEN="${OUTLEN:-128}"
MAXLEN="${MAXLEN:-4096}"
read -r -a EXTRA_SERVE_ARGS <<< "${EXTRA_SERVE:-}"
PREFIX=""
[[ "$MODEL" != "Qwen/Qwen2.5-0.5B-Instruct" ]] && PREFIX="$(basename "$MODEL" | tr 'A-Z' 'a-z')_"
[[ "$SEQS" != "256" ]] && PREFIX="${PREFIX}s${SEQS}_"
[[ -n "${EXTRA_SERVE:-}" ]] && PREFIX="${PREFIX}$(echo "$EXTRA_SERVE" | tr -c 'A-Za-z0-9\n' '_' | sed 's/__*/_/g;s/^_//;s/_$//')_"
PORT=18000
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$OUT/driver.log"; }

for len in $LENS; do
  for delay in $DELAYS; do
    tag="${PREFIX}len${len}_d${delay}"
    log "RUN $tag"
    if (echo > "/dev/tcp/127.0.0.1/${PORT}") 2>/dev/null; then log "port $PORT in use"; exit 1; fi
    setsid vllm serve "$MODEL" --max-model-len "$MAXLEN" --gpu-memory-utilization 0.85 --max-num-seqs "$SEQS" --max-num-batched-tokens "$BT" \
      --seed 0 --host 127.0.0.1 --port "$PORT" ${EXTRA_SERVE_ARGS[@]+"${EXTRA_SERVE_ARGS[@]}"} \
      --profiler-config.profiler=torch --profiler-config.torch_profiler_dir="$OUT/trace_$tag" \
      --profiler-config.torch_profiler_with_stack=false \
      --profiler-config.delay_iterations="$delay" --profiler-config.max_iterations=100 > "$OUT/server_$tag.log" 2>&1 &
    server=$!
    waited=0
    until curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; do
      kill -0 "$server" 2>/dev/null || { log "server died"; break; }
      (( waited >= 900 )) && { log "not healthy"; break; }
      sleep 3; waited=$((waited + 3))
    done
    grep -m1 "GPU KV cache size" "$OUT/server_$tag.log" | tee -a "$OUT/driver.log"
    vllm bench serve --backend vllm --model "$MODEL" --endpoint /v1/completions --host 127.0.0.1 --port "$PORT" \
      --dataset-name random --random-input-len "$len" --random-output-len "$OUTLEN" --num-prompts 1000 --request-rate inf \
      --ignore-eos --seed 0 --profile --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,95,99 \
      --save-result --save-detailed --result-dir "$OUT" --result-filename "$tag.json" > "$OUT/bench_$tag.log" 2>&1 \
      || log "bench failed for $tag"
    sleep 5
    kill -TERM -- "-$server" 2>/dev/null
    waited=0; while kill -0 "$server" 2>/dev/null && (( waited < 90 )); do sleep 1; waited=$((waited + 1)); done
    kill -KILL -- "-$server" 2>/dev/null; sleep 5
    gzip -f "$OUT/$tag.json"
    ls "$OUT/trace_$tag" 2>/dev/null | head -3 | tee -a "$OUT/driver.log"
  done
done
touch "$OUT/DONE"
