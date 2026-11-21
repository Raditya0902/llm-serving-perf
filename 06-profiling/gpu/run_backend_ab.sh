#!/usr/bin/env bash
# Unprofiled A/B of the attention backend under a mixed (prefill + decode) load: FLASH_ATTN (FlashAttention 2 on an A40, the default)
# against TRITON_ATTN. performance-report 9.10 found that FA2 loses its GQA decode shortcut in mixed batches; this measures what the
# backend switch is worth end to end, without the torch profiler's overhead. Fresh server per run, repeat-major order.
#
#   bash gpu/run_backend_ab.sh [OUT_DIR]       # REPS=3 WORKLOADS="3584x128 512x128" SEQS=256
#
# Knobs: MODEL, BT=8192 (--max-num-batched-tokens), MAXLEN=4096, NUM_PROMPTS=1000 (4000 gives a steady run for the 128x128 cell).
# A BACKENDS entry NAME:N also forces the FlashAttention generation (`--attention-config.flash_attn_version N`), e.g. on a Hopper
# GPU BACKENDS="FLASH_ATTN:3 FLASH_ATTN:2 TRITON_ATTN"; its result files are tagged NAME-faN (FLASH_ATTN-fa3_512x128_r0.json).
set -uo pipefail
cd "$(dirname "$0")/.."
ulimit -n 65536 2>/dev/null || true   # 4,000 prompts sent at once need 4,000 sockets; the default of 1,024 failed 75% of them
MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
OUT="$(mkdir -p "${1:-results/raw/backend-ab}" && cd "${1:-results/raw/backend-ab}" && pwd)"
REPS="${REPS:-3}"; WORKLOADS="${WORKLOADS:-3584x128 512x128}"; SEQS="${SEQS:-256}"; BACKENDS="${BACKENDS:-FLASH_ATTN TRITON_ATTN}"
BT="${BT:-8192}"; MAXLEN="${MAXLEN:-4096}"; NUM_PROMPTS="${NUM_PROMPTS:-1000}"
PORT=18000
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$OUT/driver.log"; }

for rep in $(seq 0 $((REPS - 1))); do
  for workload in $WORKLOADS; do
    for entry in $BACKENDS; do
      backend="${entry%%:*}"; fa_args=()
      label="$backend"
      if [[ "$entry" == *:* ]]; then fa_args=(--attention-config.flash_attn_version "${entry#*:}"); label="${backend}-fa${entry#*:}"; fi
      in_len="${workload%x*}"; out_len="${workload#*x}"
      tag="${label}_${workload}_r${rep}"
      [[ -s "$OUT/$tag.json" ]] && grep -q '"failed": 0,' "$OUT/$tag.json" && { log "SKIP $tag (done)"; continue; }
      log "RUN $tag"
      if (echo > "/dev/tcp/127.0.0.1/${PORT}") 2>/dev/null; then log "port $PORT in use"; exit 1; fi
      setsid vllm serve "$MODEL" --max-model-len "$MAXLEN" --gpu-memory-utilization 0.85 --max-num-seqs "$SEQS" --max-num-batched-tokens "$BT" \
        --attention-backend "$backend" ${fa_args[@]+"${fa_args[@]}"} --seed 0 --host 127.0.0.1 --port "$PORT" > "$OUT/server_$tag.log" 2>&1 &
      server=$!
      waited=0; healthy=1
      until curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; do
        kill -0 "$server" 2>/dev/null || { log "server died"; healthy=0; break; }
        (( waited >= 900 )) && { log "not healthy"; healthy=0; break; }
        sleep 3; waited=$((waited + 3))
      done
      if (( healthy )); then
        vllm bench serve --backend vllm --model "$MODEL" --endpoint /v1/completions --host 127.0.0.1 --port "$PORT" \
          --dataset-name random --random-input-len "$in_len" --random-output-len "$out_len" --num-prompts "$NUM_PROMPTS" --request-rate inf \
          --ignore-eos --seed 0 --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,99 \
          --save-result --result-dir "$OUT" --result-filename "$tag.json" > "$OUT/bench_$tag.log" 2>&1 || log "bench failed for $tag"
      fi
      kill -TERM -- "-$server" 2>/dev/null
      waited=0; while kill -0 "$server" 2>/dev/null && (( waited < 90 )); do sleep 1; waited=$((waited + 1)); done
      kill -KILL -- "-$server" 2>/dev/null; sleep 5
    done
  done
done
touch "$OUT/DONE"
