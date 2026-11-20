#!/usr/bin/env bash
# Round 5, item B: GPU busy fraction at 256 vs 1024 running sequences, for FlashAttention 2 and TRITON_ATTN, from an Nsight Systems
# trace that records only a steady-state window (cuda trace, no CPU sampling, no torch profiler). Separates per-step host cost (H2:
# idle gaps that grow with running sequences) from the attention penalty (device time that grows, no extra idle). One fresh server
# per (arm, delay); the window starts `delay` engine iterations after the benchmark begins and lasts ITERS iterations.
#
#   bash gpu/run_busyfrac.sh [OUT_DIR]        # ARMS="256:512x128 1024:512x128 256:128x128 1024:128x128" DELAYS="200 600" ITERS=150
#   python -m gpu.busyfrac_summary OUT_DIR
# BACKENDS entries NAME:N force the FlashAttention generation (FLASH_ATTN:3 on Hopper), tagged NAME-faN.
# Needs nsys on PATH (apt: nsight-systems-cli or the CUDA toolkit's copy) and the vLLM venv active.
set -uo pipefail
cd "$(dirname "$0")/.."
ulimit -n 65536 2>/dev/null || true
command -v nsys >/dev/null || { echo "refusing: nsys not found" >&2; exit 2; }
MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
OUT="$(mkdir -p "${1:-results/raw/busyfrac}" && cd "${1:-results/raw/busyfrac}" && pwd)"
ARMS="${ARMS:-256:512x128 1024:512x128 256:128x128 1024:128x128}"
BACKENDS="${BACKENDS:-FLASH_ATTN TRITON_ATTN}"
DELAYS="${DELAYS:-200 600}"; ITERS="${ITERS:-150}"; BT="${BT:-8192}"; MAXLEN="${MAXLEN:-4096}"; NUM_PROMPTS="${NUM_PROMPTS:-1000}"
PORT=18000
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$OUT/driver.log"; }

for delay in $DELAYS; do
  for arm in $ARMS; do
    seqs="${arm%%:*}"; workload="${arm#*:}"; in_len="${workload%x*}"; out_len="${workload#*x}"
    for entry in $BACKENDS; do
      backend="${entry%%:*}"; fa_args=(); label="$backend"
      if [[ "$entry" == *:* ]]; then fa_args=(--attention-config.flash_attn_version "${entry#*:}"); label="${backend}-fa${entry#*:}"; fi
      tag="${label}_s${seqs}_${workload}_d${delay}"
      [[ -s "$OUT/$tag.busy.json" ]] && { log "SKIP $tag (done)"; continue; }
      log "RUN $tag"
      if (echo > "/dev/tcp/127.0.0.1/${PORT}") 2>/dev/null; then log "port $PORT in use"; exit 1; fi
      setsid nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none --capture-range=cudaProfilerApi --capture-range-end=stop \
        --force-overwrite=true -o "$OUT/$tag" \
        vllm serve "$MODEL" --max-model-len "$MAXLEN" --gpu-memory-utilization 0.85 --max-num-seqs "$seqs" --max-num-batched-tokens "$BT" \
        --attention-backend "$backend" ${fa_args[@]+"${fa_args[@]}"} --seed 0 --host 127.0.0.1 --port "$PORT" \
        --profiler-config.profiler=cuda --profiler-config.delay_iterations="$delay" --profiler-config.max_iterations="$ITERS" \
        > "$OUT/server_$tag.log" 2>&1 &
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
          --ignore-eos --seed 0 --profile --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,99 \
          --save-result --result-dir "$OUT" --result-filename "$tag.json" > "$OUT/bench_$tag.log" 2>&1 || log "bench failed for $tag"
        sleep 3
      fi
      kill -TERM -- "-$server" 2>/dev/null
      waited=0; while kill -0 "$server" 2>/dev/null && (( waited < 240 )); do sleep 1; waited=$((waited + 1)); done   # nsys writes the report on exit
      kill -KILL -- "-$server" 2>/dev/null; sleep 3
      if [[ -s "$OUT/$tag.nsys-rep" ]]; then
        nsys export --type sqlite --force-overwrite true -o "$OUT/$tag.sqlite" "$OUT/$tag.nsys-rep" > "$OUT/export_$tag.log" 2>&1 \
          && python -m gpu.nsys_busy "$OUT/$tag.sqlite" > "$OUT/$tag.busy.json" 2> "$OUT/busy_$tag.err" \
          && rm -f "$OUT/$tag.sqlite" || log "busy-fraction extraction failed for $tag"
        gzip -f "$OUT/$tag.nsys-rep"
      else
        log "no nsys report for $tag"
      fi
    done
  done
done
touch "$OUT/DONE"
