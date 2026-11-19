#!/usr/bin/env bash
# Automates day5-distributed/RUNBOOK.md sections 0-4 on a host with exactly two NVIDIA GPUs.
#
#   source ../day4-vllm-internals/.venv-gpu/bin/activate
#   ./run_tp_compare.sh [OUT_DIR]          # default results/raw/tp_compare
#   REPS=1 CONCS="1 32" ./run_tp_compare.sh /tmp/smoke      # quick check first
#
# One FRESH server per (config, concurrency, repeat), as in Day 4, so prefix-cache hits never carry over.
# Everything is logged under OUT_DIR; the script keeps going after a failed run and lists failures at the end.
# Written for RunPod; not executed anywhere until the first two-GPU run, so read the log of the first run closely.
set -uo pipefail
cd "$(dirname "$0")"

MODEL="${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
OUT="${1:-results/raw/tp_compare}"
CONCS="${CONCS:-1 8 32 128}"
REPS="${REPS:-3}"
# Not 8001: RunPod's nginx listens there and answers /health with 200, which fakes a healthy second replica.
PORT_A=18000
PORT_B=18001
ONLY_DP="${ONLY_DP:-}"                       # set to 1 to rerun just the data-parallel baseline
# Extra serve flags for the main TP=2 arms only, e.g. TP2_EXTRA="--disable-custom-all-reduce" when the custom kernel crashes
# on a PCIe host (then the "tp2" label means NCCL-only all-reduce; say so in the results).
read -r -a TP2_EXTRA_ARGS <<< "${TP2_EXTRA:-}"
MAX_CONSEC_START_FAILS="${MAX_CONSEC_START_FAILS:-3}"   # abort instead of burning pod time on a broken host
CONSEC_START_FAILS=0
HEALTH_TIMEOUT_S="${HEALTH_TIMEOUT_S:-900}"
SERVE_FLAGS=(--max-model-len 4096 --gpu-memory-utilization 0.85)
mkdir -p "$OUT"
FAILURES=()

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$OUT/driver.log"; }

gpu_count="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d ' ')"
if [[ "$gpu_count" -lt 2 ]]; then echo "need 2 GPUs, found $gpu_count" >&2; exit 2; fi

# --- section 0: topology + NCCL baseline ---------------------------------------------------------
topology() {
  nvidia-smi topo -m | tee "$OUT/topo.txt" >/dev/null
  nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv | tee "$OUT/gpus.txt" >/dev/null
  uv pip freeze 2>/dev/null | grep -Ei '^(vllm|torch|nvidia-nccl)' > "$OUT/env.txt" || true
  python - <<'PY' >> "$OUT/env.txt" 2>&1 || true
import torch; print("torch.cuda.nccl.version", torch.cuda.nccl.version())
PY
  log "topology saved ($(grep -c '' "$OUT/topo.txt") lines); NVLink entries: $(grep -oE 'NV[0-9]+' "$OUT/topo.txt" | sort -u | tr '\n' ' ')"
}

nccl_baseline() {
  # nccl-tests needs nvcc + NCCL headers; if the build fails, fall back to Day 5's own torch NCCL benchmark.
  if [[ ! -x nccl-tests/build/all_reduce_perf ]]; then
    { git clone --depth 1 https://github.com/NVIDIA/nccl-tests nccl-tests \
      && make -C nccl-tests -j8 CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"; } > "$OUT/nccl_build.log" 2>&1 \
      || log "nccl-tests build failed (see nccl_build.log); using the torch fallback only"
  fi
  if [[ -x nccl-tests/build/all_reduce_perf ]]; then
    ./nccl-tests/build/all_reduce_perf -b 1K -e 256M -f 2 -g 2 > "$OUT/nccl_all_reduce.txt" 2>&1 \
      || FAILURES+=("nccl_all_reduce_perf")
  fi
  # Day 5's own torchrun entry points pick NCCL when each rank owns a GPU (dist_utils.resolve_backend). This is the
  # first time that path runs: collectives (init, all-reduce, broadcast, all-gather) and the TP layers vs a reference.
  # (run_experiments.py is CPU/Gloo only and spawns its own workers, so it is not run here.)
  for script in collectives.py tp_demo.py; do
    ./launch.sh 2 "$script" > "$OUT/day5_${script%.py}_nccl.log" 2>&1 \
      || { FAILURES+=("day5_${script%.py}_nccl"); log "$script on NCCL failed:"; tail -5 "$OUT/day5_${script%.py}_nccl.log" | tee -a "$OUT/driver.log"; }
  done
}

# --- servers -------------------------------------------------------------------------------------
SERVER_PIDS=()
start_server() {  # start_server TAG PORT GPUS [extra serve args...]  -> 0 when healthy
  local tag="$1" port="$2" gpus="$3"; shift 3
  if (echo > "/dev/tcp/127.0.0.1/${2}") 2>/dev/null; then log "port ${2} already in use before start"; return 1; fi
  CUDA_VISIBLE_DEVICES="$gpus" setsid vllm serve "$MODEL" "${SERVE_FLAGS[@]}" --port "$port" --host 127.0.0.1 "$@" \
    > "$OUT/server_${tag}.log" 2>&1 &
  SERVER_PIDS+=("$!")
  local waited=0
  until curl -sf "http://127.0.0.1:${port}/health" >/dev/null 2>&1 \
        && curl -sf "http://127.0.0.1:${port}/v1/models" | grep -q "$MODEL"; do
    if ! kill -0 "${SERVER_PIDS[-1]}" 2>/dev/null; then log "server $tag died during startup"; return 1; fi
    if (( waited >= HEALTH_TIMEOUT_S )); then log "server $tag not healthy after ${HEALTH_TIMEOUT_S}s"; return 1; fi
    sleep 3; waited=$((waited + 3))
  done
  { printf '%s: ' "$tag"; grep -m1 "GPU KV cache size" "$OUT/server_${tag}.log" || echo "(no KV line)"; } \
    >> "$OUT/kv_cache_lines.txt"
  return 0
}

stop_servers() {
  for pid in "${SERVER_PIDS[@]:-}"; do
    [[ -n "$pid" ]] && kill -TERM -- "-$pid" 2>/dev/null
  done
  local waited=0
  while pgrep -f "vllm serve" >/dev/null 2>&1 && (( waited < 40 )); do sleep 1; waited=$((waited + 1)); done
  if pgrep -f "vllm serve|VLLM::" >/dev/null 2>&1; then pkill -9 -f "vllm serve|VLLM::" 2>/dev/null; sleep 3; fi
  SERVER_PIDS=()
  # GPU memory must be back to ~0 before the next config, or the next run is silently squeezed.
  local used
  used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)"
  (( used > 1000 )) && log "WARNING: ${used} MiB still in use after stopping servers"
  return 0
}
trap 'stop_servers' EXIT INT TERM

bench() {  # bench TAG PORT CONC NUM_PROMPTS SEED
  local tag="$1" port="$2" conc="$3" n="$4" seed="$5"
  vllm bench serve --backend vllm --model "$MODEL" --endpoint /v1/completions --host 127.0.0.1 --port "$port" \
    --dataset-name random --random-input-len 512 --random-output-len 128 --ignore-eos \
    --num-prompts "$n" --num-warmups "$conc" --max-concurrency "$conc" --request-rate inf --seed "$seed" \
    --percentile-metrics ttft,tpot,itl,e2el --save-result \
    --result-dir "$OUT" --result-filename "${tag}.json" > "$OUT/bench_${tag}.log" 2>&1
}

mem_log_start() { nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu --format=csv -l 1 > "$OUT/mem_$1.csv" 2>&1 & MEM_PID=$!; }
mem_log_stop() { kill "$MEM_PID" 2>/dev/null; wait "$MEM_PID" 2>/dev/null; }
prompts_for() { local c="$1"; echo $(( c * 8 > 200 ? c * 8 : 200 )); }

run_one() {  # run_one LABEL GPUS CONC REP [serve args...]  (single server)
  local label="$1" gpus="$2" c="$3" rep="$4"; shift 4
  local tag="${label}_c${c}_r${rep}"
  log "RUN $tag"
  if start_server "$tag" "$PORT_A" "$gpus" "$@"; then
    CONSEC_START_FAILS=0
    mem_log_start "$tag"
    bench "$tag" "$PORT_A" "$c" "$(prompts_for "$c")" $((100 + rep)) || FAILURES+=("$tag")
    mem_log_stop
  else
    FAILURES+=("$tag")
    note_start_failure
  fi
  stop_servers
}

note_start_failure() {
  CONSEC_START_FAILS=$((CONSEC_START_FAILS + 1))
  if (( CONSEC_START_FAILS >= MAX_CONSEC_START_FAILS )); then
    log "ABORT: $CONSEC_START_FAILS consecutive server start failures (GPU wedged? check: CUDA_VISIBLE_DEVICES=1 python -c 'import torch;torch.ones(1).cuda()')"
    log "failures so far: ${FAILURES[*]}"
    exit 3
  fi
}

run_dp() {  # two TP=1 replicas, each gets half the prompts and half the concurrency, run in parallel
  local c="$1" rep="$2" tag="dp2_c${1}_r${2}"
  (( c < 2 )) && return 0
  log "RUN $tag"
  if start_server "${tag}_a" "$PORT_A" 0 && start_server "${tag}_b" "$PORT_B" 1; then
    local n half_c half_n
    n="$(prompts_for "$c")"; half_c=$(( c / 2 )); half_n=$(( n / 2 ))
    mem_log_start "$tag"
    bench "${tag}_a" "$PORT_A" "$half_c" "$half_n" $((100 + rep)) &
    local pa=$!
    bench "${tag}_b" "$PORT_B" "$half_c" "$half_n" $((200 + rep)) &
    local pb=$!
    wait "$pa" || FAILURES+=("${tag}_a"); wait "$pb" || FAILURES+=("${tag}_b")
    mem_log_stop
  else
    FAILURES+=("$tag")
    note_start_failure
  fi
  stop_servers
}

log "model=$MODEL out=$OUT concs=[$CONCS] reps=$REPS gpus=$gpu_count"
if [[ -z "$ONLY_DP" ]]; then
topology
nccl_baseline

for rep in $(seq 0 $((REPS - 1))); do          # repeat-major order: slow drift does not align with one config
  for c in $CONCS; do
    run_one tp1 0 "$c" "$rep" --tensor-parallel-size 1
    run_one tp2 0,1 "$c" "$rep" --tensor-parallel-size 2 ${TP2_EXTRA_ARGS[@]+"${TP2_EXTRA_ARGS[@]}"}
  done
done
NOCUSTOM_C="${NOCUSTOM_C:-8}"                  # RUNBOOK 3b: one concurrency, small messages
for rep in $(seq 0 $((REPS - 1))); do
  run_one tp2nc 0,1 "$NOCUSTOM_C" "$rep" --tensor-parallel-size 2 --disable-custom-all-reduce
done
fi
for c in ${DP_CONCS:-32 128}; do                 # RUNBOOK 4: is TP=2 even the right use of the second GPU?
  for rep in $(seq 0 $((REPS - 1))); do run_dp "$c" "$rep"; done
done

log "DONE. failures: ${#FAILURES[@]} ${FAILURES[*]:-}"
touch "$OUT/DONE"
