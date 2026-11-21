#!/usr/bin/env bash
# Two-minute check on a fresh pod before any paid run: GPU, versions, the flags round 4 depends on, a quick unit-test pass.
#   bash gpu/preflight_round4.sh        (vLLM venv active, cwd anywhere)
set -uo pipefail
cd "$(dirname "$0")/.."
fail=0
chk() { if "$@" >/dev/null 2>&1; then echo "ok    $*"; else echo "FAIL  $*"; fail=1; fi; }
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python - <<'PY'
import torch, vllm
print("vllm", vllm.__version__, "torch", torch.__version__, "cuda", torch.version.cuda, "capability", torch.cuda.get_device_capability())
PY
chk bash -c 'vllm serve --help=all 2>&1 | grep -q -- "--attention-config"'
chk bash -c 'vllm serve --help=all 2>&1 | grep -q -- "--attention-backend"'
chk python -c "from vllm.vllm_flash_attn import flash_attn_varlen_func"
chk python -c "import transformers, numpy"
chk python -m pytest -q -x tests/test_attn_sweep.py tests/test_quality_compare.py tests/test_quality_run.py tests/test_backend_ab_summary.py tests/test_e123_summary.py
python -m gpu.attn_microbench --model qwen2.5-0.5b --ctx 512 --decodes 64 --prefill 128 --iters 5 --warmup 2 | grep -E '"(mixed_over_pure|median_ms)"' | head -4
python -m gpu.attn_microbench --fa-version 3 --model qwen2.5-0.5b --ctx 512 --decodes 64 --prefill 128 --iters 5 --warmup 2 2>&1 | grep -E 'mixed_over_pure|Error|error' | head -3
exit $fail
