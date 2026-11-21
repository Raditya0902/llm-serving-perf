#!/usr/bin/env bash
# One-time setup of a Linux + NVIDIA GPU host for the Session 3 sweep.
# Follows the install steps in vllm_src/docs/getting_started/installation/gpu.cuda.inc.md
# (v0.34.0): a fresh uv environment, `uv pip install vllm --torch-backend=auto`,
# then `vllm download-kernels`.
#
#   ./setup_gpu_host.sh            # creates ./.venv-gpu and results/env.txt
#
# Requires: Linux, an NVIDIA driver (nvidia-smi), and uv (https://docs.astral.sh/uv/).
# This script does NOT install uv for you (piping a remote script into a shell is
# your call to make) and has never been run on a GPU host by its author: it was
# written on a Mac. Read it before you run it.
set -euo pipefail

VLLM_VERSION="${VLLM_VERSION:-0.34.0}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$HERE/.venv-gpu"
ENV_FILE="$HERE/results/env.txt"

die() { echo "error: $*" >&2; exit 1; }

[[ "$(uname -s)" == "Linux" ]] || die "vLLM's CUDA build needs Linux (this is $(uname -s))."
command -v nvidia-smi >/dev/null || die "nvidia-smi not found: install an NVIDIA driver first."
command -v uv >/dev/null || die "uv not found: see https://docs.astral.sh/uv/getting-started/installation/"

nvidia-smi --query-gpu=name --format=csv,noheader >/dev/null || die "nvidia-smi cannot see a GPU."

if [[ ! -d "$VENV" ]]; then
  uv venv --python "$PYTHON_VERSION" "$VENV"
fi
uv pip install --python "$VENV/bin/python" "vllm==${VLLM_VERSION}" --torch-backend=auto
"$VENV/bin/vllm" download-kernels
uv pip install --python "$VENV/bin/python" -r "$HERE/requirements.txt"

mkdir -p "$HERE/results"
# Assign first: `set -e` does not fire on a failed $(...) inside an echo argument, which
# would silently write a blank value into the environment record.
GPU_INFO="$(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | paste -sd ';' -)"
PY_VERSION="$("$VENV/bin/python" --version 2>&1)"
VLLM_INSTALLED="$("$VENV/bin/python" -c 'import vllm; print(vllm.__version__)')"
TORCH_INSTALLED="$("$VENV/bin/python" -c 'import torch; print(torch.__version__, "cuda", torch.version.cuda)')"
[[ "$VLLM_INSTALLED" == "$VLLM_VERSION" ]] || die "installed vllm is $VLLM_INSTALLED, expected $VLLM_VERSION"
{
  echo "date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "host: $(uname -srm)"
  echo "gpu: $GPU_INFO"
  echo "python: $PY_VERSION"
  echo "vllm: $VLLM_INSTALLED"
  echo "torch: $TORCH_INSTALLED"
} | tee "$ENV_FILE"

echo
echo "Done. Activate with:  source $VENV/bin/activate"
echo "Next: see RUNBOOK.md (smoke test, then run_sweep.py)."
