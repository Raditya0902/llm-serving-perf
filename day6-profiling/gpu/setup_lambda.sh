#!/usr/bin/env bash
# One-time setup of a fresh Lambda Cloud VM (Ubuntu, NVIDIA driver preinstalled, user `ubuntu` with sudo) for rounds 5+.
# Run from the repo root after untarring the repo:   bash day6-profiling/gpu/setup_lambda.sh
# Creates day4-vllm-internals/.venv-gpu (vllm 0.34.0, torch picked by --torch-backend=auto for THIS driver), adds the Day 6 test deps,
# installs tmux, and reports whether nsys and ncu exist and whether counters are readable. Idempotent.
set -uo pipefail
cd "$(dirname "$0")/../.."
echo "== driver / GPUs"; nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
echo "== topology"; nvidia-smi topo -m 2>/dev/null | head -12
command -v tmux >/dev/null || sudo apt-get install -y tmux >/dev/null 2>&1 || echo "WARN: tmux not installed"
if ! command -v uv >/dev/null; then
  python3 -m pip install --user --quiet uv 2>&1 | tail -1
  export PATH="$HOME/.local/bin:$PATH"
fi
command -v uv >/dev/null || { echo "FAIL: uv not available"; exit 1; }
( cd day4-vllm-internals && bash setup_gpu_host.sh ) || { echo "FAIL: vLLM install (see output above)"; exit 1; }
uv pip install --python day4-vllm-internals/.venv-gpu/bin/python pytest matplotlib psutil >/dev/null 2>&1 || echo "WARN: test deps"
# Lambda's packaged nsys 2024.6 cannot import its own reports (libssh mismatch on Ubuntu 22.04); NVIDIA's 2026.5.2 works.
if ! ls /opt/nvidia/nsight-systems/2026.5.2/bin/nsys >/dev/null 2>&1; then
  ( cd /tmp && wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb \
    && sudo dpkg -i cuda-keyring_1.1-1_all.deb >/dev/null 2>&1 && sudo apt-get update >/dev/null 2>&1 \
    && sudo apt-get install -y nsight-systems-2026.5.2 >/dev/null 2>&1 ) || echo "WARN: nsight-systems-2026.5.2 install failed"
fi
command -v ncu >/dev/null || sudo apt-get install -y nsight-compute >/dev/null 2>&1
export PATH=/opt/nvidia/nsight-systems/2026.5.2/bin:$PATH
echo "== profilers"
for tool in nsys ncu; do command -v "$tool" >/dev/null && echo "ok    $tool: $(command -v $tool)" || echo "MISSING $tool"; done
echo "== counter permission (root/sudo)"; cat /proc/driver/nvidia/params 2>/dev/null | grep -i RestrictProfiling || echo "no RestrictProfiling line"
echo "SETUP_DONE  -> source day4-vllm-internals/.venv-gpu/bin/activate; export PATH=/opt/nvidia/nsight-systems/2026.5.2/bin:\$PATH; export HF_HOME=\$HOME/hf; cd day6-profiling; bash gpu/preflight_round4.sh"
