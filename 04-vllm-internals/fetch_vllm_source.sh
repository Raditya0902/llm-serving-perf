#!/usr/bin/env bash
# Shallow-clone vLLM at the pinned tag into ./vllm_src and verify the commit.
# Needed for the source trace and for verify_trace.py. No GPU required.
#
#   ./fetch_vllm_source.sh                 # pinned version
#   VLLM_TAG=v0.35.0 VLLM_COMMIT= ./fetch_vllm_source.sh   # try another tag, skip SHA check
set -euo pipefail

VLLM_TAG="${VLLM_TAG:-v0.34.0}"
# Empty VLLM_COMMIT disables the check (use only when deliberately changing VLLM_TAG).
VLLM_COMMIT="${VLLM_COMMIT-62d4bc3c7e33b1b6c906e233e6af2d3506c482ca}"
DEST="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/vllm_src"

verify_commit() {
  local actual
  actual="$(git -C "$DEST" rev-parse HEAD)"
  if [[ -n "$VLLM_COMMIT" && "$actual" != "$VLLM_COMMIT" ]]; then
    echo "error: $DEST is at $actual, expected $VLLM_COMMIT ($VLLM_TAG)" >&2
    echo "       remove $DEST and re-run, or set VLLM_COMMIT= to skip this check." >&2
    exit 1
  fi
  echo "vLLM $VLLM_TAG @ $actual -> $DEST"
}

if [[ -d "$DEST/.git" ]]; then
  verify_commit
  exit 0
fi

if [[ -e "$DEST" ]]; then
  echo "error: $DEST exists but is not a git checkout; refusing to overwrite it." >&2
  exit 1
fi

git clone --depth 1 --branch "$VLLM_TAG" https://github.com/vllm-project/vllm.git "$DEST"
verify_commit
