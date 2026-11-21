#!/usr/bin/env bash
# Launch a Day 5 script under torchrun.   usage: ./launch.sh NPROC script.py [args...]
#
# On Linux / GPU hosts this is exactly `torchrun --standalone --nnodes=1 --nproc-per-node=N`.
# On macOS, `--standalone` hangs when the machine hostname does not resolve (it did on the dev
# Mac), so we rendezvous explicitly on 127.0.0.1 and pin Gloo to the loopback interface.
set -euo pipefail
cd "$(dirname "$0")"

if [[ $# -lt 2 ]]; then
  echo "usage: $0 NPROC script.py [args...]" >&2
  exit 2
fi
nproc="$1"; shift

py=".venv/bin/python"; [[ -x "$py" ]] || py="python3"
torchrun=".venv/bin/torchrun"; [[ -x "$torchrun" ]] || torchrun="torchrun"

if [[ "$(uname)" == "Darwin" ]]; then
  port="$("$py" -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1])')"
  export GLOO_SOCKET_IFNAME=lo0
  exec "$torchrun" --nnodes=1 --nproc-per-node="$nproc" \
    --rdzv-backend=c10d --rdzv-endpoint="127.0.0.1:${port}" --local-addr=127.0.0.1 "$@"
fi
exec "$torchrun" --standalone --nnodes=1 --nproc-per-node="$nproc" "$@"
