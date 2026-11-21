"""Check that every CLI flag the matrix emits exists in the pinned vLLM source.

    .venv/bin/python verify_flags.py [--src ../04-vllm-internals/vllm_src]

A flag counts as present if its ``"--name"`` string literal appears in any ``.py`` file under
``<src>/vllm`` (argparse definitions live there). This catches a renamed or removed flag before
a multi-hour GPU run does. It does not check the *meaning* or default of a flag. Exit 0 = all
present, 1 = some missing, 2 = source tree not found.
"""
from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path

import _prior
import matrix as mx

DEFAULT_SRC = _prior.VLLM_SRC_DIR
FLAG_LITERAL_RE = re.compile(r"""["'](--[a-z0-9][a-z0-9-]*)["']""")
PROBE_PORT = 8000
METADATA_FLAG = "--metadata"


def flags_in_command(cmd: Sequence[str]) -> list[str]:
    """Flag tokens of a command, stopping at ``--metadata`` (everything after is key=value data)."""
    flags = []
    for token in cmd:
        if token.startswith("--"):
            flags.append(token)
        if token == METADATA_FLAG:
            break
    return flags


def matrix_flags() -> set[str]:
    case = mx.build_matrix(cap_kv_pool=True)[0]  # capped: also covers --kv-cache-memory-bytes
    return set(flags_in_command(case.serve_command(PROBE_PORT))) | set(
        flags_in_command(case.bench_command("out", 0, PROBE_PORT))
    )


def find_flags(root: Path) -> set[str]:
    found: set[str] = set()
    for path in Path(root).rglob("*.py"):
        try:
            found.update(FLAG_LITERAL_RE.findall(path.read_text(encoding="utf-8", errors="replace")))
        except OSError:
            continue
    return found


def missing_flags(wanted: Iterable[str], src: Path) -> list[str]:
    have = find_flags(Path(src) / "vllm") if (Path(src) / "vllm").is_dir() else find_flags(Path(src))
    return sorted(set(wanted) - have)


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", type=Path, default=DEFAULT_SRC)
    args = p.parse_args(argv)
    if not args.src.is_dir():
        print(f"vLLM source not found at {args.src}; run 04-vllm-internals/fetch_vllm_source.sh",
              file=sys.stderr)
        return 2
    wanted = matrix_flags()
    missing = missing_flags(wanted, args.src)
    if missing:
        print(f"MISSING from {args.src}: {', '.join(missing)}")
        return 1
    print(f"all {len(wanted)} flags found in {args.src}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
