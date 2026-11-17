"""Verify that every symbol cited in SOURCE_TRACE.md exists in the vLLM checkout.

``trace_manifest.json`` lists each cited ``(file, symbol[, line])`` at a pinned
vLLM tag/commit.  This script parses the files with ``ast`` (nothing is
imported or executed, so no GPU or vLLM install is needed) and reports any
symbol that is missing, any file that cannot be parsed, and -- when a line is
given -- any ``def``/``class`` that has moved.  Re-run it after bumping the
vLLM version to find which parts of the trace went stale.

Symbols are dotted paths: ``Scheduler.schedule``, ``Scheduler.schedule.skip_request``
(nested function), ``Engine.flag`` (class attribute) or ``CONSTANT``.
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

EXIT_OK, EXIT_PROBLEMS, EXIT_USAGE = 0, 1, 2
_DEFS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
# Anything that can legitimately fail while reading or parsing a hostile/odd source file.
_PARSE_ERRORS = (SyntaxError, ValueError, UnicodeDecodeError, RecursionError, MemoryError, OSError)


class ManifestError(ValueError):
    """The manifest file is missing or malformed."""


@dataclass(frozen=True)
class SymbolRef:
    file: str
    symbol: str
    line: int | None = None


@dataclass(frozen=True)
class Manifest:
    vllm_tag: str
    commit: str
    symbols: tuple[SymbolRef, ...]


@dataclass(frozen=True)
class Problem:
    kind: str  # missing_file | unparseable | bad_path | missing_symbol | wrong_line
    message: str


def load_manifest(path: Path) -> Manifest:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError(f"cannot read manifest {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ManifestError("manifest must be a JSON object")
    for key in ("vllm_tag", "commit", "symbols"):
        if key not in data:
            raise ManifestError(f"manifest missing key {key!r}")

    symbols: list[SymbolRef] = []
    for i, item in enumerate(data["symbols"]):
        if not isinstance(item, dict) or "file" not in item or "symbol" not in item:
            raise ManifestError(f"symbols[{i}] needs 'file' and 'symbol'")
        line = item.get("line")
        if line is not None and (isinstance(line, bool) or not isinstance(line, int)):
            raise ManifestError(f"symbols[{i}].line must be an integer")
        symbols.append(SymbolRef(str(item["file"]), str(item["symbol"]), line))
    return Manifest(str(data["vllm_tag"]), str(data["commit"]), tuple(symbols))


def _nested(stmt: ast.stmt) -> Iterable[ast.stmt]:
    """Statements inside ``stmt``'s blocks (if/try/with/for/while/match), at any depth.

    ``def``/``class`` bodies are *scopes*: their contents belong to that symbol, so
    they are not flattened into the enclosing scope.
    """
    if isinstance(stmt, _DEFS):
        return
    blocks = [getattr(stmt, attr, []) for attr in ("body", "orelse", "finalbody")]
    blocks += [h.body for h in getattr(stmt, "handlers", [])]  # try/except, except*
    blocks += [c.body for c in getattr(stmt, "cases", [])]  # match/case
    for block in blocks:
        for inner in block:
            yield inner
            yield from _nested(inner)


def _children(node: ast.AST) -> Iterable[ast.stmt]:
    """Statements of ``node``'s scope, looking through conditional blocks."""
    for stmt in getattr(node, "body", []):
        yield stmt
        yield from _nested(stmt)


def _binds(stmt: ast.stmt, name: str) -> bool:
    if isinstance(stmt, _DEFS):
        return stmt.name == name
    if isinstance(stmt, ast.Assign):
        return any(isinstance(t, ast.Name) and t.id == name for t in stmt.targets)
    if isinstance(stmt, ast.AnnAssign):
        return isinstance(stmt.target, ast.Name) and stmt.target.id == name
    return False


def _resolve(tree: ast.Module, dotted: str) -> ast.stmt | None:
    scope: ast.AST = tree
    found: ast.stmt | None = None
    for part in dotted.split("."):
        found = next((s for s in _children(scope) if _binds(s, part)), None)
        if found is None:
            return None
        scope = found
    return found


def _locate(
    ref: SymbolRef, src_root: Path, trees: dict[Path, ast.Module | Problem]
) -> ast.stmt | Problem:
    """Find the AST node for ``ref`` or explain why it cannot be found."""
    path = (src_root / ref.file).resolve()
    if not path.is_relative_to(src_root.resolve()):
        return Problem("bad_path", f"{ref.file}: path escapes the source root")
    if not path.is_file():
        return Problem("missing_file", f"{ref.file}: file not found")

    if path not in trees:
        try:
            trees[path] = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except _PARSE_ERRORS as exc:
            reason = getattr(exc, "msg", None) or type(exc).__name__
            trees[path] = Problem("unparseable", f"{ref.file}: cannot parse ({reason})")
    tree = trees[path]
    if isinstance(tree, Problem):
        return tree

    node = _resolve(tree, ref.symbol)
    if node is None:
        return Problem("missing_symbol", f"{ref.file}: no symbol {ref.symbol!r}")
    return node


def _check_one(ref: SymbolRef, src_root: Path, trees: dict[Path, ast.Module | Problem]) -> Problem | None:
    found = _locate(ref, src_root, trees)
    if isinstance(found, Problem):
        return found
    if ref.line is not None and found.lineno != ref.line:
        return Problem(
            "wrong_line",
            f"{ref.file}: {ref.symbol} cited at line {ref.line}, actual line {found.lineno}",
        )
    return None


def verify(refs: Sequence[SymbolRef], src_root: Path) -> list[Problem]:
    """Check every ref; return all problems (empty list means the trace is current)."""
    trees: dict[Path, ast.Module | Problem] = {}
    problems = (_check_one(ref, src_root, trees) for ref in refs)
    return [p for p in problems if p is not None]


def fill_lines(refs: Sequence[SymbolRef], src_root: Path) -> tuple[SymbolRef, ...]:
    """Return refs with ``line`` set to each resolvable symbol's real line.

    Unresolvable refs are returned unchanged so ``verify`` still reports them.
    """
    trees: dict[Path, ast.Module | Problem] = {}
    filled: list[SymbolRef] = []
    for ref in refs:
        found = _locate(ref, src_root, trees)
        line = ref.line if isinstance(found, Problem) else found.lineno
        filled.append(SymbolRef(ref.file, ref.symbol, line))
    return tuple(filled)


def dump_manifest(manifest: Manifest, path: Path) -> None:
    """Write ``manifest`` as JSON (``line`` omitted when unknown)."""
    symbols = []
    for ref in manifest.symbols:
        item: dict[str, str | int] = {"file": ref.file, "symbol": ref.symbol}
        if ref.line is not None:
            item["line"] = ref.line
        symbols.append(item)
    payload = {"vllm_tag": manifest.vllm_tag, "commit": manifest.commit, "symbols": symbols}
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--src", type=Path, default=Path("vllm_src"), help="vLLM checkout root")
    parser.add_argument("--manifest", type=Path, default=Path("trace_manifest.json"))
    parser.add_argument("--write-lines", action="store_true",
                        help="record each symbol's actual line in the manifest, then verify")
    args = parser.parse_args(argv)

    if not args.src.is_dir():
        print(f"error: source directory {args.src} not found; run fetch_vllm_source.sh",
              file=sys.stderr)
        return EXIT_USAGE
    try:
        manifest = load_manifest(args.manifest)
    except ManifestError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE

    if args.write_lines:
        manifest = Manifest(manifest.vllm_tag, manifest.commit,
                            fill_lines(manifest.symbols, args.src))
        dump_manifest(manifest, args.manifest)

    problems = verify(manifest.symbols, args.src)
    for problem in problems:
        print(f"{problem.kind}: {problem.message}")
    ok = len(manifest.symbols) - len(problems)
    print(f"{ok}/{len(manifest.symbols)} symbols verified "
          f"(manifest pinned to {manifest.vllm_tag} @ {manifest.commit[:7]})")
    return EXIT_PROBLEMS if problems else EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
