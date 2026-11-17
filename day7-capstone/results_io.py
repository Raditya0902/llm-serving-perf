"""Result files: ``{meta, rows}`` JSON plus a flat CSV, and run provenance.

Conventions shared with Days 3-6: every file carries a ``meta`` block saying where the
numbers came from (``simulated`` / ``source``), and an unmeasured value is ``null`` in JSON
and ``not measured`` in CSV, never zero.

Safety properties: files are written atomically (a failed write never truncates the previous
file), string cells that spreadsheets would execute as formulas are neutralised, reads refuse
symlinks and oversized files, and ``assert_fresh`` lets a derived file prove it was computed from
the input that is on disk now.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import platform
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

CSV_NOT_MEASURED = "not measured"
CSV_FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")
GIT_TIMEOUT_S = 5
MAX_RESULT_BYTES = 256 * 1024 * 1024
HASH_CHUNK_BYTES = 1024 * 1024
HERE = Path(__file__).resolve().parent


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args], cwd=HERE, capture_output=True, text=True, timeout=GIT_TIMEOUT_S, check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip()


def run_meta(*, simulated: bool, note: str, extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Provenance block: what produced these numbers and from which code."""
    status = _git("status", "--porcelain", "--", ".")
    return {
        "simulated": simulated,
        "note": note,
        "git_commit": _git("rev-parse", "--short", "HEAD") or "unknown",
        "git_dirty": None if status is None else bool(status),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        **(extra or {}),
    }


def _flatten(row: Mapping[str, Any]) -> dict[str, Any]:
    """One level of nested dicts (metric -> stats) becomes ``metric_stat`` columns."""
    flat: dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, Mapping):
            flat.update({f"{key}_{stat}": v for stat, v in value.items()})
        else:
            flat[key] = value
    return flat


def _csv_cell(value: Any) -> Any:
    """None -> 'not measured'; strings a spreadsheet would run as formulas get a leading quote."""
    if value is None:
        return CSV_NOT_MEASURED
    if isinstance(value, str) and value.startswith(CSV_FORMULA_PREFIXES):
        return "'" + value
    return value


def _csv_text(rows: Sequence[Mapping[str, Any]]) -> str:
    flat_rows = [_flatten(r) for r in rows]
    columns = list(dict.fromkeys(k for r in flat_rows for k in r))
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns)
    writer.writeheader()
    for row in flat_rows:
        writer.writerow({k: _csv_cell(row.get(k)) for k in columns})
    return buffer.getvalue()


def _atomic_write(path: Path, text: str) -> None:
    """Write beside the target then rename, so a crash never leaves a truncated file."""
    handle, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(handle, "w", newline="") as fh:
            fh.write(text)
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def write_json_atomic(path: Path, obj: Any) -> None:
    """Atomically write one strict-JSON file (no NaN/Infinity)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(path, json.dumps(obj, allow_nan=False))


def write_results(
    name: str, rows: Sequence[Mapping[str, Any]], meta: Mapping[str, Any], out_dir: Path,
    *, write_csv: bool = True,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"meta": dict(meta), "rows": [dict(r) for r in rows]}
    json_text = json.dumps(payload, indent=2, allow_nan=False)  # fails before any file is touched
    csv_text = _csv_text(rows) if rows and write_csv else None
    _atomic_write(out_dir / f"{name}.json", json_text)
    csv_path = out_dir / f"{name}.csv"
    if csv_text is None:
        csv_path.unlink(missing_ok=True)  # never leave a stale CSV beside a newer JSON
    else:
        _atomic_write(csv_path, csv_text)


def read_results(path: Path) -> dict[str, Any]:
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{path}: must be a regular file")
    if path.stat().st_size > MAX_RESULT_BYTES:
        raise ValueError(f"{path}: too large (> {MAX_RESULT_BYTES} bytes)")
    data = json.loads(path.read_text())
    if not isinstance(data, dict) or "meta" not in data or "rows" not in data:
        raise ValueError(f"{path}: expected an object with 'meta' and 'rows'")
    return data


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_fresh(path: Path, expected_sha256: str | None) -> None:
    """Raise ValueError unless `path` still has the digest a derived file recorded."""
    if not expected_sha256:
        raise ValueError(f"analysis has no input digest for {Path(path).name}; re-run analyze.py")
    if file_sha256(path) != expected_sha256:
        raise ValueError(
            f"analysis is stale: {Path(path).name} changed since analyze.py ran; re-run analyze.py"
        )
