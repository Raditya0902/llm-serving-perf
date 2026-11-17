from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from verify_trace import (
    Manifest,
    ManifestError,
    Problem,
    SymbolRef,
    dump_manifest,
    fill_lines,
    load_manifest,
    main,
    verify,
)

SOURCE = textwrap.dedent(
    '''\
    LIMIT = 3

    def helper():
        pass

    class Engine:
        flag: int = 1

        def step(self):
            def inner():
                return 1
            return inner()

        async def aloop(self):
            pass

    if True:
        def conditional():
            pass
    '''
)


@pytest.fixture
def src(tmp_path) -> Path:
    root = tmp_path / "src"
    (root / "pkg").mkdir(parents=True)
    (root / "pkg" / "mod.py").write_text(SOURCE)
    (root / "pkg" / "broken.py").write_text("def (:\n")
    return root


def ref(symbol: str, line: int | None = None, file: str = "pkg/mod.py") -> SymbolRef:
    return SymbolRef(file=file, symbol=symbol, line=line)


@pytest.mark.parametrize(
    "symbol",
    ["LIMIT", "helper", "Engine", "Engine.flag", "Engine.step", "Engine.aloop",
     "Engine.step.inner", "conditional"],
)
def test_existing_symbols_are_found(src, symbol):
    assert verify([ref(symbol)], src) == []


@pytest.mark.parametrize("symbol", ["Missing", "Engine.missing", "Engine.step.nope", "helper.x"])
def test_missing_symbols_are_reported(src, symbol):
    problems = verify([ref(symbol)], src)

    assert len(problems) == 1
    assert problems[0].kind == "missing_symbol"
    assert symbol in problems[0].message


def test_missing_file_is_reported(src):
    problems = verify([ref("helper", file="pkg/gone.py")], src)

    assert [p.kind for p in problems] == ["missing_file"]


def test_unparseable_file_is_reported(src):
    problems = verify([ref("x", file="pkg/broken.py")], src)

    assert [p.kind for p in problems] == ["unparseable"]


def test_line_numbers_are_checked_when_given(src):
    assert verify([ref("Engine.step", line=9)], src) == []
    problems = verify([ref("Engine.step", line=10)], src)

    assert [p.kind for p in problems] == ["wrong_line"]
    assert "actual line 9" in problems[0].message


def test_path_escaping_the_source_root_is_rejected(src):
    (src.parent / "outside.py").write_text("x = 1\n")

    problems = verify([ref("x", file="../outside.py")], src)

    assert [p.kind for p in problems] == ["bad_path"]


def test_verify_reports_every_problem_not_just_the_first(src):
    problems = verify([ref("nope"), ref("alsonope"), ref("helper")], src)

    assert len(problems) == 2


def test_load_manifest_parses_symbols(tmp_path):
    path = tmp_path / "m.json"
    path.write_text(json.dumps({
        "vllm_tag": "v0.34.0", "commit": "abc",
        "symbols": [{"file": "a.py", "symbol": "A.b", "line": 4}, {"file": "a.py", "symbol": "c"}],
    }))

    manifest = load_manifest(path)

    assert manifest.commit == "abc"
    assert manifest.symbols == (SymbolRef("a.py", "A.b", 4), SymbolRef("a.py", "c", None))


@pytest.mark.parametrize(
    "payload",
    [
        "not json",
        json.dumps([]),
        json.dumps({"symbols": []}),
        json.dumps({"vllm_tag": "t", "commit": "c", "symbols": [{"file": "a.py"}]}),
        json.dumps({"vllm_tag": "t", "commit": "c",
                    "symbols": [{"file": "a.py", "symbol": "x", "line": "9"}]}),
    ],
)
def test_load_manifest_rejects_malformed_input(tmp_path, payload):
    path = tmp_path / "m.json"
    path.write_text(payload)

    with pytest.raises(ManifestError):
        load_manifest(path)


def test_main_exit_codes(src, tmp_path, capsys):
    good = tmp_path / "good.json"
    good.write_text(json.dumps({"vllm_tag": "t", "commit": "c", "symbols": [
        {"file": "pkg/mod.py", "symbol": "Engine.step", "line": 9}]}))
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"vllm_tag": "t", "commit": "c", "symbols": [
        {"file": "pkg/mod.py", "symbol": "Engine.gone"}]}))

    assert main(["--src", str(src), "--manifest", str(good)]) == 0
    assert "1/1 symbols verified" in capsys.readouterr().out
    assert main(["--src", str(src), "--manifest", str(bad)]) == 1
    assert "missing_symbol" in capsys.readouterr().out
    assert main(["--src", str(tmp_path / "nodir"), "--manifest", str(good)]) == 2
    assert main(["--src", str(src), "--manifest", str(tmp_path / "nofile.json")]) == 2


def test_fill_lines_sets_actual_lines_and_keeps_unresolved(src):
    refs = (ref("Engine.step"), ref("Missing"), ref("helper", line=99))

    filled = fill_lines(refs, src)

    assert [r.line for r in filled] == [9, None, 3]
    assert [r.symbol for r in filled] == ["Engine.step", "Missing", "helper"]
    assert refs[0].line is None  # inputs are not mutated


def test_dump_manifest_round_trips(tmp_path):
    manifest = Manifest("v0.34.0", "abc", (SymbolRef("a.py", "A.b", 4), SymbolRef("a.py", "c")))
    path = tmp_path / "m.json"

    dump_manifest(manifest, path)

    assert load_manifest(path) == manifest
    assert '"line"' in path.read_text()


def test_main_write_lines_updates_manifest_in_place(src, tmp_path):
    path = tmp_path / "m.json"
    path.write_text(json.dumps({"vllm_tag": "t", "commit": "c", "symbols": [
        {"file": "pkg/mod.py", "symbol": "Engine.step"},
        {"file": "pkg/mod.py", "symbol": "Engine.gone"}]}))

    code = main(["--src", str(src), "--manifest", str(path), "--write-lines"])

    saved = load_manifest(path)
    assert saved.symbols[0].line == 9
    assert saved.symbols[1].line is None
    assert code == 1  # the unresolved symbol is still a problem


def test_problem_is_immutable():
    with pytest.raises(AttributeError):
        Problem("missing_file", "x").kind = "y"  # type: ignore[misc]


# --- review fixes: arbitrarily nested blocks, unreadable files --------------------------------
NESTED = textwrap.dedent(
    '''\
    if A:
        if B:
            def deep():
                pass
    try:
        pass
    except Exception:
        if C:
            def in_handler():
                pass
    else:
        def in_else():
            pass
    finally:
        def in_finally():
            pass
    for i in range(1):
        with ctx():
            while True:
                def in_loop():
                    pass
    match x:
        case 1:
            def in_case():
                pass

    class K:
        if TYPE_CHECKING:
            def method_under_if(self):
                pass
    '''
)


@pytest.mark.parametrize(
    "symbol",
    ["deep", "in_handler", "in_else", "in_finally", "in_loop", "in_case", "K.method_under_if"],
)
def test_symbols_under_arbitrarily_nested_blocks_are_found(tmp_path, symbol):
    root = tmp_path / "src"
    root.mkdir()
    (root / "nested.py").write_text(NESTED)

    assert verify([SymbolRef("nested.py", symbol)], root) == []


def test_function_bodies_are_scopes_not_flattened_into_the_module(tmp_path):
    root = tmp_path / "src"
    root.mkdir()
    (root / "m.py").write_text("def outer():\n    if x:\n        def inner():\n            pass\n")

    assert verify([SymbolRef("m.py", "inner")], root)  # not a module-level symbol
    assert verify([SymbolRef("m.py", "outer.inner")], root) == []


@pytest.mark.parametrize(
    "payload",
    [b"\xff\xfe\x00bad utf8 \x80", b"x = 1\x00\n", b"(" * 5000 + b")" * 5000],
    ids=["not-utf8", "null-byte", "deeply-nested"],
)
def test_hostile_or_odd_files_become_problems_not_crashes(tmp_path, payload):
    root = tmp_path / "src"
    root.mkdir()
    (root / "odd.py").write_bytes(payload)

    problems = verify([SymbolRef("odd.py", "x"), SymbolRef("odd.py", "y")], root)

    assert [p.kind for p in problems] == ["unparseable", "unparseable"]
