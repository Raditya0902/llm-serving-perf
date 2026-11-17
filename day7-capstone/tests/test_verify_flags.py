"""Flag checker: every ``--flag`` the matrix emits must exist in the pinned vLLM source."""
from __future__ import annotations

import matrix as mx
import verify_flags as vf


def test_flags_in_command_ignores_values_and_metadata():
    cmd = ["vllm", "bench", "serve", "--port", "8000", "--ignore-eos", "--metadata", "a=--not-a-flag"]
    assert vf.flags_in_command(cmd) == ["--port", "--ignore-eos", "--metadata"]


def test_matrix_flags_cover_both_commands():
    flags = vf.matrix_flags()
    assert {"--max-concurrency", "--random-range-ratio", "--gpu-memory-utilization",
            "--max-num-batched-tokens", "--metric-percentiles", "--save-detailed"} <= flags


def test_find_flags_reads_add_argument_literals(tmp_path):
    (tmp_path / "a.py").write_text('p.add_argument("--alpha", type=int)\nx = "--beta-gamma"\ny = "not-a-flag"\n')
    assert vf.find_flags(tmp_path) == {"--alpha", "--beta-gamma"}


def test_missing_flags_are_reported(tmp_path):
    (tmp_path / "a.py").write_text('"--port"\n')
    assert vf.missing_flags({"--port", "--nope"}, tmp_path) == ["--nope"]


def test_cli_fails_when_source_is_absent(tmp_path, capsys):
    assert vf.main(["--src", str(tmp_path / "nope")]) == 2
    assert "not found" in capsys.readouterr().err


def test_cli_passes_against_a_source_that_has_every_flag(tmp_path):
    (tmp_path / "flags.py").write_text("\n".join(f'"{f}"' for f in sorted(vf.matrix_flags())))
    assert vf.main(["--src", str(tmp_path)]) == 0


def test_cli_lists_missing_flags_and_fails(tmp_path, capsys):
    (tmp_path / "flags.py").write_text('"--port"\n')
    assert vf.main(["--src", str(tmp_path)]) == 1
    assert "--max-concurrency" in capsys.readouterr().out


def test_real_pinned_source_has_every_flag_when_it_is_checked_out():
    src = vf.DEFAULT_SRC
    if not src.is_dir():
        import pytest
        pytest.skip("vllm_src not fetched (day4-vllm-internals/fetch_vllm_source.sh)")
    assert vf.missing_flags(vf.matrix_flags(), src) == []
    assert mx.MAX_MODEL_LEN  # the matrix module imported cleanly alongside
