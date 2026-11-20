import json

import pytest

from gpu import busyfrac_summary


def _write(root, name, busy, tok_s=None):
    record = {"file": name, "devices": {"0": {"busy_fraction": busy, "gap_p50_us": 10.0, "gap_p95_us": 50.0}}}
    (root / f"{name}.busy.json").write_text(json.dumps(record))
    if tok_s is not None:
        (root / f"{name}.json").write_text(json.dumps({"output_throughput": tok_s}))


def test_busy_change_between_seq_limits_is_in_points(tmp_path):
    _write(tmp_path, "FLASH_ATTN_s256_512x128_d200", 0.80, 9000)
    _write(tmp_path, "FLASH_ATTN_s256_512x128_d600", 0.82, 9100)
    _write(tmp_path, "FLASH_ATTN_s1024_512x128_d200", 0.65, 8000)
    result = busyfrac_summary.summarize(tmp_path)
    drop = result["busy_change_low_to_high_seqs"][0]
    assert drop["low_seqs"] == 256 and drop["high_seqs"] == 1024
    assert drop["busy_change_points"] == pytest.approx(-16.0)
    cell = next(c for c in result["cells"] if c["seqs"] == 256)
    assert cell["windows"] == 2 and cell["busy_fraction"] == pytest.approx(0.81) and cell["tok_s"] == pytest.approx(9050)


def test_files_without_a_busy_value_or_odd_names_are_skipped(tmp_path):
    (tmp_path / "junk.busy.json").write_text("{}")
    (tmp_path / "TRITON_ATTN_s256_512x128_d200.busy.json").write_text(json.dumps({"error": "x"}))
    assert busyfrac_summary.gather(tmp_path) == []


def test_a_single_seq_limit_has_no_change_row(tmp_path):
    _write(tmp_path, "TRITON_ATTN_s256_128x128_d200", 0.9)
    assert busyfrac_summary.summarize(tmp_path)["busy_change_low_to_high_seqs"] == []


def test_forced_flash_attention_generation_in_the_tag_is_parsed(tmp_path):
    _write(tmp_path, "FLASH_ATTN-fa3_s256_512x128_d200", 0.9)
    rows = busyfrac_summary.gather(tmp_path)
    assert rows[0]["backend"] == "FLASH_ATTN-fa3"
