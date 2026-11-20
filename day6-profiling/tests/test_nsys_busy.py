import json
import sqlite3

import pytest

from gpu import nsys_busy


def test_merge_joins_overlapping_and_touching_intervals():
    merged = nsys_busy.merge_intervals([(10, 20), (15, 30), (30, 40), (50, 60)])
    assert merged == [(10, 40), (50, 60)]


def test_merge_drops_empty_and_inverted_intervals():
    assert nsys_busy.merge_intervals([(5, 5), (9, 3), (1, 2)]) == [(1, 2)]


def test_busy_fraction_counts_union_not_sum():
    # two kernels overlap on 5 ns; busy = 15 of a 25 ns span
    stats = nsys_busy.busy_stats([(0, 10), (5, 15), (20, 25)])
    assert stats["busy_fraction"] == pytest.approx(20 / 25)
    assert stats["gaps"] == 1
    assert stats["gap_max_us"] == pytest.approx(5 / 1e3)


def test_busy_fraction_is_one_without_gaps():
    stats = nsys_busy.busy_stats([(0, 10), (10, 20)])
    assert stats["busy_fraction"] == pytest.approx(1.0)
    assert stats["gap_max_us"] == 0.0


def test_no_activity_gives_none_not_a_crash():
    assert nsys_busy.busy_stats([])["busy_fraction"] is None


def _make_db(path, kernels, memcpys=()):
    with sqlite3.connect(path) as db:
        for table, rows in (("CUPTI_ACTIVITY_KIND_KERNEL", kernels), ("CUPTI_ACTIVITY_KIND_MEMCPY", memcpys)):
            db.execute(f"CREATE TABLE {table} (deviceId INTEGER, start INTEGER, end INTEGER)")
            db.executemany(f"INSERT INTO {table} VALUES (?,?,?)", rows)


def test_devices_are_reported_separately_and_memcpy_counts(tmp_path):
    path = str(tmp_path / "t.sqlite")
    _make_db(path, kernels=[(0, 0, 10), (1, 0, 50)], memcpys=[(0, 10, 20)])
    result = nsys_busy.summarize_file(path)
    assert result["devices"]["0"]["busy_fraction"] == pytest.approx(1.0)
    assert result["devices"]["1"]["busy_ms"] == pytest.approx(50 / 1e6)


def test_missing_kernel_table_fails_loudly(tmp_path):
    path = str(tmp_path / "empty.sqlite")
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE other (x INTEGER)")
    with pytest.raises(ValueError, match="kernel table"):
        nsys_busy.load_by_device(path)


def test_main_prints_json_and_returns_error_status(tmp_path, capsys):
    good = str(tmp_path / "g.sqlite")
    _make_db(good, kernels=[(0, 0, 10), (0, 20, 30)])
    bad = str(tmp_path / "missing.sqlite")
    status = nsys_busy.main([good, bad])
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert lines[0]["devices"]["0"]["busy_fraction"] == pytest.approx(20 / 30)
    assert "error" in lines[1]
    assert status == 1
