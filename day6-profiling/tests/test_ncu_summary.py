"""Parsing of `ncu --csv --page raw` output (synthetic CSV, no GPU)."""
from __future__ import annotations

import pytest

from gpu import ncu_summary as ns

HEADER = ('"ID","Process ID","Kernel Name","Block Size","Grid Size","gpu__time_duration.sum",'
          '"gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed","sm__warps_active.avg.pct_of_peak_sustained_active",'
          '"launch__waves_per_multiprocessor","dram__bytes_read.sum"')
UNITS = '"","","","","","usecond","%","%","","Mbyte"'


def _csv(*rows):
    return "\n".join(["==PROF== some banner line", HEADER, UNITS, *rows]) + "\n"


def test_parse_reads_one_record_per_kernel_launch_skipping_banner_and_units_rows():
    text = _csv('"0","10","flash_fwd_splitkv_kernel","128","3584","410.5","88.2","31.0","4.5","284.0"',
                '"1","10","flash_fwd_kernel","128","3584","2300.0","12.4","18.5","0.3","284.0"')
    records = ns.parse(text)
    assert [r.kernel for r in records] == ["flash_fwd_splitkv_kernel", "flash_fwd_kernel"]
    assert records[0].duration_us == pytest.approx(410.5) and records[0].dram_pct_of_peak == pytest.approx(88.2)
    assert records[1].grid_size == 3584 and records[1].waves_per_sm == pytest.approx(0.3)


def test_parse_converts_units_to_us_and_mb_whatever_the_ncu_unit_row_says():
    text = ('"ID","Kernel Name","gpu__time_duration.sum","dram__bytes_read.sum"\n'
            '"","","msecond","Gbyte"\n"0","k","1.5","2.0"\n')
    record = ns.parse(text)[0]
    assert record.duration_us == pytest.approx(1500.0) and record.dram_read_mb == pytest.approx(2000.0)


def test_compare_gives_the_slowdown_and_the_metric_that_moved():
    pure = ns.parse(_csv('"0","1","flash_fwd_splitkv_kernel","128","3584","400","90","30","4","284"'))[0]
    mixed = ns.parse(_csv('"0","1","flash_fwd_kernel","128","3584","2000","10","18","0.3","284"'))[0]
    out = ns.compare(pure, mixed)
    assert out["slowdown"] == pytest.approx(5.0)
    assert out["dram_pct_of_peak"] == {"pure": 90.0, "mixed": 10.0}
    assert out["achieved_bandwidth_gb_s"]["pure"] == pytest.approx(284 / 400e-6 / 1000)


def test_parse_of_a_profile_that_failed_names_the_counter_permission_error():
    with pytest.raises(ValueError, match="ERR_NVGPUCTRPERM"):
        ns.parse("==ERROR== ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA GPU Performance Counters\n")


def test_parse_of_empty_output_is_an_error_not_an_empty_list():
    with pytest.raises(ValueError, match="no kernel records"):
        ns.parse("==PROF== Disconnected from process\n")
