"""Parse `ncu --csv --page raw` output into one record per kernel launch, and compare a pure-decode launch with a mixed one.

    python gpu/ncu_summary.py pure.csv mixed.csv [prefill_only.csv]   # prints the comparison as JSON; the third file adds the bytes/grid ratios

Units are converted whatever the unit row says (ncu prints nsecond/usecond/msecond/second and byte/Kbyte/Mbyte/Gbyte).
A profile that was refused because the container may not read GPU performance counters fails loudly with that error name.
"""
from __future__ import annotations

import csv
import io
import json
import re
import sys
from dataclasses import dataclass

TIME_TO_US = {"nsecond": 1e-3, "usecond": 1.0, "msecond": 1e3, "second": 1e6, "ns": 1e-3, "us": 1.0, "ms": 1e3, "s": 1e6}
BYTES_TO_MB = {"byte": 1e-6, "Kbyte": 1e-3, "Mbyte": 1.0, "Gbyte": 1e3}
COLUMNS = {
    "kernel": ("Kernel Name",),
    "block": ("launch__block_size", "Block Size"),
    "grid": ("launch__grid_size", "Grid Size"),
    "duration": ("gpu__time_duration.sum",),
    "dram_pct": ("gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed", "dram__throughput.avg.pct_of_peak_sustained_elapsed"),
    "warps_pct": ("sm__warps_active.avg.pct_of_peak_sustained_active",),
    "waves": ("launch__waves_per_multiprocessor",),
    "dram_read": ("dram__bytes_read.sum",),
}


@dataclass(frozen=True)
class Record:
    kernel: str
    grid_size: int | None
    block_size: int | None
    duration_us: float
    dram_pct_of_peak: float | None
    warps_active_pct: float | None
    waves_per_sm: float | None
    dram_read_mb: float | None


def _number(text: str) -> float | None:
    text = text.replace(",", "").strip()
    try:
        return float(text)
    except ValueError:
        return None


def _product(text: str) -> int | None:
    numbers = [int(n) for n in re.findall(r"\d+", text)]
    if not numbers:
        return None
    result = 1
    for n in numbers:
        result *= n
    return result


def _pick(header: list[str], names: tuple[str, ...]) -> int | None:
    for name in names:
        if name in header:
            return header.index(name)
    return None


def parse(text: str) -> list[Record]:
    if "ERR_NVGPUCTRPERM" in text:
        raise ValueError("ERR_NVGPUCTRPERM: this container may not read GPU performance counters, so Nsight Compute cannot profile here")
    lines = [line for line in text.splitlines() if line.strip() and not line.startswith("==")]
    rows = list(csv.reader(io.StringIO("\n".join(lines))))
    if len(rows) < 2 or "Kernel Name" not in rows[0]:
        raise ValueError("no kernel records in the ncu output (was --csv --page raw used, and did a kernel match the filter?)")
    header, rest = rows[0], rows[1:]
    units = rest[0] if rest and rest[0][0] == "" else None
    data = rest[1:] if units is not None else rest
    index = {key: _pick(header, names) for key, names in COLUMNS.items()}
    if index["kernel"] is None or index["duration"] is None or not data:
        raise ValueError("no kernel records in the ncu output (missing Kernel Name or gpu__time_duration.sum)")

    def unit_of(key: str) -> str:
        return units[index[key]] if units is not None and index[key] is not None else ""

    records = []
    for row in data:
        def value(key: str) -> float | None:
            return _number(row[index[key]]) if index[key] is not None and index[key] < len(row) else None

        duration = value("duration")
        if duration is None:
            continue
        read = value("dram_read")
        records.append(Record(
            kernel=row[index["kernel"]],
            grid_size=_product(row[index["grid"]]) if index["grid"] is not None else None,
            block_size=_product(row[index["block"]]) if index["block"] is not None else None,
            duration_us=duration * TIME_TO_US.get(unit_of("duration"), 1.0),
            dram_pct_of_peak=value("dram_pct"), warps_active_pct=value("warps_pct"), waves_per_sm=value("waves"),
            dram_read_mb=None if read is None else read * BYTES_TO_MB.get(unit_of("dram_read"), 1.0)))
    if not records:
        raise ValueError("no kernel records in the ncu output")
    return records


def _bandwidth_gb_s(record: Record) -> float | None:
    if record.dram_read_mb is None or record.duration_us <= 0:
        return None
    return record.dram_read_mb / (record.duration_us * 1e-6) / 1000


def compare(pure: Record, mixed: Record) -> dict:
    return {
        "kernels": {"pure": pure.kernel, "mixed": mixed.kernel},
        "duration_us": {"pure": pure.duration_us, "mixed": mixed.duration_us},
        "slowdown": mixed.duration_us / pure.duration_us,
        "dram_pct_of_peak": {"pure": pure.dram_pct_of_peak, "mixed": mixed.dram_pct_of_peak},
        "achieved_bandwidth_gb_s": {"pure": _bandwidth_gb_s(pure), "mixed": _bandwidth_gb_s(mixed)},
        "warps_active_pct": {"pure": pure.warps_active_pct, "mixed": mixed.warps_active_pct},
        "waves_per_sm": {"pure": pure.waves_per_sm, "mixed": mixed.waves_per_sm},
        "grid_size": {"pure": pure.grid_size, "mixed": mixed.grid_size},
    }


def extra_over_prefill(pure: Record, mixed: Record, prefill_only: Record) -> dict:
    """What the decodes add to a mixed launch: mixed minus the same launch without them, as multiples of the pure-decode launch.

    `bytes_ratio` is the decodes' extra DRAM read bytes over the pure-decode launch's (about the GQA group size if the KV is
    streamed once per query head instead of once per KV head); `grid_ratio` is mixed over pure thread blocks.
    """
    out: dict = {"grid_ratio": None, "bytes_ratio": None, "extra_ms": (mixed.duration_us - prefill_only.duration_us) / 1000,
                 "pure_ms": pure.duration_us / 1000}
    if mixed.grid_size and pure.grid_size:
        out["grid_ratio"] = mixed.grid_size / pure.grid_size
    if None not in (mixed.dram_read_mb, prefill_only.dram_read_mb, pure.dram_read_mb) and pure.dram_read_mb:
        out["bytes_ratio"] = (mixed.dram_read_mb - prefill_only.dram_read_mb) / pure.dram_read_mb
    return out


def main(argv: list[str]) -> int:
    if len(argv) not in (2, 3):
        print(__doc__, file=sys.stderr)
        return 2
    longest = [max(parse(open(path).read()), key=lambda r: r.duration_us) for path in argv]
    result = compare(longest[0], longest[1])
    if len(longest) == 3:
        result["decodes_added_to_mixed"] = extra_over_prefill(longest[0], longest[1], longest[2])
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
