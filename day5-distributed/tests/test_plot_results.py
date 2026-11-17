from __future__ import annotations

from pathlib import Path

import pytest

import plot_results as pr
import run_experiments as rx


@pytest.fixture(scope="module")
def smoke_results(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("smoke_results")
    rx.main(["--exp", "all", "--smoke", "--out", str(out)])
    return out


def test_make_all_writes_every_figure_from_smoke_results(smoke_results: Path) -> None:
    paths = pr.make_all(smoke_results, "_smoke")

    assert {p.name for p in paths} == {
        "tp_diagram.png", "exp_a_collectives.png", "exp_b_speedup.png",
        "exp_b_comm_fraction.png", "exp_c_model.png",
    }
    assert all(p.exists() and p.stat().st_size > 1000 for p in paths)


def test_cli_plots_smoke_files(smoke_results: Path, capsys: pytest.CaptureFixture[str]) -> None:
    pr.main(["--results", str(smoke_results), "--suffix", "_smoke"])

    assert capsys.readouterr().out.count("wrote ") == 5


def test_cli_reports_missing_results_clearly(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        pr.main(["--results", str(tmp_path)])


@pytest.mark.parametrize(("value", "expected"), [(512, "512B"), (1024, "1KB"), (4 << 20, "4MB"), (1 << 30, "1GB")])
def test_human_bytes_formats_binary_units(value: float, expected: str) -> None:
    assert pr._human_bytes(value) == expected
