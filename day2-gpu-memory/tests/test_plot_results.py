from __future__ import annotations

import plot_results as pr
import run_experiments as rx


def _write_smoke_results(out_dir):
    meta = rx.run_metadata("cpu", rx.SMOKE, True)
    rx.write_results("exp_a", rx.run_exp_a("cpu", rx.SMOKE), meta, out_dir)
    rx.write_results("exp_b", rx.run_exp_b("cpu", rx.SMOKE), meta, out_dir)
    rx.write_results("exp_c", rx.run_exp_c(rx.SMOKE), meta, out_dir)


def test_all_plots_render_from_cpu_smoke_results(tmp_path):
    _write_smoke_results(tmp_path)
    pr.plot_growth(tmp_path)
    pr.plot_exp_a(tmp_path, tmp_path)
    pr.plot_exp_b(tmp_path, tmp_path)
    pr.plot_exp_c(tmp_path, tmp_path)
    for name in ("memory_growth", "exp_a_memory_scaling", "exp_b_allocated_vs_reserved", "exp_c_paged_vs_contiguous"):
        assert (tmp_path / f"{name}.png").stat().st_size > 0


def test_cli_renders_everything(tmp_path, monkeypatch):
    import sys

    _write_smoke_results(tmp_path)
    monkeypatch.setattr(sys, "argv", ["plot_results.py", "--results", str(tmp_path)])
    pr._main()
    assert (tmp_path / "exp_c_paged_vs_contiguous.png").exists()
