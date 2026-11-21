from __future__ import annotations

import shutil

import run_experiments as rx
import plot_results as pr


def test_all_plots_render_from_smoke_results(tmp_path) -> None:
    smoke = tmp_path / "smoke"
    rx._main(["--exp", "all", "--smoke", "--out", str(smoke)])
    results = tmp_path / "results"
    results.mkdir()
    for f in smoke.glob("exp_*_smoke.json"):  # plots read the non-smoke names
        shutil.copy(f, results / f.name.replace("_smoke", ""))
    out = tmp_path / "figs"
    pr._main(["--results", str(results), "--out", str(out)])
    names = {p.name for p in out.iterdir()}
    assert {"exp_a_strategies.png", "exp_b_batch_size.png", "exp_c_kv_exhaustion.png",
            "exp_d_long_prompt.png", "exp_e_priorities.png", "exp_f_throughput_vs_p95.png",
            "timeline.png"} <= names
    assert all((out / n).stat().st_size > 5_000 for n in names)
