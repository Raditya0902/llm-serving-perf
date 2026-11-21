import json

import pytest

from conftest import SYNTHETIC_NOTE
from plot_results import main


def _exp_rows(labels, baseline, controls=()):
    return [
        {"label": str(label), "requests_per_s": 1.0 + i, "requests_per_s_min": 0.9 + i,
         "requests_per_s_max": 1.1 + i, "output_tok_s": 10.0 * (i + 1), "tpot_ms_p50": 15.0,
         "ttft_ms_p50": 100.0 * (i + 1), "is_baseline": str(label) == baseline,
         "is_control": str(label) in controls}
        for i, label in enumerate(labels)
    ]


def _write(root, run, exp, rows):
    (root / run).mkdir(parents=True, exist_ok=True)
    (root / run / f"exp_{exp}.json").write_text(
        json.dumps({"meta": {"note": SYNTHETIC_NOTE}, "rows": rows})
    )


def _profile(root, run):
    (root / run).mkdir(parents=True, exist_ok=True)
    attr = {"matmul_ms": 10.0, "attention_core_ms": 3.0, "other_ms": 1.0}
    (root / run / "profile_summary.json").write_text(
        json.dumps({"device_attribution": {"decode": attr, "prefill": attr}})
    )


@pytest.fixture()
def measured(tmp_path):
    root = tmp_path / "measured"
    for run in ("mps", "mps-r2"):
        _write(root, run, "e1", _exp_rows([1, 2, 4, 8], "1"))
        _write(root, run, "e2", _exp_rows([128, 512], "512"))
    _write(root, "mps", "e3", _exp_rows([32, 128], "128"))
    _write(root, "mps", "o1", _exp_rows(["day1", "day1_control", "deferred_sync"], "day1", {"day1_control"}))
    _write(root, "mps", "o2", _exp_rows(["sdpa", "eager"], "sdpa"))
    for run in ("mps-baseline", "mps-eager"):
        _profile(root, run)
    return root


def test_main_writes_every_figure_from_synthetic_inputs(measured, tmp_path):
    out = tmp_path / "plots"

    assert main(["--measured", str(measured), "--out", str(out)]) == 0

    names = {p.name for p in out.iterdir()}
    assert names == {"profile_breakdown.png", "e1_batching.png", "e2_e3_scaling.png", "o1_o2_variants.png"}
    assert all((out / n).stat().st_size > 1000 for n in names)
