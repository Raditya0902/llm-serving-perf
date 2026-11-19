"""Grid and analysis of the attention sweep (pure Python; the kernel timing needs a GPU)."""
from __future__ import annotations

import json

import pytest

from gpu import attn_sweep as sw


def _rows(k=2e-7, base=0.8, ctx_extra=None, prefills=(2, 128, 512, 1024, 2048, 4096, 7600), decodes=(64, 256, 512), ctxs=(512, 3584)):
    """Synthetic rows following H-dead: decode share d(P) = base(ctx, decodes) + k * decodes * P."""
    rows = []
    for ctx in ctxs:
        for n in decodes:
            for p in prefills:
                pure = 0.001 * n * (1 + ctx / 512)
                prefill_only = 0.0004 * p
                d = pure * 6.5 + k * n * p * (ctx_extra or {}).get(ctx, 1.0)
                rows.append({"ctx": ctx, "decodes": n, "prefill": p, "pure_ms": pure, "prefill_only_ms": prefill_only,
                             "mixed_ms": prefill_only + d})
    return rows


def test_grid_is_the_full_cross_product_in_a_stable_order():
    points = sw.grid([512, 3584], [64, 256], [2, 7600])
    assert len(points) == 8
    assert points[0] == sw.SweepPoint(512, 64, 2) and points[-1] == sw.SweepPoint(3584, 256, 7600)


def test_decode_share_is_mixed_minus_prefill_only():
    row = {"mixed_ms": 7.0, "prefill_only_ms": 1.5, "pure_ms": 0.9}
    assert sw.decode_share(row) == pytest.approx(5.5)
    assert sw.ratio(row) == pytest.approx(5.5 / 0.9)


def test_excess_is_zero_at_the_reference_and_positive_beyond_it():
    excess = {(r["ctx"], r["decodes"], r["prefill"]): e for r, e in sw.excess_rows(_rows())}
    assert excess[(512, 256, 1024)] == pytest.approx(0.0)
    assert excess[(512, 256, 7600)] == pytest.approx(2e-7 * 256 * (7600 - 1024))
    assert (512, 256, 512) not in excess  # only P above the reference


def test_fit_recovers_the_slope_and_a_perfect_r2_on_dead_block_data():
    fit = sw.fit_excess(_rows())
    assert fit["k_ms_per_decode_token"] == pytest.approx(2e-7, rel=1e-6) and fit["r2"] == pytest.approx(1.0)


def test_fit_r2_is_low_when_the_excess_does_not_scale_with_decodes_times_p():
    rows = _rows()
    for r in rows:  # a constant 5 ms bump for every P above the reference, whatever the decodes and P
        if r["prefill"] > 1024:
            r["mixed_ms"] = r["prefill_only_ms"] + 6.5 * r["pure_ms"] + 5.0
    assert sw.fit_excess(rows)["r2"] < 0.7


def test_context_consistency_is_one_when_excess_is_independent_of_context():
    result = sw.context_consistency(_rows())
    assert result["median_ratio"] == pytest.approx(1.0) and result["within_25pct"] and not result["refuted"]


def test_context_consistency_flags_a_context_dependent_excess_as_refuted():
    result = sw.context_consistency(_rows(ctx_extra={3584: 4.0}))
    assert result["median_ratio"] == pytest.approx(0.25) and not result["within_25pct"] and result["refuted"]


def test_dead_share_at_small_p_is_below_five_percent_when_the_dead_cost_is_small():
    shares = sw.dead_share_small_p(_rows(), fit=sw.fit_excess(_rows()), max_p=512)
    assert shares and all(s < 0.05 for s in shares.values())


def test_analyse_bundles_the_three_predictions():
    out = sw.analyse(_rows())
    assert set(out) >= {"fit", "context", "small_p", "verdict"}
    assert out["verdict"]["supported"] is True


def test_dry_run_lists_the_points_without_touching_torch(capsys):
    assert sw.main(["--dry-run", "--ctx", "512", "--decodes", "64", "--prefill", "2", "7600"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["points"] == 2 and out["fa_version"] == 2


def test_main_rejects_a_bad_fa_version():
    with pytest.raises(SystemExit):
        sw.main(["--dry-run", "--fa-version", "4"])


def _floor_rows(a=3.5e-3, k=1.0e-6, ctxs=(512, 1024, 2048), decodes=(64, 128, 256, 384, 512), prefills=(2, 128, 512, 1024, 2048, 4096, 7600)):
    """Synthetic rows that follow d(P) = max(d_live, n (a + k P)); d_live grows with ctx and n."""
    rows = []
    for ctx in ctxs:
        for n in decodes:
            live = 0.0015 * n * (ctx / 512)
            for p in prefills:
                d = max(live, n * (a + k * p))
                po = 0.0002 * p
                rows.append({"ctx": ctx, "decodes": n, "prefill": p, "pure_ms": live / 6.5, "prefill_only_ms": po, "mixed_ms": po + d})
    return rows


def test_floor_model_recovers_its_parameters_from_the_fitted_context():
    model = sw.fit_floor_model(_floor_rows(), 512)
    assert model["a_ms_per_decode"] == pytest.approx(3.5e-3, rel=1e-6) and model["k_ms_per_decode_token"] == pytest.approx(1.0e-6, rel=1e-6)


def test_floor_model_predicts_unseen_contexts_when_the_data_follow_it():
    rows = _floor_rows()
    out = sw.floor_model_errors(rows, sw.fit_floor_model(rows, 512), ctxs=[1024, 2048])
    assert out["supported"] and out["median"] < 1e-9 and out["points"] > 0


def test_floor_model_is_rejected_when_other_contexts_do_not_follow_it():
    rows = _floor_rows()
    for r in rows:
        if r["ctx"] == 2048:  # a context where the dead-block cost is twice what the 512 fit says
            r["mixed_ms"] = r["prefill_only_ms"] + 2.0 * (r["mixed_ms"] - r["prefill_only_ms"])
    out = sw.floor_model_errors(rows, sw.fit_floor_model(rows, 512), ctxs=[1024, 2048])
    assert not out["supported"] and out["max"] > 0.3


def test_excess_by_context_shrinks_when_live_work_masks_the_dead_blocks():
    excess = sw.excess_by_context(_floor_rows(), decodes=256, prefill=7600)
    assert excess[512] >= excess[1024] > excess[2048] >= 0  # non-increasing, strictly smaller once masked


def test_fit_floor_model_needs_enough_points():
    assert sw.fit_floor_model(_floor_rows(prefills=(2, 128)), 512)["a_ms_per_decode"] is None
