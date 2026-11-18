import pytest
import torch

from alloc_probe import AllocReport, build_alloc_report, measure_allocations
from launch_probe import LaunchReport, launch_verdict, measure_launch

CPU = torch.device("cpu")


@pytest.mark.parametrize(
    ("ratio", "verdict"),
    [(1.0, "cpu-bound"), (0.9, "cpu-bound"), (0.7, "mixed"), (0.5, "mixed"), (0.2, "device-bound")],
)
def test_launch_verdict_thresholds(ratio, verdict):
    assert launch_verdict(ratio) == verdict


def test_measure_launch_returns_consistent_report_on_cpu(tiny_model, prompt_ids):
    report = measure_launch(tiny_model, prompt_ids, n_steps=4)

    assert isinstance(report, LaunchReport)
    assert report.n_steps == 4
    assert 0 < report.enqueue_ms_median <= report.step_ms_median * 1.5
    assert report.enqueue_fraction == pytest.approx(
        report.enqueue_ms_median / report.step_ms_median
    )
    assert report.verdict in {"cpu-bound", "mixed", "device-bound"}


def test_measure_launch_rejects_zero_steps(tiny_model, prompt_ids):
    with pytest.raises(ValueError, match="n_steps"):
        measure_launch(tiny_model, prompt_ids, n_steps=0)


def test_build_alloc_report_flags_growth_steps():
    report = build_alloc_report(
        allocated_per_step=[1000, 1000, 1010, 1020, 1020], expected_kv_bytes_per_step=10
    )

    assert report.deltas == (0, 10, 10, 0)
    assert report.steps_with_growth == 2
    assert report.mean_growth_per_step == pytest.approx(5.0)
    assert report.growth_matches_kv is False  # half the steps are flat, mean 5 != 10


def test_build_alloc_report_recognises_pure_kv_growth():
    report = build_alloc_report([100, 110, 120, 130], expected_kv_bytes_per_step=10)

    assert report.growth_matches_kv is True


def test_build_alloc_report_needs_two_samples():
    with pytest.raises(ValueError, match="two"):
        build_alloc_report([1], expected_kv_bytes_per_step=1)


def test_measure_allocations_is_unavailable_on_cpu(tiny_model, prompt_ids):
    report = measure_allocations(tiny_model, prompt_ids, n_steps=3)

    assert isinstance(report, AllocReport)
    assert report.available is False
    assert "cpu" in report.reason.lower()
