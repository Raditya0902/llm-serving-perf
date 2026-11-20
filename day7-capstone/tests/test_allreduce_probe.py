import pytest

import allreduce_probe as ap


def test_bus_bandwidth_follows_the_nccl_tests_convention_for_two_ranks():
    # 1 GB in 10 ms between 2 GPUs: algorithm bandwidth 100 GB/s, bus = 2*(1/2) * 100 = 100 GB/s
    assert ap.bus_bandwidth_gb_s(10**9, 0.01, 2) == pytest.approx(100.0)


def test_bus_bandwidth_scales_with_rank_count():
    assert ap.bus_bandwidth_gb_s(10**9, 0.01, 4) == pytest.approx(150.0)


@pytest.mark.parametrize("seconds,ranks", [(0, 2), (-1, 2), (0.01, 1)])
def test_bus_bandwidth_rejects_degenerate_input(seconds, ranks):
    with pytest.raises(ValueError):
        ap.bus_bandwidth_gb_s(10**6, seconds, ranks)
