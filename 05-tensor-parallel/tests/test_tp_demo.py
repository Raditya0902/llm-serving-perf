from __future__ import annotations

import helpers
from dist_utils import spawn_group


def test_tp_demo_verifies_all_layers_against_reference() -> None:
    reports = spawn_group(helpers.worker_tp_demo, 2)

    for rank, lines in enumerate(reports):
        text = "\n".join(lines)
        assert "FAIL" not in text
        assert f"[rank {rank}] column_parallel+all_gather" in text
        assert f"[rank {rank}] row_parallel+all_reduce" in text
        assert f"[rank {rank}] mlp(column->gelu->row)" in text
        assert "1 all_reduce" in text  # the MLP's whole communication bill


def test_tp_demo_works_on_four_ranks() -> None:
    reports = spawn_group(helpers.worker_tp_demo, 4)

    assert all("FAIL" not in "\n".join(lines) for lines in reports)


def test_demo_verdict_does_not_depend_on_the_random_seed() -> None:
    """Regression: an absolute 1e-5 tolerance failed ~half of all seeds on pure fp32 rounding."""
    reports = spawn_group(helpers.worker_tp_demo_many_seeds, 2, tuple(range(1000, 1016)))

    assert reports[0] == 0 and reports[1] == 0  # failures per rank
