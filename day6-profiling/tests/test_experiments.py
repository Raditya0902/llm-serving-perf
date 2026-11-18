import pytest

from experiments import EXPERIMENTS, Experiment, get_experiment


def test_registry_contains_the_six_assigned_experiments_and_two_variants():
    ids = [e.exp_id for e in EXPERIMENTS]
    assert ids == ["E1", "E2", "E3", "E4", "E5", "E6", "O1", "O2"]


def test_every_experiment_has_a_hypothesis_and_baseline_among_its_values():
    for exp in EXPERIMENTS:
        assert exp.hypothesis.strip(), exp.exp_id
        assert exp.baseline in exp.values, exp.exp_id
        assert len(exp.values) >= 2, exp.exp_id


def test_scheduler_knobs_are_not_runnable_locally():
    runnable = {e.exp_id: e.runnable_on for e in EXPERIMENTS}
    assert runnable["E1"] == runnable["E2"] == runnable["E3"] == "local"
    assert runnable["E4"] == runnable["E5"] == runnable["E6"] == "gpu"


def test_get_experiment_returns_by_id_and_rejects_unknown():
    assert get_experiment("E2").variable == "prompt_len"
    with pytest.raises(KeyError):
        get_experiment("E99")


def test_experiment_rejects_baseline_outside_values():
    with pytest.raises(ValueError, match="baseline"):
        Experiment("X", "n", "v", (1, 2), 3, "m", "h", "local")


def test_experiment_rejects_empty_hypothesis_and_bad_runnable_on():
    with pytest.raises(ValueError, match="hypothesis"):
        Experiment("X", "n", "v", (1, 2), 1, "m", " ", "local")
    with pytest.raises(ValueError, match="runnable_on"):
        Experiment("X", "n", "v", (1, 2), 1, "m", "h", "tpu")
