import json

from run_meta import collect_meta, config_fingerprint


def test_config_fingerprint_is_stable_and_order_independent():
    a = config_fingerprint({"x": 1, "y": [1, 2]})
    b = config_fingerprint({"y": [1, 2], "x": 1})

    assert a == b
    assert len(a) == 12


def test_config_fingerprint_changes_when_config_changes():
    assert config_fingerprint({"x": 1}) != config_fingerprint({"x": 2})


def test_collect_meta_has_provenance_fields_and_is_json_serialisable(tiny_model):
    import torch

    meta = collect_meta(torch.device("cpu"), tiny_model, {"phase": "decode"}, smoke=True)

    for key in ("device", "dtype", "torch", "transformers", "machine", "git", "config_id", "smoke"):
        assert key in meta
    assert meta["config"] == {"phase": "decode"}
    assert meta["smoke"] is True
    assert meta["dtype"] == "torch.float32"
    json.dumps(meta)


def test_git_dirty_flag_is_a_bool_or_none_when_git_is_unavailable(monkeypatch):
    import run_meta

    assert run_meta.git_is_dirty() in (True, False)

    def no_git(*_a, **_k):
        raise FileNotFoundError("git")

    monkeypatch.setattr(run_meta.subprocess, "run", no_git)
    assert run_meta.git_is_dirty() is None
    assert run_meta.git_short_hash() == "unknown"
