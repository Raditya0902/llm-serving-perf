import json

import pytest
import torch

from profile_run import module_labels, parse_args, main


def test_module_labels_attach_and_remove_hooks(tiny_model):
    before = sum(len(m._forward_hooks) for m in tiny_model.modules())

    with module_labels(tiny_model) as labelled:
        assert labelled == 2 * len(tiny_model.model.layers)
        during = sum(len(m._forward_hooks) for m in tiny_model.modules())

    after = sum(len(m._forward_hooks) for m in tiny_model.modules())
    assert during == before + labelled
    assert after == before


def test_parse_args_defaults():
    args = parse_args([])

    assert (args.prompt_len, args.new_tokens, args.attn_impl, args.phase) == (512, 33, "sdpa", "both")


def test_smoke_run_writes_all_artifacts(tmp_path):
    code = main(
        ["--smoke", "--device", "cpu", "--prompt-len", "16", "--new-tokens", "6",
         "--warmup", "1", "--tag", "t", "--out", str(tmp_path)]
    )

    assert code == 0
    out = tmp_path / "t"
    for name in ("trace_prefill.json", "trace_decode.json", "ops_prefill.csv",
                 "ops_decode.csv", "profile_summary.json"):
        assert (out / name).is_file(), name
    summary = json.loads((out / "profile_summary.json").read_text())
    decode = summary["phases"]["decode"]
    assert decode["steps_in_window"] == 5
    assert decode["gaps"]["available"] is False  # CPU records no device kernels
    assert "matmul" in decode["category_totals"]
    assert decode["module_cpu_ms_per_step"]["d6::attn"] > 0
    assert summary["meta"]["smoke"] is True
    assert summary["launch_probe"]["verdict"] in {"cpu-bound", "mixed", "device-bound"}
    assert summary["alloc_probe"]["available"] is False


def test_decode_window_excludes_prefill_ops(tmp_path):
    main(["--smoke", "--device", "cpu", "--prompt-len", "16", "--new-tokens", "5",
          "--warmup", "0", "--phase", "decode", "--tag", "d", "--out", str(tmp_path)])

    summary = json.loads((tmp_path / "d" / "profile_summary.json").read_text())
    # one embedding lookup per decode step; a prefill inside the window would add more
    assert summary["phases"]["decode"]["steps_in_window"] == 4
    ops = {r["op"]: r for r in _read_csv(tmp_path / "d" / "ops_decode.csv")}
    assert int(ops["aten::embedding"]["count"]) == 4


def test_main_rejects_too_few_new_tokens(tmp_path):
    with pytest.raises(SystemExit):
        main(["--smoke", "--device", "cpu", "--new-tokens", "2", "--out", str(tmp_path)])


def _read_csv(path):
    import csv

    with path.open() as handle:
        return list(csv.DictReader(handle))


class _FakeProf:
    def __init__(self):
        self.events = []

    def start(self):
        self.events.append("start")

    def stop(self):
        self.events.append("stop")


def test_window_opens_and_closes_the_cuda_profiler_api_around_the_torch_profiler(monkeypatch):
    from profile_run import _Window

    calls = []
    monkeypatch.setattr(torch.cuda.profiler, "start", lambda: calls.append("cuda_start"))
    monkeypatch.setattr(torch.cuda.profiler, "stop", lambda: calls.append("cuda_stop"))
    prof = _FakeProf()

    window = _Window(torch.device("cuda"), prof)
    window.start()
    window.start()  # idempotent
    window.stop()
    window.stop()  # idempotent

    assert calls == ["cuda_start", "cuda_stop"]
    assert prof.events == ["start", "stop"]


def test_window_without_torch_profiler_still_drives_cuda_profiler_api(monkeypatch):
    from profile_run import _Window

    calls = []
    monkeypatch.setattr(torch.cuda.profiler, "start", lambda: calls.append("cuda_start"))
    monkeypatch.setattr(torch.cuda.profiler, "stop", lambda: calls.append("cuda_stop"))

    window = _Window(torch.device("cuda"), None)
    window.start()
    window.stop()

    assert calls == ["cuda_start", "cuda_stop"]


def test_window_on_cpu_never_touches_the_cuda_profiler(monkeypatch):
    from profile_run import _Window

    monkeypatch.setattr(torch.cuda.profiler, "start", lambda: pytest.fail("cuda profiler used on cpu"))
    window = _Window(torch.device("cpu"), _FakeProf())
    window.start()
    window.stop()


def test_run_window_stops_the_profiler_and_removes_hooks_when_generation_raises(tiny_model, prompt_ids, monkeypatch):
    import profile_run
    from profile_run import ProfileConfig, run_window

    def boom(*_a, **_k):
        raise RuntimeError("generation failed")

    cfg = ProfileConfig("prefill", 12, 5, 0, "sdpa")
    monkeypatch.setattr(profile_run, "generate_cached", boom)
    before = sum(len(m._forward_hooks) for m in tiny_model.modules())

    with pytest.raises(RuntimeError, match="generation failed"):
        run_window(tiny_model, prompt_ids, cfg, (1, 2))

    assert sum(len(m._forward_hooks) for m in tiny_model.modules()) == before
    monkeypatch.undo()
    run_window(tiny_model, prompt_ids, cfg, (1, 2))  # a leaked active profiler would raise here


def test_nsys_flag_requires_a_cuda_device(tmp_path):
    with pytest.raises(SystemExit, match="cuda"):
        main(["--smoke", "--device", "cpu", "--nsys", "--out", str(tmp_path)])


@pytest.mark.parametrize("flag", [["--prompt-len", "0"], ["--warmup", "-1"]])
def test_main_rejects_invalid_sizes(flag, tmp_path):
    with pytest.raises(SystemExit):
        main(["--smoke", "--device", "cpu", "--out", str(tmp_path), *flag])
