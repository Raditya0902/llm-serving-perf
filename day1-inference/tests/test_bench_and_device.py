import pytest
import torch

from bench import Stats, run_repeated, summarize
from device import MemoryTracker, pick_device, release_cached_memory, sync
from prompts import make_prompt_ids


class FakeTokenizer:
    """One char == one token, enough to test exact-length prompt construction."""

    def __call__(self, text, add_special_tokens=False, return_tensors=None):
        ids = [ord(c) % 100 for c in text]
        return type("Enc", (), {"input_ids": torch.tensor([ids])})()


@pytest.mark.parametrize("n", [1, 128, 512, 1024, 2048])
def test_prompt_builder_exact_length(n):
    ids = make_prompt_ids(FakeTokenizer(), n)

    assert ids.shape == (1, n)


def test_prompt_builder_rejects_non_positive_length():
    with pytest.raises(ValueError):
        make_prompt_ids(FakeTokenizer(), 0)


def test_summarize_reports_median_min_max():
    stats = summarize([3.0, 1.0, 2.0])

    assert stats == Stats(median=2.0, minimum=1.0, maximum=3.0)


def test_summarize_rejects_empty():
    with pytest.raises(ValueError):
        summarize([])


def test_run_repeated_discards_warmup_runs():
    calls = []

    def fn():
        calls.append(1)
        return len(calls)

    results = run_repeated(fn, warmup=2, repeats=3)

    assert len(calls) == 5
    assert results == (3, 4, 5)


def test_run_repeated_rejects_bad_counts():
    with pytest.raises(ValueError):
        run_repeated(lambda: 1, warmup=-1, repeats=1)
    with pytest.raises(ValueError):
        run_repeated(lambda: 1, warmup=0, repeats=0)


def test_pick_device_honours_preference_and_falls_back():
    assert pick_device("cpu").type == "cpu"
    assert pick_device().type in {"cuda", "mps", "cpu"}
    with pytest.raises(ValueError):
        pick_device("tpu")


def test_eos_token_ids_rejects_missing_tokens():
    from model_io import eos_token_ids

    class Tok:
        unk_token_id = 0

        def convert_tokens_to_ids(self, tokens):
            return [5, None]

    with pytest.raises(ValueError):
        eos_token_ids(Tok())


def test_sync_is_noop_safe_on_cpu():
    sync(torch.device("cpu"))


def test_release_cached_memory_is_safe_on_cpu():
    release_cached_memory(torch.device("cpu"))


def test_memory_tracker_tracks_running_peak_on_cpu():
    tracker = MemoryTracker(torch.device("cpu"))
    tracker.reset()
    tracker.sample()
    block = torch.zeros(10_000_000)  # ~40 MB
    tracker.sample()
    del block

    assert tracker.peak_bytes > 0
