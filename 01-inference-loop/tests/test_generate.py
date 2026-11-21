import pytest
import torch

from generate import GenerationResult, generate_cached, generate_uncached

NEW_TOKENS = 20


def test_cached_matches_uncached(tiny_model, prompt_ids):
    uncached = generate_uncached(tiny_model, prompt_ids, NEW_TOKENS, eos_ids=())
    cached = generate_cached(tiny_model, prompt_ids, NEW_TOKENS, eos_ids=())

    assert cached.token_ids == uncached.token_ids
    assert len(cached.token_ids) == NEW_TOKENS


def test_matches_hf_generate_greedy(tiny_model, prompt_ids):
    expected = tiny_model.generate(
        prompt_ids,
        max_new_tokens=NEW_TOKENS,
        min_new_tokens=NEW_TOKENS,
        do_sample=False,
        eos_token_id=None,
        pad_token_id=0,
    )[0, prompt_ids.shape[1]:]

    ours = generate_cached(tiny_model, prompt_ids, NEW_TOKENS, eos_ids=())

    assert list(ours.token_ids) == expected.tolist()


@pytest.mark.parametrize("fn", [generate_cached, generate_uncached])
def test_stops_on_eos_and_includes_it(fn, tiny_model, prompt_ids):
    reference = fn(tiny_model, prompt_ids, NEW_TOKENS, eos_ids=())
    eos = reference.token_ids[4]
    first_hit = reference.token_ids.index(eos)

    result = fn(tiny_model, prompt_ids, NEW_TOKENS, eos_ids=(eos,))

    assert result.token_ids == reference.token_ids[: first_hit + 1]
    assert result.stopped_on_eos is True


@pytest.mark.parametrize("fn", [generate_cached, generate_uncached])
def test_stops_at_max_new_tokens(fn, tiny_model, prompt_ids):
    result = fn(tiny_model, prompt_ids, 7, eos_ids=())

    assert len(result.token_ids) == 7
    assert result.stopped_on_eos is False


@pytest.mark.parametrize("fn", [generate_cached, generate_uncached])
def test_suppress_eos_forces_exact_length(fn, tiny_model, prompt_ids):
    reference = fn(tiny_model, prompt_ids, NEW_TOKENS, eos_ids=())
    eos = reference.token_ids[2]

    result = fn(tiny_model, prompt_ids, NEW_TOKENS, eos_ids=(eos,), suppress_eos=True)

    assert len(result.token_ids) == NEW_TOKENS
    assert eos not in result.token_ids


@pytest.mark.parametrize("fn", [generate_cached, generate_uncached])
def test_timing_fields_are_consistent(fn, tiny_model, prompt_ids):
    result = fn(tiny_model, prompt_ids, 6, eos_ids=())

    assert result.prefill_s > 0
    assert len(result.step_times_s) == 5  # first token comes from prefill
    assert result.total_s == pytest.approx(result.prefill_s + sum(result.step_times_s))
    assert result.decode_s_per_token == pytest.approx(sum(result.step_times_s) / 5)


def test_result_is_immutable(tiny_model, prompt_ids):
    result = generate_cached(tiny_model, prompt_ids, 3, eos_ids=())

    assert isinstance(result, GenerationResult)
    with pytest.raises(Exception):
        result.prefill_s = 0.0  # type: ignore[misc]


def test_cache_grows_by_one_per_step(tiny_model, prompt_ids):
    seen_lengths = []
    generate_cached(
        tiny_model,
        prompt_ids,
        6,
        eos_ids=(),
        on_step=lambda cache_len: seen_lengths.append(cache_len),
    )

    prompt_len = prompt_ids.shape[1]
    assert seen_lengths == [prompt_len + i for i in range(5)]


def test_does_not_mutate_input_ids(tiny_model, prompt_ids):
    before = prompt_ids.clone()

    generate_uncached(tiny_model, prompt_ids, 5, eos_ids=())
    generate_cached(tiny_model, prompt_ids, 5, eos_ids=())

    assert torch.equal(prompt_ids, before)


@pytest.mark.parametrize("fn", [generate_cached, generate_uncached])
def test_rejects_invalid_inputs(fn, tiny_model, prompt_ids):
    with pytest.raises(ValueError):
        fn(tiny_model, prompt_ids, 0, eos_ids=())
    with pytest.raises(ValueError):
        fn(tiny_model, prompt_ids.repeat(2, 1), 3, eos_ids=())
