import pytest
import torch

from variants import (
    DecodeRun,
    make_batch_ids,
    run_batched,
    run_day1,
    run_deferred_sync,
)

EOS = (1, 2)
NEW = 6


def test_deferred_sync_generates_the_same_tokens_as_day1(tiny_model, prompt_ids):
    baseline = run_day1(tiny_model, prompt_ids, NEW, EOS)
    deferred = run_deferred_sync(tiny_model, prompt_ids, NEW, EOS)

    assert deferred.tokens == baseline.tokens
    assert len(deferred.tokens[0]) == NEW
    assert deferred.decode_steps == NEW - 1


def test_batched_with_one_sequence_matches_day1(tiny_model, prompt_ids):
    baseline = run_day1(tiny_model, prompt_ids, NEW, EOS)
    batched = run_batched(tiny_model, prompt_ids, NEW, EOS)

    assert batched.tokens == baseline.tokens
    assert batched.batch_size == 1


def test_batched_rows_match_their_single_request_runs(tiny_model, prompt_ids):
    ids = make_batch_ids(prompt_ids, 3)
    batched = run_batched(tiny_model, ids, NEW, EOS)

    for row in range(3):
        single = run_day1(tiny_model, ids[row : row + 1], NEW, EOS)
        assert batched.tokens[row] == single.tokens[0]


def test_make_batch_ids_gives_distinct_rows_of_equal_length(prompt_ids):
    ids = make_batch_ids(prompt_ids, 4)

    assert ids.shape == (4, prompt_ids.shape[1])
    assert len({tuple(row.tolist()) for row in ids}) == 4
    assert torch.equal(ids[0], prompt_ids[0])


def test_make_batch_ids_rejects_non_positive_batch(prompt_ids):
    with pytest.raises(ValueError, match="batch"):
        make_batch_ids(prompt_ids, 0)


def test_decode_run_derived_timings():
    run = DecodeRun(
        tokens=((5, 6, 7),),
        prefill_s=0.1,
        decode_s=0.2,
        decode_steps=2,
        step_times_s=(0.1, 0.1),
        peak_mem_bytes=1,
    )

    assert run.total_s == pytest.approx(0.3)
    assert run.tpot_s == pytest.approx(0.1)
    assert run.batch_size == 1
    assert run.output_tokens == 3


def test_tpot_is_zero_when_there_are_no_decode_steps():
    run = DecodeRun(((5,),), 0.1, 0.0, 0, (), 1)
    assert run.tpot_s == 0.0
