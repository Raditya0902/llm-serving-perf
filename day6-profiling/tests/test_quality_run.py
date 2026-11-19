"""Prompt planning and record conversion of the backend quality run (the model runs need a GPU)."""
from __future__ import annotations

from types import SimpleNamespace as NS

import pytest

from gpu import quality_run as qr


def test_plan_prompts_is_deterministic_and_stays_inside_the_corpus_and_length_range():
    plan = qr.plan_prompts(n=50, seed=0, min_len=64, max_len=3000, corpus_len=20000)
    assert plan == qr.plan_prompts(n=50, seed=0, min_len=64, max_len=3000, corpus_len=20000)
    assert len(plan) == 50
    assert all(64 <= length <= 3000 and 0 <= start and start + length <= 20000 for start, length in plan)
    assert plan != qr.plan_prompts(n=50, seed=1, min_len=64, max_len=3000, corpus_len=20000)


def test_plan_prompts_mixes_short_and_long_prompts():
    lengths = [length for _, length in qr.plan_prompts(n=200, seed=0, min_len=64, max_len=3000, corpus_len=50000)]
    assert min(lengths) < 200 and max(lengths) > 1500


def test_plan_prompts_rejects_a_corpus_shorter_than_the_longest_prompt():
    with pytest.raises(ValueError, match="corpus"):
        qr.plan_prompts(n=3, seed=0, min_len=64, max_len=3000, corpus_len=1000)


def _lp(value):
    return NS(logprob=value)


def test_record_keeps_the_logprob_of_each_prompt_token_and_each_chosen_token():
    out = NS(prompt_token_ids=[10, 11, 12], prompt_logprobs=[None, {11: _lp(-1.0), 99: _lp(-0.5)}, {12: _lp(-2.0)}],
             outputs=[NS(token_ids=[5, 6], logprobs=[{5: _lp(-0.1), 7: _lp(-0.9)}, {6: _lp(-0.2)}])])
    rec = qr.record_from_output(3, out)
    assert rec["id"] == 3 and rec["n_prompt"] == 3
    assert rec["prompt_logprobs"] == [None, -1.0, -2.0]
    assert rec["tokens"] == [5, 6] and rec["logprobs"] == [-0.1, -0.2]
    assert rec["top"][0] == {"5": -0.1, "7": -0.9}


def test_parse_args_requires_exactly_one_of_backend_or_reference():
    with pytest.raises(SystemExit):
        qr.parse_args(["--out", "x.json"])
    assert qr.parse_args(["--backend", "TRITON_ATTN", "--out", "x.json"]).bt == 2048
    assert qr.parse_args(["--reference", "--out", "r.json"]).reference is True


def test_corpus_is_long_enough_for_the_longest_prompt_and_skips_virtualenvs():
    text = qr.corpus_text()
    assert len(text) > 5 * qr.MAX_LEN * 3  # several times the longest prompt in characters
    assert "site-packages" not in qr.CORPUS_ROOT.as_posix()
