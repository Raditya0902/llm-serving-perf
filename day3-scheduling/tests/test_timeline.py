from __future__ import annotations

from helpers import cfg, req, run
from timeline import gantt_segments, render_ascii


def result():
    reqs = [req(0, 0.0, p=64, o=10), req(1, 5.0, p=64, o=10), req(2, 6.0, p=64, o=5)]
    return run(cfg(max_running_seqs=2), reqs)


def test_segments_cover_wait_prefill_decode_in_order() -> None:
    segs = gantt_segments(result())
    kinds = [k for _, _, k in segs[2] if k in "WPD"]
    assert kinds[0] == "W"
    assert "P" in kinds and "D" in kinds
    assert kinds.index("P") < kinds.index("D")
    assert all(end >= start for start, end, _ in segs[0])


def test_request_that_never_waits_has_no_wait_segment_of_positive_length() -> None:
    segs = gantt_segments(result())
    assert all(end - start <= 1e-9 for start, end, k in segs[0] if k == "W")


def test_ascii_has_one_row_per_request_with_legend() -> None:
    text = render_ascii(result(), width=40)
    lines = text.splitlines()
    assert sum(1 for l in lines if l.lstrip().startswith("req")) == 3
    assert "P=prefill" in text and "D=decode" in text
    row = next(l for l in lines if l.lstrip().startswith("req 2"))
    assert "." in row and "P" in row and "D" in row


def test_ascii_marks_preemption() -> None:
    reqs = [req(i, 0.0, p=64, o=200) for i in range(6)]
    res = run(cfg(num_kv_blocks=40), reqs)
    assert res.preemptions > 0
    assert "X" in render_ascii(res, width=80)
