"""Timeline of requests entering and leaving the active batch.

`gantt_segments` returns, per request, (start_ms, end_ms, kind) segments:
    W waiting in the queue      R resident in the batch (underlay)
    P prefill iteration         D decode iteration
`render_ascii` draws one text row per request; a column is a fixed time bucket and
shows P > D > '-' (resident but not scheduled) > '.' (waiting). `X` marks a bucket
in which the request was preempted (KV freed, back to the queue). Blank = not yet
arrived or already finished.
"""
from __future__ import annotations

from request import Request
from scheduler import SchedulerConfig
from metrics import EPS_MS
from simulator import SimResult

LEGEND = "P=prefill  D=decode  -=in batch, idle  .=waiting  X=preempted  (blank = not arrived / finished)"
_PRIORITY = {"P": 3, "D": 2, "R": 1, "W": 0}


def gantt_segments(result: SimResult) -> dict[int, list[tuple[float, float, str]]]:
    segs: dict[int, list[tuple[float, float, str]]] = {r.request_id: [] for r in result.requests}
    open_since: dict[int, tuple[float, str]] = {}
    for e in result.events:
        rid = e.request_id
        if e.kind in ("ARRIVE", "PREEMPT"):
            if e.kind == "PREEMPT" and rid in open_since:
                start, kind = open_since.pop(rid)
                segs[rid].append((start, e.time_ms, kind))
            open_since[rid] = (e.time_ms, "W")
        elif e.kind == "ADMIT":
            start, kind = open_since.pop(rid)
            segs[rid].append((start, e.time_ms, kind))
            open_since[rid] = (e.time_ms, "R")
        elif e.kind == "FINISH":
            start, kind = open_since.pop(rid)
            segs[rid].append((start, e.time_ms, kind))
    for it in result.iterations:
        end = it.start_ms + it.duration_ms
        for rid, _ in it.prefill_ids:
            segs[rid].append((it.start_ms, end, "P"))
        for rid in it.decode_ids:
            segs[rid].append((it.start_ms, end, "D"))
    return segs


def _bucket_char(segs: list[tuple[float, float, str]], lo: float, hi: float) -> str:
    best = None
    for start, end, kind in segs:
        if end > lo and start < hi and (best is None or _PRIORITY[kind] > _PRIORITY[best]):
            best = kind
    return {"P": "P", "D": "D", "R": "-", "W": ".", None: " "}[best]


def render_ascii(result: SimResult, width: int = 80, request_ids: list[int] | None = None) -> str:
    segs = gantt_segments(result)
    t0, t1 = result.first_arrival, result.end_time
    step = max((t1 - t0) / width, EPS_MS)
    preempt_times: dict[int, list[float]] = {}
    for e in result.events:
        if e.kind == "PREEMPT":
            preempt_times.setdefault(e.request_id, []).append(e.time_ms)
    by_id = {r.request_id: r for r in result.requests}
    ids = request_ids if request_ids is not None else sorted(segs)
    lines = [f"time: {t0:.0f} ms -> {t1:.0f} ms, 1 column = {step:.1f} ms", LEGEND]
    for rid in ids:
        cells = []
        for c in range(width):
            lo, hi = t0 + c * step, t0 + (c + 1) * step
            if any(lo <= t < hi or (c == width - 1 and t == hi) for t in preempt_times.get(rid, [])):
                cells.append("X")
            else:
                cells.append(_bucket_char(segs[rid], lo, hi))
        r = by_id[rid]
        lines.append(f"req {rid:<3} p{r.priority} P={r.prompt_tokens:<5}O={r.max_output_tokens:<4}|{''.join(cells)}|")
    return "\n".join(lines)


def demo_scenario() -> tuple[list[Request], SchedulerConfig]:
    """Small 8-request scenario (4 batch slots, tight KV pool) used for the design doc timeline.

    It exercises queueing behind a full batch, a long prompt arriving while others decode
    (chunked into several iterations), a high-priority request, and a KV-exhaustion preemption.
    """
    spec = [  # (arrival_ms, prompt, output, priority)
        (0.0, 96, 40, 1), (5.0, 64, 24, 1), (10.0, 1024, 16, 1), (15.0, 48, 60, 1),
        (60.0, 200, 20, 0), (90.0, 32, 12, 1), (95.0, 512, 30, 1), (100.0, 64, 16, 1),
    ]
    requests = [Request(i, a, p, o, prio) for i, (a, p, o, prio) in enumerate(spec)]
    config = SchedulerConfig(
        max_running_seqs=4, token_budget=256, chunked_prefill=True, num_kv_blocks=75, block_size=16
    )
    return requests, config
