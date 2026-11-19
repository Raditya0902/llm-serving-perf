"""Sweep of the attention microbenchmark over context, decode count and prefill-chunk size (performance-report 9.9, ratio of 18).

    python -m gpu.attn_sweep --dry-run                                  # Mac: the grid only
    python -m gpu.attn_sweep --out sweep.json                            # GPU host, FlashAttention 2
    python -m gpu.attn_sweep --fa-version 3 --ctx 3584 --decodes 256 --out sweep_fa3.json   # Hopper

Pre-registered hypothesis H-dead (PREREGISTRATION_ROUND4.md, Q4): FlashAttention 2 sizes its grid by the longest query in the
batch, so every decode sequence also gets blocks that exit at once; the cost then scales with decodes x longest query and not with
the context length. The decode share of a mixed batch is d(P) = mixed(P) - prefill_only(P); the excess is d(P) - d(REFERENCE_P).
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass

import numpy as np

from gpu import attn_microbench as ab

REFERENCE_P = 1024  # on the flat part of the curve (P = 2 ... 1,024 costs the same, 9.9 section 3)
MIN_EXCESS_P = 2048
SMALL_P = 512
CONTEXT_WITHIN, CONTEXT_REFUTED_LOW, CONTEXT_REFUTED_HIGH = 0.25, 0.5, 2.0
R2_SUPPORTED, R2_REFUTED, SMALL_SHARE_LIMIT = 0.9, 0.7, 0.05
DEFAULT_CTXS, DEFAULT_DECODES = (512, 3584), (64, 256, 512)
DEFAULT_PREFILLS = (2, 128, 512, 1024, 2048, 4096, 7600)


@dataclass(frozen=True)
class SweepPoint:
    ctx: int
    decodes: int
    prefill: int


def grid(ctxs, decodes_list, prefills) -> list[SweepPoint]:
    return [SweepPoint(c, n, p) for c in ctxs for n in decodes_list for p in prefills]


def decode_share(row: dict) -> float:
    return row["mixed_ms"] - row["prefill_only_ms"]


def ratio(row: dict) -> float:
    return decode_share(row) / row["pure_ms"]


def _index(rows: list[dict]) -> dict[tuple[int, int, int], dict]:
    return {(r["ctx"], r["decodes"], r["prefill"]): r for r in rows}


def excess_rows(rows: list[dict], reference_p: int = REFERENCE_P) -> list[tuple[dict, float]]:
    """(row, d(P) - d(reference_p)) for every row whose P is above the reference and that has a reference row."""
    by_key = _index(rows)
    out = []
    for row in rows:
        ref = by_key.get((row["ctx"], row["decodes"], reference_p))
        if row["prefill"] >= reference_p and ref is not None:
            out.append((row, decode_share(row) - decode_share(ref)))
    return out


def fit_excess(rows: list[dict]) -> dict:
    """excess = k x decodes x (P - reference): least squares through the origin on the pooled points with P >= MIN_EXCESS_P."""
    pairs = [(r, e) for r, e in excess_rows(rows) if r["prefill"] >= MIN_EXCESS_P]
    if len(pairs) < 3:
        return {"k_ms_per_decode_token": None, "r2": None, "points": len(pairs)}
    x = np.array([r["decodes"] * (r["prefill"] - REFERENCE_P) for r, _ in pairs], dtype=float)
    y = np.array([e for _, e in pairs], dtype=float)
    k = float((x * y).sum() / (x * x).sum())
    ss_res = float(((y - k * x) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 if ss_tot == 0 else 1 - ss_res / ss_tot
    return {"k_ms_per_decode_token": k, "r2": r2, "points": len(pairs)}


def context_consistency(rows: list[dict]) -> dict:
    """Median of excess(ctx 512) / excess(ctx 3584) over the matching (decodes, P) points with P >= MIN_EXCESS_P."""
    ctxs = sorted({r["ctx"] for r in rows})
    if len(ctxs) < 2:
        return {"median_ratio": None, "within_25pct": False, "refuted": False, "points": 0}
    low, high = ctxs[0], ctxs[-1]
    excess = {(r["ctx"], r["decodes"], r["prefill"]): e for r, e in excess_rows(rows) if r["prefill"] >= MIN_EXCESS_P}
    ratios = [excess[(low, n, p)] / excess[(high, n, p)] for (c, n, p) in excess if c == low and (high, n, p) in excess and excess[(high, n, p)] > 0]
    if not ratios:
        return {"median_ratio": None, "within_25pct": False, "refuted": False, "points": 0}
    median = float(np.median(ratios))
    return {"median_ratio": median, "low_ctx": low, "high_ctx": high, "points": len(ratios),
            "within_25pct": 1 - CONTEXT_WITHIN <= median <= 1 + CONTEXT_WITHIN,
            "refuted": median < CONTEXT_REFUTED_LOW or median > CONTEXT_REFUTED_HIGH}


def dead_share_small_p(rows: list[dict], fit: dict, max_p: int = SMALL_P) -> dict[tuple[int, int, int], float]:
    """Fitted dead-block cost k x decodes x P as a fraction of the mixed time, for the small-P rows (the server's regime)."""
    k = fit.get("k_ms_per_decode_token")
    if k is None:
        return {}
    return {(r["ctx"], r["decodes"], r["prefill"]): k * r["decodes"] * r["prefill"] / r["mixed_ms"]
            for r in rows if max_p >= r["prefill"] > 1 and r["mixed_ms"] > 0}


def analyse(rows: list[dict]) -> dict:
    fit = fit_excess(rows)
    context = context_consistency(rows)
    small = dead_share_small_p(rows, fit)
    r2 = fit["r2"]
    supported = bool(r2 is not None and r2 >= R2_SUPPORTED and context["within_25pct"] and small and max(small.values()) < SMALL_SHARE_LIMIT)
    refuted = bool(context["refuted"] or (r2 is not None and r2 < R2_REFUTED))
    return {"fit": fit, "context": context,
            "small_p": {"max_share": max(small.values()) if small else None, "points": len(small)},
            "verdict": {"supported": supported, "refuted": refuted}}


FLOOR_MIN_P = 2048
FLOOR_MEDIAN_ERROR, FLOOR_MAX_ERROR = 0.15, 0.30


def fit_floor_model(rows: list[dict], ctx: int) -> dict:
    """Q4b: fit d(P) / n = a + k x P on the rows of one context where the dead-block term clearly dominates (P >= FLOOR_MIN_P)."""
    pts = [(r["prefill"], decode_share(r) / r["decodes"]) for r in rows if r["ctx"] == ctx and r["prefill"] >= FLOOR_MIN_P]
    if len(pts) < 3:
        return {"a_ms_per_decode": None, "k_ms_per_decode_token": None, "points": len(pts)}
    k, a = np.polyfit([p for p, _ in pts], [y for _, y in pts], 1)
    return {"a_ms_per_decode": float(a), "k_ms_per_decode_token": float(k), "points": len(pts)}


def predict_floor(row: dict, model: dict, d_live: float) -> float:
    return max(d_live, row["decodes"] * (model["a_ms_per_decode"] + model["k_ms_per_decode_token"] * row["prefill"]))


def floor_model_errors(rows: list[dict], model: dict, ctxs: list[int], min_p: int = 128) -> dict:
    """Relative error of the Q4b prediction on `ctxs` (not the fitted context); d_live is each (ctx, n)'s own d(P = smallest P)."""
    by_cn: dict[tuple[int, int], list[dict]] = {}
    for r in rows:
        by_cn.setdefault((r["ctx"], r["decodes"]), []).append(r)
    errors = []
    for (ctx, _n), group in by_cn.items():
        if ctx not in ctxs:
            continue
        live = decode_share(min(group, key=lambda r: r["prefill"]))
        errors += [abs(predict_floor(r, model, live) - decode_share(r)) / decode_share(r) for r in group if r["prefill"] >= min_p]
    if not errors:
        return {"median": None, "max": None, "points": 0, "supported": False}
    median, worst = float(np.median(errors)), float(max(errors))
    return {"median": median, "max": worst, "points": len(errors), "supported": median <= FLOOR_MEDIAN_ERROR and worst <= FLOOR_MAX_ERROR}


def excess_by_context(rows: list[dict], decodes: int, prefill: int) -> dict[int, float]:
    """d(P) - d(P = 1,024) at fixed decodes and P for every context in the rows: Q4b says it shrinks as the context grows."""
    return {r["ctx"]: e for r, e in excess_rows(rows) if r["decodes"] == decodes and r["prefill"] == prefill}


def _positive(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model", choices=sorted(ab.GEOMETRIES), default="qwen2.5-0.5b")
    p.add_argument("--ctx", type=_positive, nargs="+", default=list(DEFAULT_CTXS))
    p.add_argument("--decodes", type=_positive, nargs="+", default=list(DEFAULT_DECODES))
    p.add_argument("--prefill", type=_positive, nargs="+", default=list(DEFAULT_PREFILLS))
    p.add_argument("--fa-version", type=int, choices=[2, 3], default=2)
    p.add_argument("--block-size", type=_positive, default=ab.DEFAULT_BLOCK_SIZE)
    p.add_argument("--iters", type=_positive, default=30)
    p.add_argument("--warmup", type=_positive, default=5)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--out")
    return p.parse_args(argv)


def run_sweep(points: list[SweepPoint], args: argparse.Namespace) -> list[dict]:
    geometry = ab.GEOMETRIES[args.model]
    cache: dict[tuple, float] = {}

    def timed(spec: ab.BatchSpec) -> float:
        key = (spec.label, spec.q_lens, spec.kv_lens)
        if key not in cache:
            cache[key] = ab._time_batch(spec, geometry, args.block_size, args.iters, args.warmup, args.fa_version)["median_ms"]
        return cache[key]

    return [{"ctx": pt.ctx, "decodes": pt.decodes, "prefill": pt.prefill,
             "pure_ms": timed(ab.pure_decode(pt.decodes, pt.ctx)),
             "prefill_only_ms": timed(ab.prefill_only(pt.prefill)),
             "mixed_ms": timed(ab.mixed(pt.decodes, pt.ctx, pt.prefill))} for pt in points]


def floor_report(rows: list[dict]) -> dict:
    ctxs = sorted({r["ctx"] for r in rows})
    if not ctxs:
        return {}
    model = fit_floor_model(rows, ctxs[0])
    if model["a_ms_per_decode"] is None:
        return {"fit": model}
    return {"fit_ctx": ctxs[0], "fit": model, "out_of_sample": floor_model_errors(rows, model, ctxs[1:]),
            "in_sample": floor_model_errors(rows, model, ctxs[:1])}


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    points = grid(args.ctx, args.decodes, args.prefill)
    report = {"model": args.model, "fa_version": args.fa_version, "points": len(points), "block_size": args.block_size}
    if not args.dry_run:
        rows = run_sweep(points, args)
        report |= {"rows": rows, "analysis": analyse(rows), "floor_model": floor_report(rows), "ratios": {f"{r['ctx']}/{r['decodes']}/{r['prefill']}": ratio(r) for r in rows}}
    text = json.dumps(report, indent=2)
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
