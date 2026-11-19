"""Isolate vLLM's attention kernel on the batch shapes of a pure decode step and of a mixed (decodes + prefill chunk) step.

    python gpu/attn_microbench.py --dry-run --model qwen2.5-0.5b --ctx 3584 --decodes 256 --prefill 7600     # Mac: shapes only
    python gpu/attn_microbench.py --model qwen2.5-0.5b --ctx 3584 --decodes 256 --prefill 7600 --out r.json   # GPU host

It calls vLLM's own `flash_attn_varlen_func` (FlashAttention 2 on an A40: FA3 needs Hopper) with a paged KV cache and the same
argument pattern as `vllm/v1/attention/backends/flash_attn.py`, one layer per call, inside NVTX ranges `pure_decode`, `mixed` and
`prefill_only` so Nsight Compute can filter on them (gpu/run_ncu.sh).

Question it answers (performance-report 9.6): does the 4-5x cost of a decode inside a mixed step come from the attention kernel itself?
Per-decode cost of the mixed batch = (mixed - prefill_only) / decodes, against pure_decode / decodes.

Hypothesis, to be checked and not assumed: FlashAttention 2 treats a batch whose longest query is 1 token as a GQA "decode" and
folds the query-head group into the sequence dimension, so each KV head is read once for the whole group. When any sequence in the
batch has more than one query token that shortcut is off, and each query head may re-read the KV. If so, the penalty tracks the
GQA group size: 7 for Qwen2.5-0.5B (14 q / 2 kv heads), 4 for Llama-3.1-8B (32 / 8), 1 for the `-mha` control (14 / 14).
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass

A40_BANDWIDTH_B_S = 0.696e12
FP16_BYTES = 2
KV_PAIR = 2  # keys and values
DEFAULT_BLOCK_SIZE = 16


@dataclass(frozen=True)
class Geometry:
    q_heads: int
    kv_heads: int
    head_dim: int

    @property
    def gqa_group(self) -> int:
        return self.q_heads // self.kv_heads


GEOMETRIES = {
    "qwen2.5-0.5b": Geometry(14, 2, 64),
    "qwen2.5-0.5b-mha": Geometry(14, 14, 64),  # control: same query heads and head size, no GQA
    "llama3.1-8b": Geometry(32, 8, 128),
}


@dataclass(frozen=True)
class BatchSpec:
    label: str
    q_lens: tuple[int, ...]
    kv_lens: tuple[int, ...]


def pure_decode(n: int, ctx: int) -> BatchSpec:
    """n decoding sequences, each with `ctx` cached tokens plus the one being generated."""
    return BatchSpec("pure_decode", (1,) * n, (ctx + 1,) * n)


def mixed(n_decodes: int, ctx: int, prefill_tokens: int, prefill_ctx: int = 0) -> BatchSpec:
    """Decodes first, then one prefill chunk, the order vLLM's reordered batch uses."""
    return BatchSpec("mixed", (1,) * n_decodes + (prefill_tokens,), (ctx + 1,) * n_decodes + (prefill_ctx + prefill_tokens,))


def prefill_only(prefill_tokens: int, prefill_ctx: int = 0) -> BatchSpec:
    return BatchSpec("prefill_only", (prefill_tokens,), (prefill_ctx + prefill_tokens,))


def cu_seqlens(lens: tuple[int, ...]) -> list[int]:
    out = [0]
    for n in lens:
        out.append(out[-1] + n)
    return out


def block_table_shape(kv_lens: tuple[int, ...], block_size: int) -> tuple[int, int]:
    return len(kv_lens), max(-(-n // block_size) for n in kv_lens)


def describe(spec: BatchSpec, geometry: Geometry, block_size: int) -> dict:
    kv_bytes = sum(spec.kv_lens) * KV_PAIR * geometry.kv_heads * geometry.head_dim * FP16_BYTES
    return {"label": spec.label, "sequences": len(spec.q_lens), "q_tokens": sum(spec.q_lens), "max_seqlen_q": max(spec.q_lens),
            "max_seqlen_k": max(spec.kv_lens), "gqa_group": geometry.gqa_group, "block_table_shape": list(block_table_shape(spec.kv_lens, block_size)),
            "kv_bytes_read_per_layer": kv_bytes, "bandwidth_floor_ms": kv_bytes / A40_BANDWIDTH_B_S * 1000}


def _positive(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model", choices=sorted(GEOMETRIES), default="qwen2.5-0.5b")
    p.add_argument("--ctx", type=_positive, default=3584, help="cached tokens per decoding sequence")
    p.add_argument("--decodes", type=_positive, default=256)
    p.add_argument("--prefill", type=_positive, default=7600, help="prefill chunk size inside the mixed batch")
    p.add_argument("--prefill-ctx", type=int, default=0, help="cached tokens already in front of the prefill chunk")
    p.add_argument("--block-size", type=_positive, default=DEFAULT_BLOCK_SIZE)
    p.add_argument("--iters", type=_positive, default=50)
    p.add_argument("--warmup", type=_positive, default=10)
    p.add_argument("--batches", nargs="+", default=["pure_decode", "mixed", "prefill_only"],
                   choices=["pure_decode", "mixed", "prefill_only"], help="for Nsight Compute, run one range with a small --iters")
    p.add_argument("--dry-run", action="store_true", help="print the batch shapes and exit (no torch import)")
    p.add_argument("--out", help="write the result JSON here")
    return p.parse_args(argv)


def build_specs(args: argparse.Namespace) -> list[BatchSpec]:
    by_label = {"pure_decode": pure_decode(args.decodes, args.ctx),
                "mixed": mixed(args.decodes, args.ctx, args.prefill, args.prefill_ctx),
                "prefill_only": prefill_only(args.prefill, args.prefill_ctx)}
    return [by_label[label] for label in args.batches]


def _time_batch(spec: BatchSpec, geometry: Geometry, block_size: int, iters: int, warmup: int) -> dict:
    import torch
    from vllm.vllm_flash_attn import flash_attn_varlen_func

    device = torch.device("cuda")
    n_seq, max_blocks = block_table_shape(spec.kv_lens, block_size)
    n_blocks = sum(-(-n // block_size) for n in spec.kv_lens)
    q = torch.randn(sum(spec.q_lens), geometry.q_heads, geometry.head_dim, dtype=torch.float16, device=device)
    key_cache = torch.randn(n_blocks, block_size, geometry.kv_heads, geometry.head_dim, dtype=torch.float16, device=device)
    value_cache = torch.randn_like(key_cache)
    table = torch.zeros(n_seq, max_blocks, dtype=torch.int32, device=device)
    next_block = 0
    for i, kv_len in enumerate(spec.kv_lens):  # each sequence owns a contiguous run of blocks; the kernel sees real paged addressing
        count = -(-kv_len // block_size)
        table[i, :count] = torch.arange(next_block, next_block + count, dtype=torch.int32, device=device)
        next_block += count
    cu_q = torch.tensor(cu_seqlens(spec.q_lens), dtype=torch.int32, device=device)
    seqused_k = torch.tensor(spec.kv_lens, dtype=torch.int32, device=device)
    out = torch.empty_like(q)

    def call() -> None:
        flash_attn_varlen_func(q=q, k=key_cache, v=value_cache, out=out, cu_seqlens_q=cu_q, max_seqlen_q=max(spec.q_lens),
                               seqused_k=seqused_k, max_seqlen_k=max(spec.kv_lens), softmax_scale=geometry.head_dim ** -0.5,
                               causal=True, block_table=table, fa_version=2)

    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    torch.cuda.nvtx.range_push(spec.label)
    for i in range(iters):
        starts[i].record()
        call()
        ends[i].record()
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()
    times = sorted(s.elapsed_time(e) for s, e in zip(starts, ends))
    return {"median_ms": times[len(times) // 2], "min_ms": times[0], "max_ms": times[-1], "iters": iters}


def _per_decode(results: dict[str, dict], decodes: int) -> dict:
    out = {}
    if "pure_decode" in results:
        out["pure_decode_ms_per_decode"] = results["pure_decode"]["median_ms"] / decodes
    if "mixed" in results and "prefill_only" in results:
        out["mixed_ms_per_decode"] = (results["mixed"]["median_ms"] - results["prefill_only"]["median_ms"]) / decodes
    if len(out) == 2 and out["pure_decode_ms_per_decode"] > 0:
        out["mixed_over_pure"] = out["mixed_ms_per_decode"] / out["pure_decode_ms_per_decode"]
    return out


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    geometry = GEOMETRIES[args.model]
    specs = build_specs(args)
    report = {"model": args.model, "geometry": vars(geometry) | {"gqa_group": geometry.gqa_group}, "ctx": args.ctx, "decodes": args.decodes,
              "prefill": args.prefill, "block_size": args.block_size, "batches": [describe(s, geometry, args.block_size) for s in specs]}
    if not args.dry_run:
        results = {s.label: _time_batch(s, geometry, args.block_size, args.iters, args.warmup) for s in specs}
        for batch in report["batches"]:
            batch.update(results[batch["label"]])
        report["per_decode"] = _per_decode(results, args.decodes)
    text = json.dumps(report, indent=2)
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
