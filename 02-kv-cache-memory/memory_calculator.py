"""Theoretical GPU memory calculator for an LLM deployment.

Everything here is arithmetic on the model config: no GPU, no torch. The numbers are
*theoretical data-storage requirements*. Real deployments add activations, the CUDA
context, allocator fragmentation and framework buffers on top (see README).

KV formula (same as day 1):
    bytes/token = 2 (K and V) x layers x kv_heads x head_dim x dtype_bytes
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass

BYTES_PER_GIB = 1024**3
DEFAULT_GROWTH_CONTEXTS = (512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072)
KV_TENSORS_PER_LAYER = 2  # one K and one V
DEFAULT_WEIGHT_DTYPE = "fp16"

DTYPE_BYTES: dict[str, float] = {
    "fp32": 4,
    "fp16": 2,
    "bf16": 2,
    "fp8": 1,
    "int8": 1,
    "int4": 0.5,
}


def _check_dtype(name: str) -> None:
    if name not in DTYPE_BYTES:
        raise ValueError(f"unknown dtype {name!r}; choose from {sorted(DTYPE_BYTES)}")


def _check_positive(**values: float) -> None:
    for name, value in values.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")


@dataclass(frozen=True)
class ModelSpec:
    """Static properties of a model that determine its memory footprint."""

    params: float
    weight_dtype: str
    num_layers: int
    num_kv_heads: int
    head_dim: int

    def __post_init__(self) -> None:
        _check_positive(
            params=self.params,
            num_layers=self.num_layers,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
        )
        _check_dtype(self.weight_dtype)


@dataclass(frozen=True)
class DeploymentSpec:
    """A model plus the serving workload: context length and concurrency."""

    model: ModelSpec
    context_len: int
    concurrent_seqs: int
    kv_dtype: str = "fp16"

    def __post_init__(self) -> None:
        _check_positive(context_len=self.context_len, concurrent_seqs=self.concurrent_seqs)
        _check_dtype(self.kv_dtype)


PRESETS: dict[str, ModelSpec] = {
    # `01-inference-loop` model: 24 layers, 2 KV heads (14 query heads), head_dim 64.
    "qwen2.5-0.5b": ModelSpec(0.494e9, "fp16", 24, 2, 64),
    "llama3-8b": ModelSpec(8.03e9, "fp16", 32, 8, 128),
    "llama3-70b": ModelSpec(70.6e9, "fp16", 80, 8, 128),
}
PRESET_QUERY_HEADS = {"qwen2.5-0.5b": 14, "llama3-8b": 32, "llama3-70b": 64}


def weight_bytes(spec: DeploymentSpec) -> int:
    """Memory for model parameters. Independent of context and concurrency."""
    return round(spec.model.params * DTYPE_BYTES[spec.model.weight_dtype])


def kv_bytes_per_token(spec: DeploymentSpec) -> int:
    """KV-cache bytes added by one token of one sequence (across all layers)."""
    m = spec.model
    elems = KV_TENSORS_PER_LAYER * m.num_layers * m.num_kv_heads * m.head_dim
    return round(elems * DTYPE_BYTES[spec.kv_dtype])


def kv_cache_bytes(spec: DeploymentSpec) -> int:
    """KV cache if every sequence reaches the full context length."""
    return kv_bytes_per_token(spec) * spec.context_len * spec.concurrent_seqs


def total_bytes(spec: DeploymentSpec) -> int:
    """Theoretical total: weights + KV cache (no activations/overhead)."""
    return weight_bytes(spec) + kv_cache_bytes(spec)


def memory_growth(spec: DeploymentSpec, context_lens: list[int] | tuple[int, ...]) -> list[dict]:
    """KV and total memory as the context length grows, other inputs held fixed."""
    rows = []
    for ctx in context_lens:
        point = DeploymentSpec(spec.model, ctx, spec.concurrent_seqs, spec.kv_dtype)
        rows.append(
            {
                "context_len": ctx,
                "kv_bytes": kv_cache_bytes(point),
                "total_bytes": total_bytes(point),
            }
        )
    return rows


def max_concurrent_seqs(spec: DeploymentSpec, gpu_bytes: float, overhead_frac: float = 0.1) -> int:
    """How many full-context sequences fit once weights and overhead are paid for.

    Returns 0 when the weights alone do not fit. When this number is exceeded a server
    must queue, reject, or preempt requests (see README).
    """
    if not 0 <= overhead_frac < 1:
        raise ValueError(f"overhead_frac must be in [0, 1), got {overhead_frac}")
    budget = gpu_bytes * (1 - overhead_frac) - weight_bytes(spec)
    if budget <= 0:
        return 0
    one_seq = kv_cache_bytes(DeploymentSpec(spec.model, spec.context_len, 1, spec.kv_dtype))
    return int(budget // one_seq)


def gqa_savings(spec: DeploymentSpec, num_query_heads: int) -> float:
    """KV-cache reduction factor of GQA vs full multi-head attention."""
    _check_positive(num_query_heads=num_query_heads)
    return num_query_heads / spec.model.num_kv_heads


def _gib(n: float) -> str:
    return f"{n / BYTES_PER_GIB:,.2f} GiB"


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preset", choices=sorted(PRESETS), help="use a built-in model config")
    p.add_argument("--params", type=float, help="parameter count, e.g. 8e9")
    p.add_argument("--weight-dtype", default=None, choices=sorted(DTYPE_BYTES),
                   help=f"weight precision (default: preset's, else {DEFAULT_WEIGHT_DTYPE})")
    p.add_argument("--layers", type=int, help="number of transformer layers")
    p.add_argument("--kv-heads", type=int, help="number of KV heads")
    p.add_argument("--head-dim", type=int, help="dimension per head")
    p.add_argument("--context", type=int, default=4096, help="context length (tokens)")
    p.add_argument("--seqs", type=int, default=1, help="concurrent sequences")
    p.add_argument("--kv-dtype", default="fp16", choices=sorted(DTYPE_BYTES))
    p.add_argument("--gpu-gb", type=float, help="GPU memory (GiB) for a headroom check")
    p.add_argument("--overhead", type=float, default=0.1, help="reserved fraction for activations/context")
    return p


def _model_from_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> ModelSpec:
    if args.preset:
        base = PRESETS[args.preset]
        # Explicit flags override the preset, including falsy ones (validation rejects 0).
        return ModelSpec(
            _first_set(args.params, base.params),
            _first_set(args.weight_dtype, base.weight_dtype),
            _first_set(args.layers, base.num_layers),
            _first_set(args.kv_heads, base.num_kv_heads),
            _first_set(args.head_dim, base.head_dim),
        )
    missing = [f for f in ("params", "layers", "kv_heads", "head_dim") if getattr(args, f) is None]
    if missing:
        parser.error(f"provide --preset or all of --params --layers --kv-heads --head-dim (missing: {missing})")
    return ModelSpec(
        args.params, _first_set(args.weight_dtype, DEFAULT_WEIGHT_DTYPE), args.layers, args.kv_heads, args.head_dim
    )


def _first_set(value, default):
    """`value` unless it was not given (None); unlike `or`, keeps 0 so it gets validated."""
    return default if value is None else value


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        model = _model_from_args(args, parser)
        spec = DeploymentSpec(model, args.context, args.seqs, args.kv_dtype)
    except ValueError as exc:
        parser.error(str(exc))

    print(f"Model: {model.params / 1e9:.2f}B params ({model.weight_dtype}), {model.num_layers} layers, "
          f"{model.num_kv_heads} KV heads x {model.head_dim} dim")
    print(f"Workload: context {spec.context_len:,} x {spec.concurrent_seqs} seqs, KV {spec.kv_dtype}")
    print(f"KV per token: {kv_bytes_per_token(spec):,} B")
    print(f"Weights  : {_gib(weight_bytes(spec))}")
    print(f"KV cache : {_gib(kv_cache_bytes(spec))}")
    print(f"Total    : {_gib(total_bytes(spec))}  (theoretical, excludes activations/overhead)")
    if args.gpu_gb is not None:
        gpu = args.gpu_gb * BYTES_PER_GIB
        n = max_concurrent_seqs(spec, gpu, args.overhead)
        verdict = "fits" if total_bytes(spec) <= gpu * (1 - args.overhead) else "DOES NOT FIT"
        print(f"On {args.gpu_gb:g} GiB GPU ({args.overhead:.0%} overhead): {verdict}; "
              f"max concurrent seqs at this context = {n}")

    print("\nKV growth with context length (concurrency fixed):")
    print(f"{'context':>10} {'KV cache':>14} {'total':>14}")
    for row in memory_growth(spec, DEFAULT_GROWTH_CONTEXTS):
        print(f"{row['context_len']:>10,} {_gib(row['kv_bytes']):>14} {_gib(row['total_bytes']):>14}")


if __name__ == "__main__":
    main()
