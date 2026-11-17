"""KV-pool sizing and a hardware-derived cost model (Day 2 byte counts -> Day 3 simulator).

Everything is arithmetic; the hardware numbers are ILLUSTRATIVE published peaks plus an
assumed utilisation, not measurements. They exist so the simulated matrix is internally
consistent: the same bytes that size the KV pool also set the decode step time.

    kv_bytes/token  = 2 (K,V) x layers x kv_heads x head_dim x dtype_bytes     (Day 2)
    pool_bytes      = gpu_bytes x util - weight_bytes - activation_reserve
    pool_blocks     = floor(pool_bytes / (block_size x kv_bytes/token))
    weight_read_ms  = weight_bytes / mem_bw            (decode is bandwidth-bound)
    compute_ms/tok  = 2 x params / (peak_flops x mfu)  (prefill is compute-bound)
    kv_read_ms/tok  = kv_bytes/token / mem_bw
"""
from __future__ import annotations

from dataclasses import dataclass

import _prior  # noqa: F401  (puts Days 2-5 on sys.path)

# isort: split
from cost_model import CostModel
from memory_calculator import (
    PRESETS,
    DeploymentSpec,
    ModelSpec,
    kv_bytes_per_token,
    weight_bytes,
)

GIB = 1024**3
BLOCK_SIZE = 16
ACTIVATION_RESERVE_GIB = 2.0  # activations + CUDA context + graphs, assumed
MS_PER_S = 1000.0


@dataclass(frozen=True)
class HardwareProfile:
    """One GPU. ILLUSTRATIVE: vendor peak numbers and an assumed compute efficiency."""

    name: str
    gpu_bytes: int
    mem_bw_bytes_s: float
    peak_flops: float  # dense bf16/fp16
    assumed_mfu: float  # fraction of peak that prefill actually reaches
    note: str = "illustrative, not measured"

    def __post_init__(self) -> None:
        if min(self.gpu_bytes, self.mem_bw_bytes_s, self.peak_flops) <= 0:
            raise ValueError("hardware numbers must be positive")
        if not 0 < self.assumed_mfu <= 1:
            raise ValueError("assumed_mfu must be in (0, 1]")


HARDWARE: dict[str, HardwareProfile] = {
    "a100-40gb": HardwareProfile("a100-40gb", 40 * GIB, 1.555e12, 312e12, 0.5),
    "a10g-24gb": HardwareProfile("a10g-24gb", 24 * GIB, 0.600e12, 125e12, 0.5),
}

MODELS: dict[str, ModelSpec] = {
    **PRESETS,
    # Qwen2.5-7B-Instruct: 28 layers, 4 KV heads (28 query heads), head_dim 128. Ungated.
    "qwen2.5-7b": ModelSpec(7.62e9, "fp16", 28, 4, 128),
}


def model_spec(name: str) -> ModelSpec:
    try:
        return MODELS[name]
    except KeyError:
        raise ValueError(f"unknown model {name!r}; valid: {sorted(MODELS)}") from None


def deployment_for(model: ModelSpec, kv_dtype: str = "fp16") -> DeploymentSpec:
    return DeploymentSpec(model, context_len=1, concurrent_seqs=1, kv_dtype=kv_dtype)


def kv_bytes_per_token_for(model: ModelSpec, kv_dtype: str = "fp16") -> int:
    return kv_bytes_per_token(deployment_for(model, kv_dtype))


def kv_pool_blocks(
    model: ModelSpec,
    hardware: HardwareProfile,
    gpu_memory_utilization: float,
    block_size: int = BLOCK_SIZE,
    activation_reserve_gib: float = ACTIVATION_RESERVE_GIB,
) -> int:
    """Number of KV blocks vLLM would have after weights and activations. 0 if none fit."""
    if not 0.0 < gpu_memory_utilization <= 1.0:
        raise ValueError("gpu_memory_utilization must be in (0, 1]")
    pool_bytes = (
        hardware.gpu_bytes * gpu_memory_utilization
        - weight_bytes(deployment_for(model))
        - activation_reserve_gib * GIB
    )
    if pool_bytes <= 0:
        return 0
    return int(pool_bytes // (block_size * kv_bytes_per_token_for(model)))


def cost_model_for(model: ModelSpec, hardware: HardwareProfile) -> CostModel:
    """Day 3 `CostModel` whose constants come from bytes and bandwidth, not guesses."""
    return CostModel(
        weight_read_ms=weight_bytes(deployment_for(model)) / hardware.mem_bw_bytes_s * MS_PER_S,
        kv_read_ms_per_token=kv_bytes_per_token_for(model) / hardware.mem_bw_bytes_s * MS_PER_S,
        compute_ms_per_token=2 * model.params
        / (hardware.peak_flops * hardware.assumed_mfu)
        * MS_PER_S,
    )


def capacity_pressure(
    model: ModelSpec,
    hardware: HardwareProfile,
    gpu_memory_utilization: float,
    *,
    concurrency: int,
    mean_input: int,
    mean_output: int,
    block_size: int = BLOCK_SIZE,
) -> dict[str, float | int | bool]:
    """Analytic check: do `concurrency` mean-sized sequences fit in the KV pool?

    Demand uses the time-averaged footprint of a sequence (prompt plus half its output),
    which is what a steady closed loop holds, not the peak. `pressured` means the
    average demand alone exceeds the pool, so preemption is expected.
    """
    pool_tokens = kv_pool_blocks(model, hardware, gpu_memory_utilization, block_size) * block_size
    demand = concurrency * (mean_input + mean_output / 2)
    ratio = demand / pool_tokens if pool_tokens else float("inf")
    return {
        "pool_tokens": pool_tokens,
        "demand_tokens": demand,
        "demand_ratio": ratio,
        "pressured": ratio > 1.0,
    }
