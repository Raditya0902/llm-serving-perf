"""Analytic two-GPU / multi-node model: what tensor parallelism buys and costs for decode.

ASSUMED, not measured. Link constants are the Day 5 "assumption" rows (NVLink-class, PCIe-class)
plus a 100 GbE network entry; GPU peaks come from ``kv_sizing.HARDWARE``. Day 5 measured only
Gloo/CPU AllReduce latency, so nothing here has been validated on NVLink/NCCL.

Per decode step with `T` tokens in the batch (T = concurrent sequences), tensor parallel degree `n`:

    compute  = max(2 * params * T / F, weight_bytes / BW) / n        # decode: the weight read wins
    comm     = L_ar * alpha  +  beta(n) * L_ar * T * hidden * 2 B    # L_ar = 2 AllReduces / layer
    beta(n)  = (2 (n-1) / n) / busbw                                  # ring AllReduce, nccl-tests
    t_tp     = compute + comm

TP also changes capacity, which is the bigger effect for long-context serving: weights are
sharded, so each GPU frees ``weight_bytes (1 - 1/n)`` for KV, and the KV heads are sharded too
(when ``num_kv_heads % n == 0``). Group KV capacity therefore grows *faster than n*.

The Day 5 roofline (`bench_model.predict_tp_speedup`) models one AllReduce; here the per-call
latency is multiplied by the number of AllReduces in a step, which is exact for that model.
"""
from __future__ import annotations

from dataclasses import dataclass

import _prior  # noqa: F401  (puts Days 2-5 on sys.path)

# isort: split
from bench_model import bus_bandwidth_factor, predict_tp_speedup
from memory_calculator import weight_bytes

import kv_sizing

MS_PER_S = 1000.0
ALLREDUCES_PER_LAYER = 2  # after attention's output projection and after the MLP's down projection
ACTIVATION_BYTES = 2  # fp16 activations on the wire

HIDDEN: dict[str, int] = {
    "qwen2.5-0.5b": 896, "qwen2.5-7b": 3584, "llama3-8b": 4096, "llama3-70b": 8192,
}


@dataclass(frozen=True)
class Link:
    name: str
    alpha_s: float  # per-AllReduce latency
    busbw_bytes_s: float  # nccl-tests "bus bandwidth"
    note: str = "ASSUMED"


LINKS: dict[str, Link] = {
    "nvlink": Link("nvlink", 15e-6, 200e9, "Day 5 nvlink_class assumption"),
    "pcie": Link("pcie", 30e-6, 12e9, "Day 5 pcie_class assumption"),
    "network-100gbe": Link("network-100gbe", 50e-6, 10e9, "100 GbE, ~80% of line rate; TP across nodes"),
}


def _link(name: str) -> Link:
    try:
        return LINKS[name]
    except KeyError:
        raise ValueError(f"unknown link {name!r}; valid: {sorted(LINKS)}") from None


def _check_tp(tp: int) -> None:
    if tp < 1:
        raise ValueError(f"tp must be >= 1, got {tp}")


def tp_decode_step(model_preset: str, hardware: str, link: str, *, tp: int, batch_tokens: int) -> dict:
    _check_tp(tp)
    model, hw, lk = kv_sizing.model_spec(model_preset), kv_sizing.HARDWARE[hardware], _link(link)
    layers = model.num_layers
    n_ar = ALLREDUCES_PER_LAYER * layers
    comm_bytes = n_ar * batch_tokens * HIDDEN[model_preset] * ACTIVATION_BYTES
    beta = bus_bandwidth_factor("all_reduce", tp) / lk.busbw_bytes_s if tp > 1 else 0.0
    pred = predict_tp_speedup(
        flops=2 * model.params * batch_tokens,
        weight_bytes=weight_bytes(kv_sizing.deployment_for(model)),
        comm_bytes=comm_bytes,
        device_flops=hw.peak_flops * hw.assumed_mfu,
        device_bw=hw.mem_bw_bytes_s,
        comm_alpha_s=n_ar * lk.alpha_s,
        comm_beta_s_per_byte=beta,
        world=tp,
    )
    return {
        "link": link, "tp": tp, "batch_tokens": batch_tokens, "n_allreduce": n_ar,
        "comm_bytes": comm_bytes,
        "t_single_ms": pred.t_single * MS_PER_S, "t_tp_ms": pred.t_tp * MS_PER_S,
        "t_comm_ms": pred.t_comm * MS_PER_S, "speedup": pred.speedup,
        "comm_fraction": pred.comm_fraction,
    }


def _kv_shard_factor(num_kv_heads: int, tp: int) -> tuple[int, bool, bool]:
    """(ways a token's KV is split across the group, sharded, replicated).

    kv_heads divisible by tp: each GPU holds kv_heads/tp heads (sharded). tp a multiple of kv_heads:
    each head lives on tp/kv_heads GPUs, so the split is kv_heads-way (vLLM replicates heads).
    Otherwise no useful split is assumed."""
    if num_kv_heads % tp == 0:
        return tp, tp > 1, False
    if tp % num_kv_heads == 0:
        return num_kv_heads, False, True
    return 1, False, False


def kv_pool_per_gpu(model_preset: str, hardware: str, util: float, *, tp: int) -> tuple[int, int]:
    """(blocks, block_bytes) of the KV pool each GPU holds when weights and KV heads are sharded over `tp`.

    ``block_bytes`` is one block's size on one GPU (the KV shard), so ``blocks * block_bytes`` is the
    per-GPU value for vLLM's ``--kv-cache-memory-bytes`` (which is per GPU) and ``blocks * BLOCK_SIZE``
    is the group's capacity in tokens."""
    _check_tp(tp)
    if not 0.0 < util <= 1.0:
        raise ValueError("util must be in (0, 1]")
    model, hw = kv_sizing.model_spec(model_preset), kv_sizing.HARDWARE[hardware]
    shard, _, _ = _kv_shard_factor(model.num_kv_heads, tp)
    per_gpu_bytes = (
        hw.gpu_bytes * util
        - weight_bytes(kv_sizing.deployment_for(model)) / tp
        - kv_sizing.ACTIVATION_RESERVE_GIB * kv_sizing.GIB
    )
    block_bytes = kv_sizing.BLOCK_SIZE * kv_sizing.kv_bytes_per_token_for(model) // shard
    return max(int(per_gpu_bytes // block_bytes), 0), block_bytes


def kv_capacity(model_preset: str, hardware: str, util: float, *, tp: int) -> dict:
    """Group-wide KV capacity (tokens) with weights sharded over `tp` and KV split per `_kv_shard_factor`.

    ``gain_vs_tp1`` is None when one GPU cannot hold the weights (the ratio is undefined, and
    infinity would not survive JSON)."""
    blocks, _ = kv_pool_per_gpu(model_preset, hardware, util, tp=tp)
    model, hw = kv_sizing.model_spec(model_preset), kv_sizing.HARDWARE[hardware]
    _, kv_sharded, kv_replicated = _kv_shard_factor(model.num_kv_heads, tp)
    tokens = blocks * kv_sizing.BLOCK_SIZE
    base = kv_sizing.kv_pool_blocks(model, hw, util) * kv_sizing.BLOCK_SIZE
    return {
        "tp": tp, "kv_sharded": kv_sharded, "kv_replicated": kv_replicated,
        "pool_tokens_total": tokens, "gain_vs_tp1": tokens / base if base else None,
    }


def sweep(
    model_preset: str, hardware: str, *, util: float, tps: tuple[int, ...] = (1, 2, 4, 8),
    batches: tuple[int, ...] = (1, 16, 64),
) -> list[dict]:
    rows = []
    for link in LINKS:
        for tp in tps:
            cap = kv_capacity(model_preset, hardware, util, tp=tp)
            for batch in batches:
                step = tp_decode_step(model_preset, hardware, link, tp=tp, batch_tokens=batch)
                rows.append({
                    "model": model_preset, "hardware": hardware, **step,
                    "pool_tokens_total": cap["pool_tokens_total"], "gain_vs_tp1": cap["gain_vs_tp1"],
                    "kv_sharded": cap["kv_sharded"], "kv_replicated": cap["kv_replicated"], "assumed": True,
                })
    return rows
