"""Experiments A (prompt scaling), B (output scaling), C (KV cache on/off).

Usage:
    python run_experiments.py --exp all
    python run_experiments.py --exp C --smoke        # tiny sizes, sanity check only
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Sequence

import torch

from bench import summarize
from device import MemoryTracker, pick_device, release_cached_memory
from generate import DEFAULT_MODEL_ID, GenerationResult, generate_cached, generate_uncached
from model_io import eos_token_ids, load
from prompts import make_prompt_ids

BYTES_PER_MB = 1024 * 1024
RESULTS_DIR = Path(__file__).parent / "results"
DRIFT_WARN_FRACTION = 0.10  # cached vs uncached prefill should agree within this

Row = dict[str, object]
IdsFactory = Callable[[int], torch.Tensor]
Generator = Callable[..., GenerationResult]


@dataclass(frozen=True)
class ExperimentConfig:
    warmup: int
    repeats: int
    a_prompt_lens: tuple[int, ...]
    a_new_tokens: int
    b_prompt_len: int
    b_new_tokens: tuple[int, ...]
    c_prompt_len: int
    c_new_tokens: tuple[int, ...]
    uncached_warmup: int = 1  # uncached runs are long; extra warmups only add heat
    cooldown_s: float = 0.0  # pause between configurations to limit thermal drift


FULL = ExperimentConfig(
    warmup=3,
    repeats=5,
    a_prompt_lens=(128, 512, 1024, 2048),
    a_new_tokens=64,
    b_prompt_len=512,
    b_new_tokens=(32, 128, 256),
    c_prompt_len=512,
    c_new_tokens=(32, 128, 256),
    cooldown_s=10.0,
)
SMOKE = ExperimentConfig(
    warmup=1,
    repeats=1,
    a_prompt_lens=(16, 32),
    a_new_tokens=4,
    b_prompt_len=16,
    b_new_tokens=(4, 8),
    c_prompt_len=16,
    c_new_tokens=(4, 8),
)

GENERATORS: dict[str, Generator] = {"cached": generate_cached, "uncached": generate_uncached}


def kv_bytes_per_token(model) -> int:
    """K and V, for every layer: 2 * layers * kv_heads * head_dim * bytes/elem."""
    cfg = model.config
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    elem_bytes = torch.empty(0, dtype=next(model.parameters()).dtype).element_size()
    return 2 * cfg.num_hidden_layers * cfg.num_key_value_heads * head_dim * elem_bytes


def collect_runs(
    model,
    input_ids: torch.Tensor,
    n_new: int,
    eos_ids: Sequence[int],
    modes: Sequence[str],
    cfg: ExperimentConfig,
) -> tuple[dict[str, tuple[GenerationResult, ...]], dict[str, float]]:
    """Warm up each mode, then run the timed repeats *interleaved* (A B A B ...).

    Interleaving means slow drift (thermal throttling, clocks) hits every mode equally
    instead of biasing whichever mode happens to run later.
    Returns (runs per mode, baseline memory in MB per mode).
    """
    device = input_ids.device
    release_cached_memory(device)
    baselines = {}
    for mode in modes:
        tracker = MemoryTracker(device)
        tracker.reset()
        baselines[mode] = tracker.peak_bytes / BYTES_PER_MB

    def run(mode: str) -> GenerationResult:
        return GENERATORS[mode](model, input_ids, n_new, eos_ids, suppress_eos=True)

    for mode in modes:
        for _ in range(cfg.uncached_warmup if mode == "uncached" else cfg.warmup):
            run(mode)
    collected: dict[str, list[GenerationResult]] = {mode: [] for mode in modes}
    for _ in range(cfg.repeats):
        for mode in modes:
            collected[mode].append(run(mode))
    if cfg.cooldown_s > 0:
        time.sleep(cfg.cooldown_s)
    return {mode: tuple(runs) for mode, runs in collected.items()}, baselines


def summarize_runs(
    mode: str,
    input_ids: torch.Tensor,
    n_new: int,
    runs: Sequence[GenerationResult],
    baseline_mb: float,
) -> Row:
    prefill = summarize([r.prefill_s for r in runs])
    total = summarize([r.total_s for r in runs]).median
    peak_mb = max(r.peak_mem_bytes for r in runs) / BYTES_PER_MB
    generated = len(runs[-1].token_ids)
    return {
        "mode": mode,
        "prompt_len": int(input_ids.shape[1]),
        "new_tokens": n_new,
        "prefill_ms": prefill.median * 1e3,
        "prefill_ms_min": prefill.minimum * 1e3,
        "prefill_ms_max": prefill.maximum * 1e3,
        "prefill_tok_per_s": input_ids.shape[1] / prefill.median,
        "decode_total_s": summarize([sum(r.step_times_s) for r in runs]).median,
        # For uncached this is the mean cost of re-running a *growing* sequence, so it
        # depends on n_new and is not comparable to the cached per-token decode cost.
        "decode_ms_per_token": summarize([r.decode_s_per_token for r in runs]).median * 1e3,
        "total_s": total,
        "tokens_per_s": generated / total,
        "peak_mem_mb": peak_mb,
        "baseline_mem_mb": baseline_mb,
        "peak_over_baseline_mb": max(0.0, peak_mb - baseline_mb),
        "step_times_ms": [t * 1e3 for t in runs[-1].step_times_s],
    }


def measure_mode(model, input_ids, n_new, eos_ids, mode, cfg) -> Row:
    runs, baselines = collect_runs(model, input_ids, n_new, eos_ids, (mode,), cfg)
    return summarize_runs(mode, input_ids, n_new, runs[mode], baselines[mode])


def run_exp_a(model, make_ids: IdsFactory, eos_ids, cfg: ExperimentConfig) -> list[Row]:
    """Prompt-length scaling at a fixed output length (cached)."""
    return [
        measure_mode(model, make_ids(n), cfg.a_new_tokens, eos_ids, "cached", cfg)
        for n in cfg.a_prompt_lens
    ]


def run_exp_b(model, make_ids: IdsFactory, eos_ids, cfg: ExperimentConfig) -> list[Row]:
    """Output-length scaling at a fixed prompt length (cached)."""
    ids = make_ids(cfg.b_prompt_len)
    return [measure_mode(model, ids, n, eos_ids, "cached", cfg) for n in cfg.b_new_tokens]


def run_exp_c(model, make_ids: IdsFactory, eos_ids, cfg: ExperimentConfig) -> list[Row]:
    """Cached vs uncached at the same prompt and output lengths, interleaved."""
    ids = make_ids(cfg.c_prompt_len)
    per_token = kv_bytes_per_token(model)
    modes = ("cached", "uncached")
    rows: list[Row] = []
    for n in cfg.c_new_tokens:
        runs, baselines = collect_runs(model, ids, n, eos_ids, modes, cfg)
        reference = runs["cached"][0].token_ids
        all_match = all(r.token_ids == reference for rs in runs.values() for r in rs)
        per_mode = {m: summarize_runs(m, ids, n, runs[m], baselines[m]) for m in modes}
        speedup = float(per_mode["uncached"]["total_s"]) / float(per_mode["cached"]["total_s"])
        # Identical work in both modes (same first forward over the prompt); a large gap
        # means the machine drifted (thermals/clocks) and the speedup should be distrusted.
        drift = abs(per_mode["uncached"]["prefill_ms"] - per_mode["cached"]["prefill_ms"]) / (
            per_mode["cached"]["prefill_ms"]
        )
        # The final generated token is never fed back, so the cache holds prompt + n - 1.
        analytic_kv_mb = per_token * (cfg.c_prompt_len + n - 1) / BYTES_PER_MB
        for mode in modes:
            rows.append(
                {
                    **per_mode[mode],
                    "tokens_match_other_mode": all_match,
                    "speedup_vs_uncached": speedup,
                    "prefill_drift_fraction": drift,
                    "drift_warning": drift > DRIFT_WARN_FRACTION,
                    "analytic_kv_mb": analytic_kv_mb if mode == "cached" else 0.0,
                }
            )
    return rows


def run_metadata(model, device: torch.device, cfg: ExperimentConfig, smoke: bool) -> dict:
    return {
        "model": DEFAULT_MODEL_ID,
        "device": str(device),
        "dtype": str(next(model.parameters()).dtype),
        "torch": torch.__version__,
        "machine": platform.platform(),
        "smoke": smoke,
        "config": asdict(cfg),
        "kv_bytes_per_token": kv_bytes_per_token(model),
    }


def write_results(name: str, rows: list[Row], meta: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{name}.json").write_text(json.dumps({"meta": meta, "rows": rows}, indent=2))
    scalar_keys = [k for k, v in rows[0].items() if not isinstance(v, list)]
    with (out_dir / f"{name}.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=scalar_keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


EXPERIMENTS = {"A": run_exp_a, "B": run_exp_b, "C": run_exp_c}


def _print_row(row: Row) -> None:
    shown = {k: v for k, v in row.items() if k != "step_times_ms"}
    print({k: round(v, 3) if isinstance(v, float) else v for k, v in shown.items()}, flush=True)


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp", choices=["A", "B", "C", "all"], default="all")
    parser.add_argument("--smoke", action="store_true", help="tiny sizes; not for reporting")
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()

    cfg = SMOKE if args.smoke else FULL
    device = pick_device(args.device)
    tokenizer, model = load(DEFAULT_MODEL_ID, device)
    eos_ids = eos_token_ids(tokenizer)

    def make_ids(n: int) -> torch.Tensor:
        return make_prompt_ids(tokenizer, n).to(device)

    meta = run_metadata(model, device, cfg, args.smoke)
    names = list(EXPERIMENTS) if args.exp == "all" else [args.exp]
    suffix = "_smoke" if args.smoke else ""
    for name in names:
        print(f"== Experiment {name} on {device} ==", flush=True)
        rows = EXPERIMENTS[name](model, make_ids, eos_ids, cfg)
        write_results(f"exp_{name.lower()}{suffix}", rows, meta, args.out)
        for row in rows:
            _print_row(row)
            if row.get("drift_warning"):
                print("  WARNING: cached/uncached prefill differ by >10%; distrust speedup", flush=True)


if __name__ == "__main__":
    _main()
