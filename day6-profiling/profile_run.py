"""Profile one representative Day 1 generation run (prefill window and decode window).

Usage:
    python profile_run.py --device mps --prompt-len 512 --new-tokens 33 --tag mps-baseline
    python profile_run.py --smoke --device cpu --out /tmp/x     # tiny random model, no download

Windowing: warmup generations run first (unprofiled). The prefill window is one
`generate_cached(max_new_tokens=1)`. The decode window starts inside Day 1's `on_step`
hook at the first decode step, so it contains only decode and no prefill.

On CUDA, `torch.cuda.profiler.start/stop` bracket the window and NVTX ranges label
attention/MLP, so the script can be wrapped by `nsys --capture-range=cudaProfilerApi`
(see gpu/run_nsys.sh, which passes `--nsys`: windows only, no torch.profiler, no probes).
Module ranges on async devices time CPU-side issue, not device work.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

import _day1  # noqa: F401
import torch
from device import pick_device, sync
from generate import generate_cached
from torch.profiler import ProfilerActivity, profile, record_function

from alloc_probe import measure_allocations
from launch_probe import measure_launch
from module_timing import (
    ModuleTiming,
    attribution,
    coarse_selector,
    replay_linears,
    sync_bracket_overhead_us,
    time_modules,
)
from op_classify import category_totals, classify, rows_from_events, top_ops
from run_meta import collect_meta
from trace_analysis import analyze_gaps, load_trace, summarize_alloc_events

RESULTS_DIR = Path(__file__).parent / "results" / "raw"
TOP_OPS_KEPT = 15
PREFILL_TIMING_REPEATS = 5
DECODE_TIMING_REPEATS = 3
LABELLED_MODULES = {"self_attn": "d6::attn", "mlp": "d6::mlp"}
SMOKE_EOS = (1, 2)
SMOKE_VOCAB = 128


@dataclass(frozen=True)
class ProfileConfig:
    phase: str  # "prefill" | "decode"
    prompt_len: int
    new_tokens: int
    warmup: int
    attn_impl: str


class _Label:
    """A `record_function` range (plus an NVTX range on CUDA) opened/closed by module hooks."""

    def __init__(self, name: str, use_nvtx: bool) -> None:
        self._name, self._use_nvtx = name, use_nvtx
        self._ctx: record_function | None = None

    def enter(self) -> None:
        self._ctx = record_function(self._name)
        self._ctx.__enter__()
        if self._use_nvtx:
            torch.cuda.nvtx.range_push(self._name)

    def exit(self) -> None:
        if self._use_nvtx:
            torch.cuda.nvtx.range_pop()
        if self._ctx is not None:
            self._ctx.__exit__(None, None, None)
            self._ctx = None


@contextlib.contextmanager
def module_labels(model, use_nvtx: bool = False) -> Iterator[int]:
    """Label every decoder layer's attention and MLP; hooks are removed on exit, even on error."""
    labelled = 0
    with contextlib.ExitStack() as stack:
        for layer in model.model.layers:
            for attr, name in LABELLED_MODULES.items():
                module = getattr(layer, attr, None)
                if module is None:
                    continue
                label = _Label(name, use_nvtx)
                stack.callback(module.register_forward_pre_hook(lambda _m, _a, lbl=label: lbl.enter()).remove)
                stack.callback(module.register_forward_hook(lambda _m, _a, _o, lbl=label: lbl.exit()).remove)
                labelled += 1
        yield labelled


def _activities(device: torch.device) -> list[ProfilerActivity]:
    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)
    return activities


def _new_profiler(device: torch.device) -> profile:
    return profile(activities=_activities(device), profile_memory=True, record_shapes=False)


class _Window:
    """start()/stop() for one profiled window.

    On CUDA it calls `torch.cuda.profiler.start/stop` (cudaProfilerStart/Stop), which is what
    `nsys --capture-range=cudaProfilerApi` waits for. The torch profiler is optional: under nsys
    it is skipped (`prof=None`) so Kineto's CUPTI subscription cannot clash with nsys.
    """

    def __init__(self, device: torch.device, prof: profile | None) -> None:
        self._cuda, self._prof, self._active = device.type == "cuda", prof, False

    def start(self) -> None:
        if self._active:
            return
        self._active = True
        if self._cuda:
            torch.cuda.profiler.start()
        if self._prof is not None:
            self._prof.start()

    def stop(self) -> None:
        if not self._active:
            return
        self._active = False
        if self._prof is not None:
            self._prof.stop()
        if self._cuda:
            torch.cuda.profiler.stop()


def run_window(
    model, input_ids: torch.Tensor, cfg: ProfileConfig, eos_ids, use_torch_profiler: bool = True
) -> profile | None:
    """Warm up, then profile exactly one window. Returns the stopped torch profiler (or None)."""
    device = input_ids.device
    for _ in range(cfg.warmup):
        generate_cached(model, input_ids, max(2, cfg.new_tokens), eos_ids, suppress_eos=True)
    sync(device)

    prof = _new_profiler(device) if use_torch_profiler else None
    window = _Window(device, prof)
    try:
        with module_labels(model, use_nvtx=device.type == "cuda"):
            if cfg.phase == "prefill":
                window.start()
                generate_cached(model, input_ids, 1, eos_ids, suppress_eos=True)
            else:
                generate_cached(
                    model, input_ids, cfg.new_tokens, eos_ids, suppress_eos=True,
                    on_step=lambda _cache_len: window.start(),  # first decode step opens the window
                )
            sync(device)
    finally:
        window.stop()
    return prof


def summarize_window(prof: profile, trace_path: Path, cfg: ProfileConfig, device: torch.device) -> dict:
    prof.export_chrome_trace(str(trace_path))
    rows = rows_from_events(prof.key_averages())
    use_device = device.type == "cuda"
    totals = category_totals(rows, use_device_time=use_device)
    steps = 1 if cfg.phase == "prefill" else cfg.new_tokens - 1
    by_name = {r.name: r for r in rows}
    module_ms = {
        name: by_name[name].total_cpu_us / 1e3 / steps for name in LABELLED_MODULES.values() if name in by_name
    }
    aten_ops = sum(r.count for r in rows if r.name.startswith("aten::"))
    trace = load_trace(trace_path)
    gaps = analyze_gaps(trace)
    allocs = summarize_alloc_events(trace)
    return {
        "phase": cfg.phase,
        "steps_in_window": steps,
        "time_basis": "device" if use_device else "cpu-self-time (device kernels not recorded)",
        "category_totals": {c: asdict(t) for c, t in totals.items()},
        "top_ops": [asdict(r) | {"category": classify(r.name)} for r in top_ops(rows, TOP_OPS_KEPT, use_device)],
        "module_cpu_ms_per_step": module_ms,
        "aten_ops_per_step": aten_ops / steps,
        "gaps": {k: v for k, v in asdict(gaps).items() if k != "gaps_us"},
        "alloc_events": asdict(allocs),
    }


def write_ops_csv(prof: profile, path: Path) -> None:
    rows = rows_from_events(prof.key_averages())
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["op", "category", "count", "self_cpu_us", "self_device_us", "total_cpu_us"])
        for r in sorted(rows, key=lambda r: r.self_cpu_us, reverse=True):
            writer.writerow([r.name, classify(r.name), r.count, r.self_cpu_us, r.self_device_us, r.total_cpu_us])


def _median_by_label(timings: list[ModuleTiming]) -> dict[str, float]:
    labels = timings[0].ms_per_step.keys()
    return {k: statistics.median(t.ms_per_step[k] for t in timings) for k in labels}


def device_attribution(model, input_ids, eos_ids, phase: str, step_ms: float, n_steps: int) -> dict:
    """Matmul vs attention-core vs other on a backend with no device timeline (see module_timing.py)."""
    repeats = PREFILL_TIMING_REPEATS if phase == "prefill" else DECODE_TIMING_REPEATS
    steps = 1 if phase == "prefill" else n_steps
    coarse = [time_modules(model, input_ids, eos_ids, coarse_selector, phase, steps) for _ in range(repeats)]
    coarse_ms = _median_by_label(coarse)
    replay = replay_linears(model, input_ids, eos_ids, phase)
    overhead_us = sync_bracket_overhead_us(input_ids.device)
    attn_overhead_ms = coarse[0].calls_per_step.get("attn", 0) * overhead_us / 1e3
    return {
        "step_ms_uninstrumented": step_ms,
        "coarse_bracketed_ms_per_step": coarse_ms,
        "matmul_replay_ms": replay.ms_per_step,
        "matmul_replay_calls": replay.calls_per_step,
        "matmul_replay_enqueue_fraction": replay.enqueue_fraction,
        "sync_bracket_overhead_us": overhead_us,
        "attn_overhead_ms": attn_overhead_ms,
        **attribution(coarse_ms, replay.ms_per_step, step_ms, attn_overhead_ms),
    }


def uninstrumented_prefill_ms(model, input_ids, eos_ids, repeats: int) -> float:
    runs = [generate_cached(model, input_ids, 1, eos_ids, suppress_eos=True).prefill_s for _ in range(repeats)]
    return statistics.median(runs) * 1e3


def smoke_model():
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(0)
    config = Qwen2Config(
        vocab_size=SMOKE_VOCAB, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512,
    )
    return Qwen2ForCausalLM(config).eval()


def build_inputs(device: torch.device, prompt_len: int, attn_impl: str, smoke: bool):
    if smoke:
        model = smoke_model().to(device)
        generator = torch.Generator().manual_seed(0)
        ids = torch.randint(3, SMOKE_VOCAB, (1, prompt_len), generator=generator).to(device)
        return model, ids, SMOKE_EOS
    from loader import load_model
    from model_io import eos_token_ids
    from prompts import make_prompt_ids

    tokenizer, model = load_model(device, attn_impl)
    return model, make_prompt_ids(tokenizer, prompt_len).to(device), eos_token_ids(tokenizer)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default=None)
    parser.add_argument("--phase", choices=["prefill", "decode", "both"], default="both")
    parser.add_argument("--prompt-len", type=int, default=512)
    parser.add_argument("--new-tokens", type=int, default=33, help="decode window = new_tokens - 1 steps")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--attn-impl", choices=["sdpa", "eager"], default="sdpa")
    parser.add_argument("--tag", default="profile")
    parser.add_argument("--out", type=Path, default=RESULTS_DIR)
    parser.add_argument("--smoke", action="store_true", help="tiny random model; not for reporting")
    parser.add_argument("--nsys", action="store_true",
                        help="CUDA only: open/close the cudaProfilerApi window with no torch.profiler/probes")
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if args.new_tokens < 3:
        raise SystemExit("--new-tokens must be >= 3 so the decode window has at least 2 steps")
    if args.prompt_len < 1:
        raise SystemExit("--prompt-len must be >= 1")
    if args.warmup < 0:
        raise SystemExit("--warmup must be >= 0")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    validate_args(args)
    device = pick_device(args.device)
    if args.nsys and device.type != "cuda":
        raise SystemExit("--nsys needs --device cuda")
    model, input_ids, eos_ids = build_inputs(device, args.prompt_len, args.attn_impl, args.smoke)
    out_dir = args.out / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    phases = ["prefill", "decode"] if args.phase == "both" else [args.phase]
    summary: dict = {"phases": {}}
    for phase in phases:
        cfg = ProfileConfig(phase, args.prompt_len, args.new_tokens, args.warmup, args.attn_impl)
        prof = run_window(model, input_ids, cfg, eos_ids, use_torch_profiler=not args.nsys)
        if prof is None:  # --nsys: the window ran under cudaProfilerApi; nothing else to record
            print(f"[{phase}] cudaProfilerApi window done", flush=True)
            continue
        write_ops_csv(prof, out_dir / f"ops_{phase}.csv")
        summary["phases"][phase] = summarize_window(prof, out_dir / f"trace_{phase}.json", cfg, device)
        print(f"[{phase}] profiled; artifacts in {out_dir}", flush=True)

    if args.nsys:
        return 0
    n_steps = args.new_tokens - 1
    summary["launch_probe"] = asdict(measure_launch(model, input_ids, n_steps))
    alloc = asdict(measure_allocations(model, input_ids, n_steps))
    summary["alloc_probe"] = {k: v for k, v in alloc.items() if k != "deltas"}
    summary["alloc_deltas_bytes"] = alloc["deltas"]
    summary["device_attribution"] = {
        "decode": device_attribution(
            model, input_ids, eos_ids, "decode", summary["launch_probe"]["step_ms_median"], n_steps
        ),
        "prefill": device_attribution(
            model, input_ids, eos_ids, "prefill",
            uninstrumented_prefill_ms(model, input_ids, eos_ids, PREFILL_TIMING_REPEATS), 1,
        ),
    }
    summary["meta"] = collect_meta(device, model, vars(args) | {"out": str(args.out)}, args.smoke)
    (out_dir / "profile_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps({k: summary[k] for k in ("launch_probe", "alloc_probe")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
