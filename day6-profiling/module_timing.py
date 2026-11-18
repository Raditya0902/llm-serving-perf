"""Device-inclusive time per module on backends whose profiler has no device timeline (MPS).

On MPS the PyTorch profiler records only CPU-side operators and almost all CPU self-time
lands in the one `.item()` wait, so it cannot say how long the *device* spent in matmuls
vs attention. Two techniques fill the gap:

  replay_linears  - capture every nn.Linear's real input during one step, then re-run all
                    of them back-to-back with ONE sync. Accurate device time for matmuls
                    (real weights, real shapes), grouped as proj (q/k/v/o), mlp_linear,
                    head_linear. `enqueue_fraction` flags when the CPU, not the device,
                    limits the replay (then the time is only an upper bound).
  time_modules    - sync before/after each coarse module (attn/mlp/norm/...). Each sync
                    costs a command-buffer round trip (~0.2 ms on MPS), so absolute values
                    are inflated; `sync_bracket_overhead_us` calibrates that cost so it can
                    be subtracted for the attention estimate.

  matmul          = proj + mlp_linear + head_linear               (replay, accurate)
  attention core  = attn_bracketed - proj - calls*overhead        (estimate)
"""

from __future__ import annotations

import contextlib
import statistics
import time
from dataclasses import dataclass
from typing import Callable, Sequence

import _day1  # noqa: F401
import torch
from device import sync
from generate import generate_cached

COARSE_LABELS = ("attn", "mlp", "norm", "lm_head", "embed", "rotary")
Selector = Callable[[str, torch.nn.Module], str | None]
DEFAULT_OVERHEAD_REPEATS = 200


@dataclass(frozen=True)
class ModuleTiming:
    phase: str
    steps: int
    ms_per_step: dict[str, float]
    calls_per_step: dict[str, int]


def coarse_selector(name: str, module: torch.nn.Module) -> str | None:
    leaf = name.rsplit(".", 1)[-1]
    if leaf == "self_attn":
        return "attn"
    if leaf == "mlp":
        return "mlp"
    if leaf in ("input_layernorm", "post_attention_layernorm") or name == "model.norm":
        return "norm"
    if name == "lm_head":
        return "lm_head"
    if leaf == "embed_tokens":
        return "embed"
    if leaf == "rotary_emb":
        return "rotary"
    return None


def linear_selector(name: str, module: torch.nn.Module) -> str | None:
    if not isinstance(module, torch.nn.Linear):
        return None
    if name == "lm_head":
        return "head_linear"
    return "mlp_linear" if ".mlp." in name else "proj"


def sync_bracket_overhead_us(device: torch.device, repeats: int = DEFAULT_OVERHEAD_REPEATS) -> float:
    """Median cost of `sync; tiny op; sync`: the floor each bracketed module call pays."""
    scratch = torch.zeros(8, device=device)
    samples = []
    for _ in range(repeats):
        sync(device)
        start = time.perf_counter()
        scratch.add_(1)
        sync(device)
        samples.append(time.perf_counter() - start)
    return sorted(samples)[len(samples) // 2] * 1e6


def _check_phase(phase: str) -> None:
    if phase not in ("prefill", "decode"):
        raise ValueError("phase must be 'prefill' or 'decode'")


@contextlib.contextmanager
def _hooked(model, selector: Selector, make_hooks):
    """Register hooks from `make_hooks(label) -> (pre, post|None)`; always removed, even on error."""
    with contextlib.ExitStack() as stack:
        for name, module in model.named_modules():
            label = selector(name, module)
            if label is None:
                continue
            pre, post = make_hooks(label)
            stack.callback(module.register_forward_pre_hook(pre).remove)
            if post is not None:
                stack.callback(module.register_forward_hook(post).remove)
        yield


def _run_window(model, input_ids, eos_ids, phase: str, n_steps: int, state: dict) -> int:
    """Run generation; hooks are enabled for the whole of prefill, or from the first decode step."""

    def enable_on_first_decode(_cache_len: int) -> None:
        state["enabled"] = True

    if phase == "prefill":
        generate_cached(model, input_ids, 1, eos_ids, suppress_eos=True)
        return 1
    generate_cached(model, input_ids, n_steps + 1, eos_ids, suppress_eos=True, on_step=enable_on_first_decode)
    return n_steps


def time_modules(
    model,
    input_ids: torch.Tensor,
    eos_ids: Sequence[int],
    selector: Selector,
    phase: str,
    n_steps: int,
) -> ModuleTiming:
    _check_phase(phase)
    device = input_ids.device
    totals: dict[str, float] = {}
    calls: dict[str, int] = {}
    starts: dict[int, float] = {}
    state = {"enabled": phase == "prefill"}

    def make_hooks(label: str):
        def pre(module, _args):
            if state["enabled"]:
                sync(device)
                starts[id(module)] = time.perf_counter()

        def post(module, _args, _out):
            if state["enabled"]:
                sync(device)
                elapsed = time.perf_counter() - starts.pop(id(module))
                totals[label] = totals.get(label, 0.0) + elapsed
                calls[label] = calls.get(label, 0) + 1

        return pre, post

    with _hooked(model, selector, make_hooks):
        steps = _run_window(model, input_ids, eos_ids, phase, n_steps, state)

    uneven = {k: c for k, c in calls.items() if c % steps}
    if uneven:  # a window that leaked prefill (or lost a step) would show up here
        raise RuntimeError(f"module call counts {uneven} are not a multiple of {steps} steps")
    return ModuleTiming(
        phase=phase,
        steps=steps,
        ms_per_step={k: v * 1e3 / steps for k, v in totals.items()},
        calls_per_step={k: calls[k] // steps for k in calls},
    )


@dataclass(frozen=True)
class ReplayTiming:
    phase: str
    ms_per_step: dict[str, float]  # group -> device-inclusive ms for one step's worth of calls
    calls_per_step: dict[str, int]
    enqueue_fraction: dict[str, float]  # CPU enqueue time / total replay time per group


def _capture_linear_inputs(model, input_ids, eos_ids, phase: str) -> list[tuple[str, torch.nn.Module, torch.Tensor]]:
    """Inputs of every nn.Linear for one real step (the whole prompt, or the first decode step)."""
    captured: list[tuple[str, torch.nn.Module, torch.Tensor]] = []
    state = {"enabled": phase == "prefill"}

    def make_capture(label: str):
        def capture(module, args):
            if state["enabled"]:
                captured.append((label, module, args[0].detach().clone()))

        return capture, None

    def stop_after_first_decode(_cache_len: int) -> None:
        state["enabled"] = not captured

    with _hooked(model, linear_selector, make_capture):
        if phase == "prefill":
            generate_cached(model, input_ids, 1, eos_ids, suppress_eos=True)
        else:
            generate_cached(model, input_ids, 3, eos_ids, suppress_eos=True, on_step=stop_after_first_decode)
    return captured


def _time_group(work, device: torch.device, repeats: int) -> tuple[float, float]:
    """Median (total seconds, enqueue seconds) to run `work` back-to-back with one sync."""
    totals, enqueues = [], []
    for _ in range(repeats + 2):  # first two repeats are warmup
        sync(device)
        start = time.perf_counter()
        for module, x in work:
            module(x)
        enqueued = time.perf_counter()
        sync(device)
        totals.append(time.perf_counter() - start)
        enqueues.append(enqueued - start)
    return statistics.median(totals[2:]), statistics.median(enqueues[2:])


def replay_linears(
    model, input_ids: torch.Tensor, eos_ids: Sequence[int], phase: str, repeats: int = 20
) -> ReplayTiming:
    _check_phase(phase)
    captured = _capture_linear_inputs(model, input_ids, eos_ids, phase)
    ms: dict[str, float] = {}
    enqueue: dict[str, float] = {}
    calls: dict[str, int] = {}
    with torch.no_grad():
        for group in sorted({label for label, _, _ in captured}):
            work = [(m, x) for label, m, x in captured if label == group]
            total, enq = _time_group(work, input_ids.device, repeats)
            calls[group], ms[group], enqueue[group] = len(work), total * 1e3, enq / total
    return ReplayTiming(phase, ms, calls, enqueue)


def attribution(
    coarse_ms: dict[str, float], matmul_ms: dict[str, float], step_ms: float, attn_overhead_ms: float = 0.0
) -> dict[str, float | bool]:
    """Matmul / attention-core / other as shares of the real (uninstrumented) step time.

    `attention_core_ms` is an ESTIMATE (a serialised, sync-bracketed measurement minus a
    pipelined replay minus a calibrated overhead), so it is flagged when the subtraction
    went negative (`attention_core_clamped`) and `other_negative` is set if the parts exceed
    the real step: either means the pieces do not reconcile and the split is unreliable.
    """
    if step_ms <= 0:
        raise ValueError("step_ms must be positive")
    matmul = sum(matmul_ms.get(k, 0.0) for k in ("proj", "mlp_linear", "head_linear"))
    raw_core = coarse_ms.get("attn", 0.0) - matmul_ms.get("proj", 0.0) - attn_overhead_ms
    core = max(0.0, raw_core)
    other = step_ms - matmul - core
    return {
        "matmul_ms": matmul,
        "attention_core_ms": core,
        "other_ms": other,
        "matmul_share": matmul / step_ms,
        "attention_core_share": core / step_ms,
        "attention_core_clamped": raw_core < 0.0,
        "other_negative": other < 0.0,
    }
