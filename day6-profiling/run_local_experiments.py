"""Run the experiments that can be measured on the dev machine (E1-E3, O1, O2).

Usage:
    python run_local_experiments.py --exp all --device mps --tag mps
    python run_local_experiments.py --exp O1 --smoke --device cpu --out /tmp/x   # tiny model

Every experiment is: baseline + one changed parameter, interleaved repeats, a validation
column (do the generated tokens match?), and a noise check against the baseline's spread.
The hypothesis text comes from experiments.py and is stored in each result file.
"""

from __future__ import annotations

import argparse
import copy
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import _day1  # noqa: F401
import torch
from device import pick_device
from run_experiments import write_results  # Day 1's JSON + CSV writer

from experiments import EXPERIMENTS, Experiment, get_experiment
from measure import (
    DRIFT_WARN_FRACTION,
    balanced_repeats,
    flatten_tokens,
    interleaved,
    match_fraction,
    summarize_case,
    with_baseline_deltas,
)
from run_meta import collect_meta
from variants import DecodeRun, make_batch_ids, run_batched, run_day1, run_deferred_sync

RESULTS_DIR = Path(__file__).parent / "results" / "raw"
Reps = list[list[DecodeRun]]
Collected = dict[str, Reps]


@dataclass(frozen=True)
class RunConfig:
    warmup: int
    repeats: int
    cooldown_s: float
    pause_s: float  # sleep between experiments (lets the machine settle)
    prompt_len: int
    new_tokens: int
    n_requests: int
    overrides: dict = field(default_factory=dict)  # exp_id -> (values, baseline)


FULL = RunConfig(warmup=2, repeats=5, cooldown_s=0.0, pause_s=30.0, prompt_len=512, new_tokens=64, n_requests=8)
SMOKE = RunConfig(
    warmup=1, repeats=2, cooldown_s=0.0, pause_s=0.0, prompt_len=16, new_tokens=6, n_requests=4,
    overrides={"E1": ((1, 2, 4), 1), "E2": ((8, 16), 16), "E3": ((4, 8), 8)},
)


@dataclass(frozen=True)
class Context:
    get_model: Callable[[str], object]  # attn_impl -> model
    eos_ids: tuple[int, ...]
    make_ids: Callable[[int], torch.Tensor]  # prompt length -> [1, n] ids on device
    cfg: RunConfig


@dataclass(frozen=True)
class CaseSet:
    cases: dict[str, Callable[[], list[DecodeRun]]]
    baseline: str
    n_requests: int
    validate: Callable[[str, Collected], float]  # label -> fraction of matching token rows
    controls: frozenset[str] = frozenset()  # A/A cases: identical work, measure the noise floor


def values_and_baseline(exp: Experiment, cfg: RunConfig):
    return cfg.overrides.get(exp.exp_id, (exp.values, exp.baseline))


def _deterministic(label: str, collected: Collected) -> float:
    """Repeat-to-repeat determinism only: it does NOT check correctness against a reference."""
    reps = [flatten_tokens(rep) for rep in collected[label]]
    return min(match_fraction(r, reps[0]) for r in reps)


def _same_as_baseline(baseline: str):
    def validate(label: str, collected: Collected) -> float:
        pairs = zip(collected[label], collected[baseline])
        return min(match_fraction(flatten_tokens(a), flatten_tokens(b)) for a, b in pairs)

    return validate


def build_e1(ctx: Context, exp: Experiment) -> CaseSet:
    values, baseline = values_and_baseline(exp, ctx.cfg)
    model, n, total = ctx.get_model("sdpa"), ctx.cfg.new_tokens, ctx.cfg.n_requests
    indivisible = [v for v in values if total % v]
    if indivisible:
        raise ValueError(f"n_requests {total} is not divisible by batch size(s) {indivisible}")
    ids = make_batch_ids(ctx.make_ids(ctx.cfg.prompt_len), total)
    reference = tuple(run_day1(model, ids[i : i + 1], n, ctx.eos_ids).tokens[0] for i in range(total))

    def serve(batch: int) -> list[DecodeRun]:
        if total % batch:
            raise ValueError(f"n_requests {total} is not divisible by batch size {batch}")
        runner = run_day1 if batch == 1 else run_batched
        return [runner(model, ids[i : i + batch], n, ctx.eos_ids) for i in range(0, total, batch)]

    def validate(label: str, collected: Collected) -> float:
        return min(match_fraction(flatten_tokens(rep), reference) for rep in collected[label])

    return CaseSet({str(v): (lambda v=v: serve(v)) for v in values}, str(baseline), total, validate)


def build_e2(ctx: Context, exp: Experiment) -> CaseSet:
    values, baseline = values_and_baseline(exp, ctx.cfg)
    model, n = ctx.get_model("sdpa"), ctx.cfg.new_tokens
    cases = {str(v): (lambda v=v: [run_day1(model, ctx.make_ids(v), n, ctx.eos_ids)]) for v in values}
    return CaseSet(cases, str(baseline), 1, _deterministic)


def build_e3(ctx: Context, exp: Experiment) -> CaseSet:
    values, baseline = values_and_baseline(exp, ctx.cfg)
    model, ids = ctx.get_model("sdpa"), ctx.make_ids(ctx.cfg.prompt_len)
    cases = {str(v): (lambda v=v: [run_day1(model, ids, v, ctx.eos_ids)]) for v in values}
    return CaseSet(cases, str(baseline), 1, _deterministic)


def build_o1(ctx: Context, exp: Experiment) -> CaseSet:
    model, ids, n = ctx.get_model("sdpa"), ctx.make_ids(ctx.cfg.prompt_len), ctx.cfg.new_tokens
    def day1() -> list[DecodeRun]:
        return [run_day1(model, ids, n, ctx.eos_ids)]

    cases = {
        "day1": day1,
        "day1_control": day1,  # A/A: same code as the baseline; any gap is measurement noise/bias
        "deferred_sync": lambda: [run_deferred_sync(model, ids, n, ctx.eos_ids)],
    }
    return CaseSet(cases, "day1", 1, _same_as_baseline("day1"), frozenset({"day1_control"}))


def build_o2(ctx: Context, exp: Experiment) -> CaseSet:
    ids, n = ctx.make_ids(ctx.cfg.prompt_len), ctx.cfg.new_tokens
    cases = {
        impl: (lambda impl=impl: [run_day1(ctx.get_model(impl), ids, n, ctx.eos_ids)])
        for impl in exp.values
    }
    cases["sdpa_control"] = cases["sdpa"]  # A/A noise floor, as in O1
    return CaseSet(cases, "sdpa", 1, _same_as_baseline("sdpa"), frozenset({"sdpa_control"}))


BUILDERS = {"E1": build_e1, "E2": build_e2, "E3": build_e3, "O1": build_o1, "O2": build_o2}


def run_experiment(exp: Experiment, ctx: Context) -> list[dict]:
    case_set = BUILDERS[exp.exp_id](ctx, exp)
    cfg = ctx.cfg
    repeats = balanced_repeats(cfg.repeats, len(case_set.cases))
    collected = interleaved(case_set.cases, cfg.warmup, repeats, cfg.cooldown_s)
    rows = []
    for label, reps in collected.items():
        row = summarize_case(label, reps, case_set.n_requests)
        row["tokens_match"] = case_set.validate(label, collected)
        row["exp_id"], row["variable"] = exp.exp_id, exp.variable
        row["is_control"] = label in case_set.controls
        row["valid"] = row["tokens_match"] == 1.0  # False => outputs diverged: speed is not comparable
        rows.append(row)
    return with_baseline_deltas(rows, case_set.baseline)


def smoke_context(device: torch.device, cfg: RunConfig) -> Context:
    from profile_run import SMOKE_EOS, SMOKE_VOCAB, smoke_model

    base = smoke_model().to(device)
    models = {"sdpa": base}

    def get_model(impl: str):
        if impl not in models:
            clone = copy.deepcopy(base)
            clone.config._attn_implementation = impl
            models[impl] = clone
        return models[impl]

    generator = torch.Generator().manual_seed(0)

    def make_ids(n: int) -> torch.Tensor:
        return torch.randint(3, SMOKE_VOCAB, (1, n), generator=generator.manual_seed(n)).to(device)

    return Context(get_model, SMOKE_EOS, make_ids, cfg)


def real_context(device: torch.device, cfg: RunConfig) -> Context:
    from loader import load_model
    from model_io import eos_token_ids
    from prompts import make_prompt_ids

    models: dict[str, object] = {}
    tokenizer = None

    def get_model(impl: str):
        nonlocal tokenizer
        if impl not in models:
            tokenizer, models[impl] = load_model(device, impl)
        return models[impl]

    get_model("sdpa")
    return Context(
        get_model,
        eos_token_ids(tokenizer),
        lambda n: make_prompt_ids(tokenizer, n).to(device),
        cfg,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exp", nargs="+", default=["all"], help="E1 E2 E3 O1 O2 or 'all'")
    parser.add_argument("--device", default=None)
    parser.add_argument("--tag", default="local")
    parser.add_argument("--out", type=Path, default=RESULTS_DIR)
    parser.add_argument("--smoke", action="store_true", help="tiny random model; not for reporting")
    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument("--values", nargs="+", type=int, default=None,
                        help="override the swept values of ONE integer-valued experiment (E1-E3)")
    return parser.parse_args(argv)


def selected_experiments(names: list[str]) -> list[Experiment]:
    if names == ["all"]:
        return [e for e in EXPERIMENTS if e.runnable_on == "local"]
    if "all" in names:
        raise SystemExit("'all' cannot be combined with experiment ids")
    known = {e.exp_id for e in EXPERIMENTS}
    unknown = [n for n in names if n not in known]
    if unknown:
        raise SystemExit(f"unknown experiment(s) {unknown}; choose from {sorted(known)}")
    chosen = [get_experiment(n) for n in names]
    remote = [e.exp_id for e in chosen if e.runnable_on != "local"]
    if remote:
        raise SystemExit(f"{remote} need a GPU host (see RUNBOOK.md) or sim_predictions.py")
    return chosen


def values_override(chosen: list[Experiment], values: list[int]) -> dict:
    """`--values` applies to exactly one experiment whose values are integers; baseline must stay in."""
    if len(chosen) != 1 or not all(isinstance(v, int) for v in chosen[0].values):
        raise SystemExit("--values needs exactly one integer-valued experiment (E1, E2 or E3)")
    if min(values) < 1 or len(set(values)) != len(values):
        raise SystemExit("--values must be positive and unique")
    exp = chosen[0]
    if exp.baseline not in values:
        raise SystemExit(f"--values must include the baseline {exp.baseline}")
    return {exp.exp_id: (tuple(values), exp.baseline)}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.repeats is not None and args.repeats < 1:
        raise SystemExit("--repeats must be >= 1")
    if args.warmup is not None and args.warmup < 0:
        raise SystemExit("--warmup must be >= 0")
    chosen = selected_experiments(args.exp)
    base = SMOKE if args.smoke else FULL
    cfg = RunConfig(**{
        **asdict(base),
        **({"repeats": args.repeats} if args.repeats is not None else {}),
        **({"warmup": args.warmup} if args.warmup is not None else {}),
        **({"overrides": values_override(chosen, args.values)} if args.values else {}),
    })
    device = pick_device(args.device)
    ctx = (smoke_context if args.smoke else real_context)(device, cfg)
    out_dir = args.out / args.tag
    for index, exp in enumerate(chosen):
        if index and cfg.pause_s > 0:
            time.sleep(cfg.pause_s)
        print(f"== {exp.exp_id} {exp.name} on {device} ==", flush=True)
        rows = run_experiment(exp, ctx)
        meta = collect_meta(device, ctx.get_model("sdpa"), asdict(cfg), args.smoke)
        meta["experiment"] = asdict(exp)
        write_results(f"exp_{exp.exp_id.lower()}", rows, meta, out_dir)
        for row in rows:
            print({k: round(v, 3) if isinstance(v, float) else v for k, v in row.items()}, flush=True)
            if not row["valid"]:
                print(f"  WARNING: {row['label']} outputs diverged (tokens_match={row['tokens_match']:.2f}); "
                      "do not compare its speed", flush=True)
            if row["wall_drift_fraction"] > DRIFT_WARN_FRACTION:
                print(f"  WARNING: {row['label']} drifted >{DRIFT_WARN_FRACTION:.0%}; distrust", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
