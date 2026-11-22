"""Run the 12-case matrix (and an ablation) on the simulator in `03-batching-simulator`. SIMULATED, not measured.

    .venv/bin/python run_matrix_sim.py                     # 12 cases x 3 reps (+key reps, +A/A)
    .venv/bin/python run_matrix_sim.py --ablation          # also attribute `tuned` knob by knob
    .venv/bin/python run_matrix_sim.py --smoke --out /tmp/x

Writes ``matrix.{json,csv}`` (one row per run), ``matrix_summary.{json,csv}`` (one row per
case: mean/std/min/max/cv over repeats) and, with ``--ablation``, ``ablation*.{json,csv}``.
The vLLM runner (``run_matrix_gpu.py``) produces rows with the identical schema.
"""
from __future__ import annotations

import argparse
import dataclasses
import logging
from collections.abc import Sequence
from pathlib import Path

import closed_loop_sim
import kv_sizing
import matrix as mx
import results_io
import stability

logger = logging.getLogger(__name__)

SIM_NOTE = (
    "SIMULATED with the `03-batching-simulator` engine and a roofline cost model derived from illustrative GPU "
    "peaks (kv_sizing.HARDWARE); not a GPU measurement. peak_gpu_mem_gib is not measured."
)
DEFAULT_OUT = Path(__file__).resolve().parent / "results" / "sim"
DEFAULT_REPS, DEFAULT_KEY_REPS = 3, 5
ADMISSION_CAPS = (16, 24, 32)  # max_num_seqs values tried at the highest concurrency


def run_matrix(runs: Sequence[mx.RunSpec]) -> list[dict]:
    return [closed_loop_sim.row_for(run.case, run.rep) for run in runs]


def ablation_configs() -> list[mx.ServingConfig]:
    """baseline, tuned, each of tuned's two knobs alone, and admission caps (`max_num_seqs`)."""
    base, tuned = mx.CONFIGS["baseline"], mx.CONFIGS["tuned"]
    variants = [
        base, tuned,
        dataclasses.replace(base, name="util-only", gpu_memory_utilization=tuned.gpu_memory_utilization),
        dataclasses.replace(base, name="tokens-only", max_num_batched_tokens=tuned.max_num_batched_tokens),
    ]
    variants += [dataclasses.replace(base, name=f"cap{n}", max_num_seqs=n) for n in ADMISSION_CAPS]
    return variants


def ablation_runs(reps: int, **matrix_kwargs: str) -> list[mx.RunSpec]:
    template = mx.build_matrix(**matrix_kwargs)[0]
    runs = []
    for workload in mx.WORKLOADS.values():
        for conc in mx.CONCURRENCY_LEVELS:
            for cfg in ablation_configs():
                is_cap = cfg.name.startswith("cap")
                if is_cap and conc != mx.KEY_CONCURRENCY:
                    continue  # an admission cap only matters once demand exceeds it
                case = dataclasses.replace(template, workload=workload, config=cfg, concurrency=conc)
                runs += [mx.RunSpec(case, rep) for rep in range(reps)]
    return runs


def _positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return value


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--reps", type=_positive_int, default=DEFAULT_REPS)
    p.add_argument("--key-reps", type=_positive_int, default=DEFAULT_KEY_REPS)
    p.add_argument("--no-control", action="store_true", help="skip the A/A control")
    p.add_argument("--ablation", action="store_true")
    p.add_argument("--smoke", action="store_true", help="1 rep, no control: a fast end-to-end check")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--hardware", choices=sorted(kv_sizing.HARDWARE), default=mx.DEFAULT_HARDWARE)
    p.add_argument("--model-preset", choices=sorted(kv_sizing.MODELS), default=mx.DEFAULT_PRESET)
    return p


def _write(name: str, rows: list[dict], meta: dict, out: Path) -> None:
    results_io.write_results(name, rows, meta, out)
    results_io.write_results(f"{name}_summary", stability.aggregate_cases(rows), meta, out)
    logger.info("wrote %s (%d runs)", out / f"{name}.json", len(rows))


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    reps, key_reps = (1, 1) if args.smoke else (args.reps, max(args.key_reps, args.reps))
    kwargs = {"hardware": args.hardware, "model_preset": args.model_preset}
    meta = results_io.run_meta(
        simulated=True, note=SIM_NOTE,
        extra={
            "smoke": args.smoke, "reps": reps, "key_reps": key_reps, **kwargs,
            "hardware_profile": dataclasses.asdict(kv_sizing.HARDWARE[args.hardware]),
        },
    )
    runs = mx.build_runs(reps, key_reps, include_control=not (args.smoke or args.no_control), **kwargs)
    _write("matrix", run_matrix(runs), meta, args.out)
    if args.ablation:
        _write("ablation", run_matrix(ablation_runs(reps, **kwargs)), meta, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
