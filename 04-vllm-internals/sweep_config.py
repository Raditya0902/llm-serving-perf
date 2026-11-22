"""One-factor-at-a-time sweep definition for ``vllm serve`` + ``vllm bench serve``.

The sweep makes configuration comparisons that change **one serving
parameter at a time** while holding model, workload and hardware fixed.  This
module encodes that rule so it can be checked mechanically: ``build_sweep``
returns a ``baseline`` run followed by runs that each differ from it in
exactly one field (asserted in the tests).

Flag names and the ``--metadata`` mechanism were checked against the vLLM
v0.34.0 source (``vllm/benchmarks/serve.py``, ``vllm/config/scheduler.py``).
Every result JSON carries ``sweep_*`` and ``p_*`` metadata keys so it is
self-describing; ``parse_results.py`` relies on them.

Nothing here talks to a GPU, so it is fully testable on any machine.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import Literal, Sequence

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
BASELINE_RATE_BRIEF = 2.0
LOAD_PROFILES = ("brief", "saturated")
MAX_PORT = 65535
PERCENTILE_METRICS = "ttft,tpot,itl,e2el"
METRIC_PERCENTILES = "50,99"
LOOPBACK = "127.0.0.1"  # the sweep never needs the server reachable from the network
CONFIG_ID_LENGTH = 12
# Hub ids ("org/name"), local paths, no whitespace/shell metacharacters; must not
# start with "-" because the model is a positional argument to `vllm serve`.
MODEL_RE = re.compile(r"^(?!-)[A-Za-z0-9_.~/:\-]+$")


def format_value(value: float | int) -> str:
    """Short, filename-safe rendering used in run ids (``2.0`` -> ``2``)."""
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


@dataclass(frozen=True)
class ServeParams:
    """Arguments of ``vllm serve`` that the sweep controls."""

    model: str = DEFAULT_MODEL
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.85
    # Pinned explicitly (not left at vLLM's hardware-dependent defaults) so the
    # baseline is reproducible across hosts.
    max_num_seqs: int = 256
    max_num_batched_tokens: int = 8192
    port: int = 8000
    seed: int = 0

    def __post_init__(self) -> None:
        if not MODEL_RE.fullmatch(self.model):
            raise ValueError(
                f"model {self.model!r} must be a Hub id or path without whitespace, "
                "shell characters or a leading '-'"
            )
        if self.max_model_len < 1:
            raise ValueError("max_model_len must be >= 1")
        if not 0.0 < self.gpu_memory_utilization <= 1.0:
            raise ValueError("gpu_memory_utilization must be in (0, 1]")
        if self.max_num_seqs < 1:
            raise ValueError("max_num_seqs must be >= 1")
        if self.max_num_batched_tokens < 1:
            raise ValueError("max_num_batched_tokens must be >= 1")
        # vllm/config/scheduler.py: verify_max_model_len() rejects this.
        if self.max_num_batched_tokens < self.max_num_seqs:
            raise ValueError(
                "max_num_batched_tokens must be >= max_num_seqs "
                "(vLLM SchedulerConfig rejects the opposite)"
            )
        if not 1 <= self.port <= MAX_PORT:
            raise ValueError("port must be in 1..65535")


@dataclass(frozen=True)
class BenchParams:
    """Arguments of ``vllm bench serve`` that the sweep controls."""

    request_rate: float = BASELINE_RATE_BRIEF  # math.inf == send all at once
    input_len: int = 512
    output_len: int = 128
    num_prompts: int = 100
    port: int = 8000
    seed: int = 0
    ignore_eos: bool = True

    def __post_init__(self) -> None:
        if not self.request_rate > 0:
            raise ValueError("request_rate must be > 0 (use math.inf for no pacing)")
        if self.input_len < 1 or self.output_len < 1:
            raise ValueError("input_len and output_len must be >= 1")
        if self.num_prompts < 1:
            raise ValueError("num_prompts must be >= 1")
        if not 1 <= self.port <= MAX_PORT:
            raise ValueError("port must be in 1..65535")


@dataclass(frozen=True)
class SweepRun:
    """A fully specified serve + bench pair."""

    run_id: str
    knob: str
    value: str
    serve: ServeParams
    bench: BenchParams

    def __post_init__(self) -> None:
        if self.bench.input_len + self.bench.output_len > self.serve.max_model_len:
            raise ValueError(
                f"input_len + output_len ({self.bench.input_len} + "
                f"{self.bench.output_len}) exceeds max_model_len "
                f"({self.serve.max_model_len})"
            )
        if self.serve.port != self.bench.port:
            raise ValueError("serve and bench must target the same port")

    @property
    def base_url(self) -> str:
        return f"http://{LOOPBACK}:{self.serve.port}"

    @property
    def config_id(self) -> str:
        """Fingerprint of everything that defines this run's experiment (port excluded).

        Stored in each result file so a resumed or aggregated run can prove it was
        produced by the same configuration (model, load, prompts, seed, knobs).
        """
        payload = {
            "knob": self.knob, "value": self.value,
            "serve": {k: v for k, v in dataclasses.asdict(self.serve).items() if k != "port"},
            "bench": {k: v for k, v in dataclasses.asdict(self.bench).items() if k != "port"},
        }
        blob = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()[:CONFIG_ID_LENGTH]

    def serve_command(self) -> list[str]:
        s = self.serve
        return [
            "vllm", "serve", s.model,
            "--max-model-len", str(s.max_model_len),
            "--gpu-memory-utilization", str(s.gpu_memory_utilization),
            "--max-num-seqs", str(s.max_num_seqs),
            "--max-num-batched-tokens", str(s.max_num_batched_tokens),
            "--host", LOOPBACK,
            "--port", str(s.port),
            "--seed", str(s.seed),
        ]

    def bench_command(self, result_dir: str, rep: int) -> list[str]:
        b, s = self.bench, self.serve
        cmd = [
            "vllm", "bench", "serve",
            "--backend", "vllm",
            "--model", s.model,
            "--endpoint", "/v1/completions",
            "--host", LOOPBACK,
            "--port", str(b.port),
            "--dataset-name", "random",
            "--random-input-len", str(b.input_len),
            "--random-output-len", str(b.output_len),
            "--num-prompts", str(b.num_prompts),
            "--request-rate", str(float(b.request_rate)),
            "--seed", str(b.seed),
            "--percentile-metrics", PERCENTILE_METRICS,
            "--metric-percentiles", METRIC_PERCENTILES,
            "--save-result",
            "--result-dir", result_dir,
            "--result-filename", f"{self.run_id}__r{rep}.json",
        ]
        if b.ignore_eos:
            cmd.append("--ignore-eos")
        # --metadata is nargs="*", so it must come last.
        cmd.append("--metadata")
        cmd.extend(self._metadata())
        return cmd

    def _metadata(self) -> list[str]:
        s, b = self.serve, self.bench
        return [
            f"sweep_run_id={self.run_id}",
            f"sweep_knob={self.knob}",
            f"sweep_value={self.value}",
            f"sweep_config_id={self.config_id}",
            f"p_num_prompts={b.num_prompts}",
            f"p_seed={b.seed}",
            f"p_ignore_eos={b.ignore_eos}",
            f"p_max_num_seqs={s.max_num_seqs}",
            f"p_max_num_batched_tokens={s.max_num_batched_tokens}",
            f"p_gpu_memory_utilization={s.gpu_memory_utilization}",
            f"p_max_model_len={s.max_model_len}",
            f"p_request_rate={float(b.request_rate)}",
            f"p_input_len={b.input_len}",
            f"p_output_len={b.output_len}",
        ]


@dataclass(frozen=True)
class KnobSpec:
    target: Literal["serve", "bench"]
    values: tuple[float | int, ...]


SWEEP_KNOBS: dict[str, KnobSpec] = {
    "max_num_seqs": KnobSpec("serve", (4, 16, 64, 256, 1024)),
    "max_num_batched_tokens": KnobSpec("serve", (512, 1024, 2048, 4096, 8192, 16384)),
    "gpu_memory_utilization": KnobSpec("serve", (0.5, 0.7, 0.85, 0.95)),
    "request_rate": KnobSpec("bench", (1.0, 2.0, 4.0, 8.0, 16.0, math.inf)),
    "input_len": KnobSpec("bench", (128, 512, 1024, 2048, 3584)),
}


def _active_knobs(load: str) -> dict[str, KnobSpec]:
    """Under saturated load the request rate is fixed at inf, not swept."""
    if load == "saturated":
        return {k: v for k, v in SWEEP_KNOBS.items() if k != "request_rate"}
    return dict(SWEEP_KNOBS)


def build_sweep(
    *,
    model: str = DEFAULT_MODEL,
    load: str = "brief",
    only: Sequence[str] | None = None,
    port: int = 8000,
    seed: int = 0,
    num_prompts: int = 100,
    ignore_eos: bool = True,
) -> list[SweepRun]:
    """Return ``[baseline, *one-knob-at-a-time runs]``.

    ``load="brief"`` is the paced workload from the task (request rate 2/s).
    ``load="saturated"`` sends every request at once (rate = inf): scheduling
    limits such as ``max_num_seqs`` only bite when demand exceeds them, which
    a 2 req/s workload on a 0.5B model never does.
    """
    if load not in LOAD_PROFILES:
        raise ValueError(f"load must be one of {LOAD_PROFILES}, got {load!r}")
    knobs = _active_knobs(load)
    unknown = sorted(set(only or ()) - set(knobs))
    if unknown:
        raise ValueError(f"unknown knob(s) {unknown}; valid: {sorted(knobs)}")

    base_rate = math.inf if load == "saturated" else BASELINE_RATE_BRIEF
    serve = ServeParams(model=model, port=port, seed=seed)
    bench = BenchParams(
        request_rate=base_rate, num_prompts=num_prompts, port=port, seed=seed,
        ignore_eos=ignore_eos,
    )
    runs = [SweepRun("baseline", "baseline", "baseline", serve, bench)]

    for knob, spec in knobs.items():
        if only and knob not in only:
            continue
        current = getattr(serve if spec.target == "serve" else bench, knob)
        for value in spec.values:
            if value == current:
                continue  # that is the baseline run
            if spec.target == "serve":
                run_serve, run_bench = dataclasses.replace(serve, **{knob: value}), bench
            else:
                run_serve, run_bench = serve, dataclasses.replace(bench, **{knob: value})
            text = format_value(value)
            runs.append(SweepRun(f"{knob}-{text}", knob, text, run_serve, run_bench))
    return runs
