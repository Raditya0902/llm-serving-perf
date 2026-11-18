"""The Day 7 benchmark matrix: 2 workloads x 3 concurrency levels x 2 serving configs = 12 cases.

Fixed BEFORE any run (see README "Experiment design"):

* Workload A, interactive chat : ~256 in / ~128 out, TTFT-sensitive, never KV-bound.
* Workload B, long context     : ~2560 in / ~384 out, high KV pressure at concurrency 64.
* Concurrency 4 / 16 / 64, closed loop (vLLM ``--max-concurrency N --request-rate inf``).
* ``baseline`` is Day 4's baseline. ``tuned`` changes exactly two knobs
  (``gpu_memory_utilization`` and ``max_num_batched_tokens``); ``max_num_seqs`` is equal
  in both and above every concurrency level, so it cannot bind (a Day 4 finding).

The same ``MatrixCase`` drives both the real vLLM runner (commands below) and the
simulated matrix (``run_matrix_sim.py``), so the two sources describe identical experiments.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass

import _prior  # noqa: F401  (puts Days 2-5 on sys.path)

# isort: split
from sweep_config import BenchParams, ServeParams, SweepRun

import kv_sizing

MAX_MODEL_LEN = 4096
CONCURRENCY_LEVELS: tuple[int, ...] = (4, 16, 64)
MIN_PROMPTS = 100
PROMPTS_PER_SLOT = 8
BASE_SEED = 0
CONTROL_SEED_OFFSET = 1000  # A/A control draws different prompts: a conservative noise floor
CONTROL_SUFFIX = "-aa"
DEFAULT_HUB_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_PRESET = "llama3-8b"
DEFAULT_HARDWARE = "a100-40gb"
PERCENTILE_METRICS = "ttft,tpot,itl,e2el"
METRIC_PERCENTILES = "50,95,99"
CONTROL_CASE = ("B", 16, "baseline")  # the A/A control repeats this case
KEY_CONCURRENCY = max(CONCURRENCY_LEVELS)  # cases that feed the optimization decision
CONFIG_ID_LENGTH = 12


@dataclass(frozen=True)
class WorkloadSpec:
    key: str
    label: str
    input_len: int
    output_len: int
    range_ratio: float  # vLLM --random-range-ratio: lengths ~ U[len(1-r), len(1+r)]
    description: str

    def __post_init__(self) -> None:
        if not 0.0 <= self.range_ratio < 1.0:
            raise ValueError("range_ratio must be in [0, 1)")
        if self.max_input_len + self.max_output_len > MAX_MODEL_LEN:
            raise ValueError(
                f"workload {self.key}: worst case {self.max_input_len} + {self.max_output_len} "
                f"exceeds max_model_len {MAX_MODEL_LEN}"
            )

    @property
    def max_input_len(self) -> int:
        return int(self.input_len * (1 + self.range_ratio))

    @property
    def max_output_len(self) -> int:
        return int(self.output_len * (1 + self.range_ratio))


@dataclass(frozen=True)
class ServingConfig:
    name: str
    gpu_memory_utilization: float
    max_num_batched_tokens: int
    max_num_seqs: int = 256
    extra_serve_args: tuple[str, ...] = ()  # appended verbatim to `vllm serve` (e.g. tensor parallelism)


WORKLOADS: dict[str, WorkloadSpec] = {
    "A": WorkloadSpec("A", "interactive chat", 256, 128, 0.25, "short prompts, short outputs, TTFT-sensitive"),
    "B": WorkloadSpec("B", "long-context generation", 2560, 384, 0.25, "long prompts, long outputs, KV pressure"),
}
CONFIGS: dict[str, ServingConfig] = {
    "baseline": ServingConfig("baseline", gpu_memory_utilization=0.85, max_num_batched_tokens=8192),
    "tuned": ServingConfig("tuned", gpu_memory_utilization=0.92, max_num_batched_tokens=2048),
}


@dataclass(frozen=True)
class MatrixCase:
    workload: WorkloadSpec
    config: ServingConfig
    concurrency: int
    model: str = DEFAULT_HUB_MODEL  # Hub id passed to `vllm serve`
    model_preset: str = DEFAULT_PRESET  # Day 2 preset used for sizing and the simulator
    hardware: str = DEFAULT_HARDWARE
    # Run on a larger GPU but pin the KV pool to what `hardware` would have, so the modeled
    # memory-pressure regime can be reproduced (vLLM then ignores gpu_memory_utilization for sizing).
    cap_kv_pool: bool = False
    is_control: bool = False

    def __post_init__(self) -> None:
        ServeParams(model=self.model)  # reuses Day 4's model-id validation
        kv_sizing.model_spec(self.model_preset)
        if self.hardware not in kv_sizing.HARDWARE:
            raise ValueError(f"unknown hardware {self.hardware!r}; valid: {sorted(kv_sizing.HARDWARE)}")
        if self.concurrency < 1:
            raise ValueError("concurrency must be >= 1")
        if self.cap_kv_pool and self._modeled_pool_blocks() <= 0:
            raise ValueError(
                f"cannot cap the KV pool: model {self.model_preset!r} leaves an empty KV pool on {self.hardware!r}"
            )

    def _modeled_pool_blocks(self) -> int:
        return kv_sizing.kv_pool_blocks(
            kv_sizing.model_spec(self.model_preset),
            kv_sizing.HARDWARE[self.hardware],
            self.config.gpu_memory_utilization,
        )

    @property
    def kv_cache_cap_bytes(self) -> int | None:
        """Bytes for `--kv-cache-memory-bytes`, or None when the pool is left to vLLM."""
        if not self.cap_kv_pool:
            return None
        bytes_per_block = kv_sizing.BLOCK_SIZE * kv_sizing.kv_bytes_per_token_for(
            kv_sizing.model_spec(self.model_preset)
        )
        return self._modeled_pool_blocks() * bytes_per_block

    @property
    def case_id(self) -> str:
        base = f"{self.workload.key}-c{self.concurrency}-{self.config.name}"
        return base + CONTROL_SUFFIX if self.is_control else base

    @property
    def num_prompts(self) -> int:
        return max(MIN_PROMPTS, PROMPTS_PER_SLOT * self.concurrency)

    @property
    def config_id(self) -> str:
        """Fingerprint of the experiment itself; identical for a case and its A/A control."""
        payload = dataclasses.asdict(self)
        payload.pop("is_control")
        if not self.cap_kv_pool:
            payload.pop("cap_kv_pool")  # keeps fingerprints of uncapped runs unchanged
        blob = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()[:CONFIG_ID_LENGTH]

    def seed_for(self, rep: int) -> int:
        return BASE_SEED + rep + (CONTROL_SEED_OFFSET if self.is_control else 0)

    def serve_command(self, port: int) -> list[str]:
        """`vllm serve` argv, built by Day 4's `SweepRun` so both days share one definition."""
        cfg = self.config
        serve = ServeParams(
            model=self.model, max_model_len=MAX_MODEL_LEN,
            gpu_memory_utilization=cfg.gpu_memory_utilization,
            max_num_seqs=cfg.max_num_seqs,
            max_num_batched_tokens=cfg.max_num_batched_tokens, port=port, seed=BASE_SEED,
        )
        bench = BenchParams(input_len=1, output_len=1, port=port)  # unused; satisfies SweepRun
        cmd = SweepRun(self.case_id, "matrix", self.case_id, serve, bench).serve_command()
        if self.kv_cache_cap_bytes is not None:
            cmd = [*cmd, "--kv-cache-memory-bytes", str(self.kv_cache_cap_bytes)]
        return [*cmd, *cfg.extra_serve_args]

    def bench_command(self, result_dir: str, rep: int, port: int) -> list[str]:
        w = self.workload
        cmd = [
            "vllm", "bench", "serve",
            "--backend", "vllm",
            "--model", self.model,
            "--endpoint", "/v1/completions",
            "--host", "127.0.0.1",
            "--port", str(port),
            "--dataset-name", "random",
            "--random-input-len", str(w.input_len),
            "--random-output-len", str(w.output_len),
            "--random-range-ratio", str(w.range_ratio),
            "--num-prompts", str(self.num_prompts),
            "--num-warmups", str(self.concurrency),
            "--request-rate", "inf",
            "--max-concurrency", str(self.concurrency),
            "--seed", str(self.seed_for(rep)),
            "--ignore-eos",
            "--percentile-metrics", PERCENTILE_METRICS,
            "--metric-percentiles", METRIC_PERCENTILES,
            "--save-result", "--save-detailed",
            "--result-dir", result_dir,
            "--result-filename", f"{self.case_id}__r{rep}.json",
            "--metadata",  # nargs="*": must be last
        ]
        return cmd + self.metadata(rep)

    def metadata(self, rep: int) -> list[str]:
        return [
            f"matrix_case_id={self.case_id}",
            f"matrix_config_id={self.config_id}",
            f"matrix_workload={self.workload.key}",
            f"matrix_config={self.config.name}",
            f"matrix_concurrency={self.concurrency}",
            f"matrix_rep={rep}",
            f"matrix_control={self.is_control}",
        ]

    def predicted_pressure(self) -> dict[str, float | int | bool]:
        """Analytic KV-capacity check (Day 2 arithmetic), used to cross-check the simulator."""
        return kv_sizing.capacity_pressure(
            kv_sizing.model_spec(self.model_preset),
            kv_sizing.HARDWARE[self.hardware],
            self.config.gpu_memory_utilization,
            concurrency=self.concurrency,
            mean_input=self.workload.input_len,
            mean_output=self.workload.output_len,
        )


@dataclass(frozen=True)
class RunSpec:
    case: MatrixCase
    rep: int

    @property
    def run_id(self) -> str:
        return f"{self.case.case_id}__r{self.rep}"


def build_matrix(
    *, model: str = DEFAULT_HUB_MODEL, model_preset: str = DEFAULT_PRESET,
    hardware: str = DEFAULT_HARDWARE, cap_kv_pool: bool = False,
) -> list[MatrixCase]:
    return [
        MatrixCase(w, cfg, conc, model, model_preset, hardware, cap_kv_pool)
        for w in WORKLOADS.values()
        for conc in CONCURRENCY_LEVELS
        for cfg in CONFIGS.values()
    ]


def build_runs(
    reps: int, key_reps: int, *, include_control: bool = True, **matrix_kwargs: str | bool,
) -> list[RunSpec]:
    """Every (case, rep) to execute: `reps` each, `key_reps` for the highest-concurrency cases
    that feed the optimization decision, plus an A/A control (a repeat of one baseline case)."""
    if reps < 1:
        raise ValueError("reps must be >= 1")
    if key_reps < reps:
        raise ValueError("key_reps must be >= reps")
    cases = build_matrix(**matrix_kwargs)
    runs = [
        RunSpec(c, r)
        for c in cases
        for r in range(key_reps if c.concurrency == KEY_CONCURRENCY else reps)
    ]
    if include_control:
        original = next(
            c for c in cases
            if (c.workload.key, c.concurrency, c.config.name) == CONTROL_CASE
        )
        control = dataclasses.replace(original, is_control=True)
        runs += [RunSpec(control, r) for r in range(key_reps)]
    return runs
