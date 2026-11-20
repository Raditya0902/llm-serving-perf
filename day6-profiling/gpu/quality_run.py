"""One run of the backend quality check (performance-report 9.9 follow-up; compare with gpu/quality_compare.py).

    python -m gpu.quality_run --backend FLASH_ATTN --bt 2048 --out q/fa2.json
    python -m gpu.quality_run --backend TRITON_ATTN --bt 2048 --out q/triton.json
    python -m gpu.quality_run --backend FLASH_ATTN --bt 8192 --out q/fa2_bt8192.json      # same kernel, other chunking: the noise floor
    python -m gpu.quality_run --reference --out q/reference.json                          # fp32 HF prompt log-probs, short prompts only

Prompts are token windows cut from this folder's own markdown (real English and code, 64-3,000 tokens), identical for every run.
All 200 prompts are submitted at once with a small token budget, so decode steps and prefill chunks share engine steps, the
mixed-batch regime where the two backends differ. Greedy decoding, 64 new tokens, 5 alternative log-probs per position.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import random
import sys
from pathlib import Path

CORPUS_ROOT = Path(__file__).resolve().parents[1]  # day6-profiling: the same files on the Mac and on a pod that only received this folder
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
N_PROMPTS, MIN_LEN, MAX_LEN, NEW_TOKENS, TOP_LOGPROBS = 200, 64, 3000, 64, 5
REFERENCE_PROMPTS, REFERENCE_MAX_LEN = 20, 1024
SEED = 0


def plan_prompts(n: int, seed: int, min_len: int, max_len: int, corpus_len: int) -> list[tuple[int, int]]:
    """(start, length) windows into a token stream; lengths are log-uniform so short and long prompts both occur."""
    if corpus_len < max_len:
        raise ValueError(f"corpus of {corpus_len} tokens is shorter than the longest prompt ({max_len})")
    rng = random.Random(seed)
    plan = []
    for _ in range(n):
        length = int(math.exp(rng.uniform(math.log(min_len), math.log(max_len))))
        plan.append((rng.randrange(0, corpus_len - length + 1), length))
    return plan


def record_from_output(pid: int, out) -> dict:
    prompt_lp = [None if d is None else d[tok].logprob for tok, d in zip(out.prompt_token_ids, out.prompt_logprobs)]
    completion = out.outputs[0]
    chosen = [d[tok].logprob for tok, d in zip(completion.token_ids, completion.logprobs)]
    top = [{str(tok): lp.logprob for tok, lp in d.items()} for d in completion.logprobs]
    return {"id": pid, "n_prompt": len(out.prompt_token_ids), "prompt_logprobs": prompt_lp,
            "tokens": list(completion.token_ids), "logprobs": chosen, "top": top}


def corpus_text() -> str:
    paths = sorted(glob.glob(str(CORPUS_ROOT / "**" / "*.md"), recursive=True))
    paths = [p for p in paths if not any(skip in p for skip in (".venv", "node_modules", "site-packages", "results/raw"))]
    return "\n\n".join(Path(p).read_text(errors="ignore") for p in paths)


def prompt_token_ids(tokenizer) -> list[list[int]]:
    stream = tokenizer.encode(corpus_text())
    return [stream[s:s + n] for s, n in plan_prompts(N_PROMPTS, SEED, MIN_LEN, MAX_LEN, len(stream))]


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model", default=MODEL)
    p.add_argument("--backend", help="vLLM attention backend name, e.g. FLASH_ATTN or TRITON_ATTN")
    p.add_argument("--reference", action="store_true", help="HF float32 prompt log-probs instead of a vLLM run")
    p.add_argument("--fa-version", type=int, choices=[2, 3], default=None, help="force the FlashAttention generation (with --backend FLASH_ATTN)")
    p.add_argument("--bt", type=int, default=2048, help="--max-num-batched-tokens")
    p.add_argument("--seqs", type=int, default=256)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    if bool(args.backend) == bool(args.reference):
        p.error("give exactly one of --backend or --reference")
    return args


def run_vllm(args: argparse.Namespace) -> dict:
    from vllm import LLM, SamplingParams

    backend_args = ({"attention_config": {"backend": args.backend, "flash_attn_version": args.fa_version}} if args.fa_version
                    else {"attention_backend": args.backend})
    llm = LLM(model=args.model, **backend_args, max_num_batched_tokens=args.bt, max_num_seqs=args.seqs,
              max_model_len=4096, gpu_memory_utilization=0.85, seed=SEED, enable_prefix_caching=False)
    prompts = prompt_token_ids(llm.get_tokenizer())
    params = SamplingParams(temperature=0.0, max_tokens=NEW_TOKENS, logprobs=TOP_LOGPROBS, prompt_logprobs=1)
    outputs = llm.generate([{"prompt_token_ids": ids} for ids in prompts], params)
    return {"config": {"model": args.model, "backend": args.backend, "fa_version": args.fa_version, "bt": args.bt, "seqs": args.seqs, "new_tokens": NEW_TOKENS},
            "prompts": [record_from_output(i, o) for i, o in enumerate(outputs)]}


def run_reference(args: argparse.Namespace) -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32).cuda().eval()
    prompts = [(i, ids) for i, ids in enumerate(prompt_token_ids(tokenizer)) if len(ids) <= REFERENCE_MAX_LEN][:REFERENCE_PROMPTS]
    records = []
    with torch.no_grad():
        for pid, ids in prompts:
            logits = model(torch.tensor([ids], device="cuda")).logits[0].float()
            logp = torch.log_softmax(logits[:-1], dim=-1).gather(1, torch.tensor(ids[1:], device="cuda")[:, None])[:, 0]
            records.append({"id": pid, "prompt_logprobs": [None] + logp.tolist()})
    return {"config": {"model": args.model, "dtype": "float32"}, "prompts": records}


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    report = run_reference(args) if args.reference else run_vllm(args)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report))
    print(f"wrote {args.out}: {len(report['prompts'])} prompts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
