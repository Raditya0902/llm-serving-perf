# Session 3 results

**No GPU run yet.** This repository was built on a Mac without an NVIDIA GPU, so `vllm serve` and
`vllm bench serve` have not been run and there are **no measurements** here. Any numbers you
see in `tests/` are synthetic fixtures for exercising the parser and plots and mean nothing.

To produce real results, follow [RUNBOOK.md](RUNBOOK.md) on a Linux CUDA host, then:

```bash
python parse_results.py --raw results/raw/<tag> --out results/<tag>
python make_report.py   --summary results/<tag>/sweep_summary.json --env results/env.txt --out results/<tag>/RESULTS.md
```

and replace this file with (or link to) the generated report(s).
