# LoRA 3-Hour Geometry Demo

This repository now runs three experiments end-to-end:
- full-FT SVD geometry on SST-2,
- LoRA-vs-full-FT overlap on SST-2,
- measured oracle-routed multi-task LoRA evaluation across six GLUE tasks.

## Environment Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If CUDA is unavailable or unstable:

```bash
python -m pip uninstall -y torch
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.5.1
python -m pip install -r requirements.txt --no-deps
```

## Run All Experiments

```bash
python scripts/run_all_experiments.py
```

## Run Individual Experiment 3 (Oracle Routing)

```bash
python scripts/oracle_routing_eval.py \
  --tasks sst2,mrpc,rte,mnli,qnli,qqp \
  --max-examples-per-task 2000 \
  --batch-size 0 \
  --device auto
```

- `--batch-size 0` enables auto batch sizing.
- `--device auto` prefers GPU and falls back to CPU.

## Main Outputs

- `figures/sst2_svd_energy.png`
- `figures/sst2_lora_ft_overlap.png`
- `figures/oracle_routing_per_task.png`
- `tables/oracle_routing_results.tex`
- `outputs/oracle_routing_metrics.json`
- `presentation_snippets.md`

Backward-compatible router outputs are also written:
- `figures/router_demo.png`
- `tables/router_demo_metrics.tex`

## Runtime Expectations

On a 24GB-class GPU, the oracle-routing evaluation with 2,000 examples per task (6 tasks) is typically well within the 3-hour total project budget, including model loads and baseline passes.

## Caveats

- Oracle routing uses known task IDs, not a learned router.
- This is benchmark-lite and presentation-oriented.
- Model-card tables contain reported values unless explicitly recomputed.
