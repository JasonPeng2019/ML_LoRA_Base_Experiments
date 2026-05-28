# codex_presentation_results

Presentation-ready results bundle for:
- subspace overlap,
- per-experiment outcomes,
- oracle-routed LoRA (Mixture-of-Experts style) vs per-task LoRA baseline and per-task Full-FT baselines.

## Contents

- `data/overlap_svd_metrics.json`: measured SST-2 SVD + overlap summary metrics.
- `data/oracle_routing_metrics_snapshot.json`: snapshot of measured oracle routing metrics.
- `tables/*.tex`: ready-to-input LaTeX tables.
- `figures/*.png`: generated snippable charts.
- `presentation_results_all_in_one.tex`: single LaTeX file including all tables and figures.
- `figure_bullets.txt`: bullet points for each figure.
- `speaker_script.txt`: talk track script for presenting results.
- `scripts/recompute_overlap_svd_metrics.py`: recompute overlap/SVD numeric data from checkpoints.
- `scripts/generate_presentation_assets.py`: regenerate all `.tex` tables, figures, and text files from `data/*.json`.

## Regenerate

From repository root:

```bash
.venv/bin/python codex_presentation_results/scripts/recompute_overlap_svd_metrics.py
.venv/bin/python codex_presentation_results/scripts/generate_presentation_assets.py
```

Optional compile:

```bash
cd codex_presentation_results
pdflatex -interaction=nonstopmode -halt-on-error presentation_results_all_in_one.tex
```
