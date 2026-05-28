# Slide Snippets: LoRA 3-Hour Geometry + Oracle Routing Demo

## Motivation
LoRA reaches competitive task performance with far fewer trainable parameters. We ask: what update geometry is learned, and can task-routed experts make one base-family system cover many tasks?

## Setup
We use RoBERTa-base with existing full fine-tuned and LoRA checkpoints from Hugging Face. This is a measured engineering demo, not a publication-grade multi-seed benchmark.

## Figure 1: Full-FT Spectral Concentration
`figures/sst2_svd_energy.png` shows cumulative spectral energy of full-FT deltas on attention query/value matrices.

## Figure 2: LoRA vs Full-FT Subspace Overlap
`figures/sst2_lora_ft_overlap.png` shows overlap `phi(k)` between top-k update subspaces.

## Figure 3: Oracle-Routed Multi-Task Results
`figures/oracle_routing_per_task.png` compares:
- oracle-routed LoRA experts,
- per-task LoRA standalone,
- per-task full fine-tuned baselines.

## Tables
- `tables/oracle_routing_results.tex`: measured per-task routing results.
- `tables/model_card_metrics.tex`: model-card-reported reference values.

## Caveats
- Oracle routing uses known task IDs (not a learned router).
- Results are validation-split measurements for selected tasks.
- This workflow prioritizes clear end-to-end evidence over exhaustive benchmarking.
