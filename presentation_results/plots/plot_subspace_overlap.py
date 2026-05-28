"""
Subspace overlap plot: phi(k) for LoRA checkpoint delta vs Full-FT delta (SST-2).
Two modes:
  1. RECOMPUTE=True  -- loads pipeline and recomputes exact values (requires running from scripts/ dir)
  2. RECOMPUTE=False -- uses approximate values read from the existing figure (default)

Saves to presentation_results/plots/subspace_overlap.png
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

RECOMPUTE = False  # Set True to recompute from model weights (loads ~1GB of models)

K_VALUES = [1, 2, 4, 8, 16, 32, 64]
# Exact values from codex_presentation_results/data/overlap_svd_metrics.json
MEAN_APPROX = [0.04717, 0.05456, 0.06252, 0.08469, 0.10911, 0.10307, 0.12275]
STD_APPROX  = [0.05891, 0.05105, 0.03732, 0.04083, 0.04152, 0.02431, 0.01275]

D = 768  # RoBERTa-base hidden dimension


def random_baseline(k):
    return k / D


def recompute_overlap():
    """Loads pipeline_common and recomputes overlap values. Must run from scripts/ dir."""
    scripts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "..", "..", "scripts")
    sys.path.insert(0, os.path.abspath(scripts_dir))

    import torch
    from pipeline_common import (
        build_qv_deltas, common_qv_names, load_base_fft_and_lora_merged,
    )

    bundle = load_base_fft_and_lora_merged()
    base_sd = bundle.base.state_dict()
    fft_sd  = bundle.fft_sst2.state_dict()
    lora_sd = bundle.lora_merged.state_dict()

    names = common_qv_names(base_sd, fft_sd, lora_sd)
    ft_deltas   = build_qv_deltas(base_sd, fft_sd, names)
    lora_deltas = build_qv_deltas(base_sd, lora_sd, names)

    common = sorted(set(ft_deltas) & set(lora_deltas))

    def phi(u_ft, u_lora, k):
        k_eff = min(k, u_ft.shape[1], u_lora.shape[1])
        a, b = u_ft[:, :k_eff], u_lora[:, :k_eff]
        return float(torch.linalg.norm(a.T @ b, ord="fro").pow(2) / k_eff)

    by_k = {k: [] for k in K_VALUES}
    for name in common:
        u_ft,  _, _ = torch.linalg.svd(ft_deltas[name],   full_matrices=False)
        u_lora,_, _ = torch.linalg.svd(lora_deltas[name], full_matrices=False)
        for k in K_VALUES:
            by_k[k].append(phi(u_ft, u_lora, k))

    means = np.array([np.mean(by_k[k]) for k in K_VALUES])
    stds  = np.array([np.std(by_k[k])  for k in K_VALUES])
    return means, stds


def main():
    if RECOMPUTE:
        print("Recomputing overlap from model weights...")
        means, stds = recompute_overlap()
    else:
        means = np.array(MEAN_APPROX)
        stds  = np.array(STD_APPROX)

    random = np.array([random_baseline(k) for k in K_VALUES])

    fig, ax = plt.subplots(figsize=(8.5, 5.2))

    ax.plot(K_VALUES, means, marker="o", linewidth=2.3, color="#389E0D",
            label=r"Mean $\phi(k)$ (LoRA vs Full-FT)", zorder=3)
    ax.fill_between(K_VALUES,
                    np.clip(means - stds, 0, None),
                    np.clip(means + stds, 0, 1),
                    alpha=0.2, color="#389E0D", label=r"$\pm$1 std")
    ax.plot(K_VALUES, random, marker="s", linewidth=1.5, linestyle="--",
            color="#AAAAAA", label=r"Random baseline ($k/d$, $d=768$)", zorder=2)

    ax.set_title(r"LoRA vs Full-FT Update Subspace Overlap (SST-2, RoBERTa-base)" + "\n"
                 r"$\phi(k) = \|U_{FT,1:k}^{T}\, U_{LoRA,1:k}\|_F^2\,/\,k$",
                 fontsize=11)
    ax.set_xlabel("Top-$k$ subspace dimension", fontsize=11)
    ax.set_ylabel(r"Overlap $\phi(k)$", fontsize=11)
    ax.set_ylim(0.0, 0.30)
    ax.set_xscale("log", base=2)
    ax.set_xticks(K_VALUES)
    ax.set_xticklabels([str(k) for k in K_VALUES])
    ax.grid(alpha=0.25)
    ax.legend(fontsize=10)

    # Annotate ratio at k=1
    ratio1 = means[0] / random[0]
    ax.annotate(f"k=1: {ratio1:.0f}× random",
                xy=(K_VALUES[0], means[0]),
                xytext=(2.2, means[0] + 0.04),
                fontsize=9, color="#389E0D",
                arrowprops=dict(arrowstyle="->", color="#389E0D", lw=1.1))

    plt.tight_layout()

    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(out_dir, "subspace_overlap.png")
    plt.savefig(out_path, dpi=180)
    print(f"Saved: {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
