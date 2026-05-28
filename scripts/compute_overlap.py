from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from pipeline_common import (
    build_qv_deltas,
    common_qv_names,
    ensure_dirs,
    get_device,
    load_base_fft_and_lora_merged,
)

K_VALUES = [1, 2, 4, 8, 16, 32, 64]


def subspace_overlap(u_ft: torch.Tensor, u_lora: torch.Tensor, k: int) -> float:
    k_eff = min(k, u_ft.shape[1], u_lora.shape[1])
    if k_eff <= 0:
        return 0.0
    a = u_ft[:, :k_eff]
    b = u_lora[:, :k_eff]
    score = torch.linalg.norm(a.T @ b, ord="fro").pow(2) / float(k_eff)
    return float(score.item())


def main() -> None:
    ensure_dirs()
    get_device()

    bundle = load_base_fft_and_lora_merged()

    base_sd = bundle.base.state_dict()
    fft_sd = bundle.fft_sst2.state_dict()
    lora_sd = bundle.lora_merged.state_dict()

    names = common_qv_names(base_sd, fft_sd, lora_sd)
    ft_deltas = build_qv_deltas(base_sd, fft_sd, names)
    lora_deltas = build_qv_deltas(base_sd, lora_sd, names)

    common_names = sorted(list(set(ft_deltas.keys()) & set(lora_deltas.keys())))
    if not common_names:
        raise RuntimeError("No comparable q/v matrices found for overlap computation.")

    print(f"Comparable q/v matrices for overlap: {len(common_names)}")

    by_k = {k: [] for k in K_VALUES}
    for name in common_names:
        ft_delta = ft_deltas[name]
        lora_delta = lora_deltas[name]

        u_ft, _, _ = torch.linalg.svd(ft_delta, full_matrices=False)
        u_lora, _, _ = torch.linalg.svd(lora_delta, full_matrices=False)

        for k in K_VALUES:
            by_k[k].append(subspace_overlap(u_ft, u_lora, k))

    means = np.array([np.mean(by_k[k]) for k in K_VALUES], dtype=np.float64)
    stds = np.array([np.std(by_k[k]) for k in K_VALUES], dtype=np.float64)

    plt.figure(figsize=(8.2, 5.0))
    plt.plot(K_VALUES, means, marker="o", linewidth=2.3, color="#389E0D", label="mean overlap")
    plt.fill_between(K_VALUES, np.clip(means - stds, 0.0, None), np.clip(means + stds, 0.0, 1.0), alpha=0.2, color="#389E0D", label="±1 std")

    plt.title("LoRA vs Full-FT Update Subspace Overlap (SST-2)")
    plt.xlabel("Top-k subspace dimension")
    plt.ylabel("Overlap $\\phi(k)$")
    plt.ylim(0.0, 1.02)
    plt.xscale("log", base=2)
    plt.xticks(K_VALUES, [str(k) for k in K_VALUES])
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()

    out_path = Path("figures/sst2_lora_ft_overlap.png")
    plt.savefig(out_path, dpi=180)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
