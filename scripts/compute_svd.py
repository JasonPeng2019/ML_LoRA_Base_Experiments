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
    load_classifier_model,
    BASE_MODEL_ID,
    FFT_SST2_MODEL_ID,
)

MAX_K = 128


def cumulative_energy_from_delta(delta: torch.Tensor, max_k: int = MAX_K) -> np.ndarray:
    singular_vals = torch.linalg.svdvals(delta)
    energies = singular_vals.pow(2)
    total = float(energies.sum().item())
    if total <= 0.0:
        return np.zeros(max_k, dtype=np.float64)

    cum = torch.cumsum(energies, dim=0) / total
    cum_np = cum.cpu().numpy()

    out = np.ones(max_k, dtype=np.float64)
    n = min(max_k, cum_np.shape[0])
    out[:n] = cum_np[:n]
    if n < max_k:
        out[n:] = cum_np[n - 1]
    return out


def main() -> None:
    ensure_dirs()
    get_device()

    base = load_classifier_model(BASE_MODEL_ID)
    fft = load_classifier_model(FFT_SST2_MODEL_ID)

    base_sd = base.state_dict()
    fft_sd = fft.state_dict()

    names = common_qv_names(base_sd, fft_sd)
    deltas = build_qv_deltas(base_sd, fft_sd, names)

    if not deltas:
        raise RuntimeError("No comparable query/value matrices found for SVD.")

    print(f"Comparable q/v matrices: {len(deltas)}")

    all_curves = []
    for name, delta in deltas.items():
        curve = cumulative_energy_from_delta(delta, max_k=MAX_K)
        all_curves.append(curve)
        print(f"Computed SVD curve for {name}")

    stacked = np.stack(all_curves, axis=0)
    mean_curve = stacked.mean(axis=0)

    # Minimal acceptance sanity: cumulative energy should be monotonic non-decreasing.
    if np.any(np.diff(mean_curve) < -1e-8):
        raise RuntimeError("Mean cumulative energy curve is not monotonic.")

    x = np.arange(1, MAX_K + 1)
    plt.figure(figsize=(8.5, 5.2))
    for curve in stacked:
        plt.plot(x, curve, alpha=0.15, linewidth=1.0, color="#5B8FF9")
    plt.plot(x, mean_curve, linewidth=2.5, color="#1D39C4", label="mean over q/v matrices")

    plt.title("SST-2 Full-FT Delta: Cumulative Spectral Energy")
    plt.xlabel("Top-k singular directions")
    plt.ylabel("Cumulative energy")
    plt.ylim(0.0, 1.02)
    plt.xlim(1, MAX_K)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()

    out_path = Path("figures/sst2_svd_energy.png")
    plt.savefig(out_path, dpi=180)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
