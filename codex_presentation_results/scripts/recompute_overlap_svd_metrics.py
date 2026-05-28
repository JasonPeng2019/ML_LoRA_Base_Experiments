from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_pipeline_common():
    repo = _repo_root()
    scripts_dir = repo / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.append(str(scripts_dir))
    from pipeline_common import (  # pylint: disable=import-outside-toplevel
        BASE_MODEL_ID,
        FFT_SST2_MODEL_ID,
        build_qv_deltas,
        common_qv_names,
        load_base_fft_and_lora_merged,
        load_classifier_model,
    )

    return {
        "BASE_MODEL_ID": BASE_MODEL_ID,
        "FFT_SST2_MODEL_ID": FFT_SST2_MODEL_ID,
        "build_qv_deltas": build_qv_deltas,
        "common_qv_names": common_qv_names,
        "load_base_fft_and_lora_merged": load_base_fft_and_lora_merged,
        "load_classifier_model": load_classifier_model,
    }


def subspace_overlap(u_ft: torch.Tensor, u_lora: torch.Tensor, k: int) -> float:
    k_eff = min(k, u_ft.shape[1], u_lora.shape[1])
    if k_eff <= 0:
        return 0.0
    a = u_ft[:, :k_eff]
    b = u_lora[:, :k_eff]
    score = torch.linalg.norm(a.T @ b, ord="fro").pow(2) / float(k_eff)
    return float(score.item())


def cumulative_curve(delta: torch.Tensor, max_k: int) -> np.ndarray:
    singular_vals = torch.linalg.svdvals(delta)
    energies = singular_vals.pow(2)
    total = float(energies.sum().item())
    if total <= 0.0:
        return np.zeros(max_k, dtype=np.float64)

    cum = (torch.cumsum(energies, dim=0) / total).cpu().numpy()
    out = np.ones(max_k, dtype=np.float64)
    n = min(max_k, cum.shape[0])
    out[:n] = cum[:n]
    if n < max_k and n > 0:
        out[n:] = cum[n - 1]
    return out


def main() -> None:
    pc = _load_pipeline_common()
    k_values = [1, 2, 4, 8, 16, 32, 64]
    max_k = 128

    bundle = pc["load_base_fft_and_lora_merged"]()

    base_sd = bundle.base.state_dict()
    fft_sd = bundle.fft_sst2.state_dict()
    lora_sd = bundle.lora_merged.state_dict()

    names = pc["common_qv_names"](base_sd, fft_sd, lora_sd)
    ft_deltas = pc["build_qv_deltas"](base_sd, fft_sd, names)
    lora_deltas = pc["build_qv_deltas"](base_sd, lora_sd, names)

    common_names = sorted(set(ft_deltas.keys()) & set(lora_deltas.keys()))
    by_k = {k: [] for k in k_values}
    for name in common_names:
        u_ft, _, _ = torch.linalg.svd(ft_deltas[name], full_matrices=False)
        u_lora, _, _ = torch.linalg.svd(lora_deltas[name], full_matrices=False)
        for k in k_values:
            by_k[k].append(subspace_overlap(u_ft, u_lora, k))

    base = pc["load_classifier_model"](pc["BASE_MODEL_ID"])
    fft = pc["load_classifier_model"](pc["FFT_SST2_MODEL_ID"])

    names_svd = pc["common_qv_names"](base.state_dict(), fft.state_dict())
    deltas = pc["build_qv_deltas"](base.state_dict(), fft.state_dict(), names_svd)
    curves = [cumulative_curve(delta, max_k=max_k) for delta in deltas.values()]
    stacked = np.stack(curves, axis=0)
    mean_curve = stacked.mean(axis=0)

    thresholds = [0.8, 0.9, 0.95, 0.99]
    k_for_threshold = {}
    for threshold in thresholds:
        idx = np.where(mean_curve >= threshold)[0]
        k_for_threshold[str(threshold)] = int(idx[0] + 1 if len(idx) > 0 else max_k)

    results = {
        "overlap": {
            "k_values": k_values,
            "num_matrices": len(common_names),
            "mean": {str(k): float(np.mean(by_k[k])) for k in k_values},
            "std": {str(k): float(np.std(by_k[k])) for k in k_values},
            "min": {str(k): float(np.min(by_k[k])) for k in k_values},
            "max": {str(k): float(np.max(by_k[k])) for k in k_values},
        },
        "svd": {
            "num_matrices": len(curves),
            "max_k": max_k,
            "mean_energy_at_k": {
                str(k): float(mean_curve[k - 1]) for k in [1, 2, 4, 8, 16, 32, 64, 128]
            },
            "k_for_threshold": k_for_threshold,
        },
    }

    out_path = _repo_root() / "codex_presentation_results" / "data" / "overlap_svd_metrics.json"
    out_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
