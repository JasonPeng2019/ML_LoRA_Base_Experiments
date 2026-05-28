from __future__ import annotations

import json
from pathlib import Path

from pipeline_common import (
    BASE_MODEL_ID,
    FFT_SST2_MODEL_ID,
    LORA_SST2_MODEL_ID,
    common_qv_names,
    ensure_dirs,
    get_device,
    load_base_fft_and_lora_merged,
    num_params,
    qv_names_from_state_dict,
)


def main() -> None:
    ensure_dirs()
    get_device()

    bundle = load_base_fft_and_lora_merged()

    base_sd = bundle.base.state_dict()
    fft_sd = bundle.fft_sst2.state_dict()
    lora_sd = bundle.lora_merged.state_dict()

    base_qv = qv_names_from_state_dict(base_sd)
    fft_qv = qv_names_from_state_dict(fft_sd)
    lora_qv = qv_names_from_state_dict(lora_sd)

    common_base_fft = common_qv_names(base_sd, fft_sd)
    common_all = common_qv_names(base_sd, fft_sd, lora_sd)

    print("\n=== Parameter Counts ===")
    print(f"base params:     {num_params(bundle.base):,}")
    print(f"fft_sst2 params: {num_params(bundle.fft_sst2):,}")
    print(f"lora params:     {num_params(bundle.lora_merged):,}")

    print("\n=== Matrix Coverage ===")
    print(f"base q/v: {len(base_qv)}")
    print(f"fft  q/v: {len(fft_qv)}")
    print(f"lora q/v: {len(lora_qv)}")
    print(f"common base-vs-fft: {len(common_base_fft)}")
    print(f"common base-vs-fft-vs-lora: {len(common_all)}")
    print(f"LoRA load mode: {bundle.lora_load_mode}")

    print("\n=== Sample Matrix Names ===")
    for idx, name in enumerate(common_base_fft[:5], start=1):
        print(f"{idx}. {name}")

    summary = {
        "models": {
            "base": BASE_MODEL_ID,
            "fft_sst2": FFT_SST2_MODEL_ID,
            "lora_sst2": LORA_SST2_MODEL_ID,
        },
        "lora_load_mode": bundle.lora_load_mode,
        "param_counts": {
            "base": num_params(bundle.base),
            "fft_sst2": num_params(bundle.fft_sst2),
            "lora_model": num_params(bundle.lora_merged),
        },
        "qv_counts": {
            "base": len(base_qv),
            "fft_sst2": len(fft_qv),
            "lora_model": len(lora_qv),
            "common_base_fft": len(common_base_fft),
            "common_all": len(common_all),
        },
        "sample_common_qv_names": common_base_fft[:10],
    }

    out_path = Path("outputs/download_verify_summary.json")
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
