from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from transformers import AutoModelForSequenceClassification

BASE_MODEL_ID = "roberta-base"
FFT_SST2_MODEL_ID = "rambodazimi/roberta-base-finetuned-FFT-SST2"
LORA_SST2_MODEL_ID = "rambodazimi/roberta-base-finetuned-LoRA-SST2"

Q_SUBSTR = "attention.self.query.weight"
V_SUBSTR = "attention.self.value.weight"


@dataclass
class LoadBundle:
    base: torch.nn.Module
    fft_sst2: torch.nn.Module
    lora_merged: torch.nn.Module
    lora_load_mode: str


def ensure_dirs() -> None:
    for name in ("figures", "tables", "cache", "outputs"):
        Path(name).mkdir(parents=True, exist_ok=True)


def get_device() -> str:
    if torch.cuda.is_available():
        try:
            gpu_name = torch.cuda.get_device_name(0)
        except Exception:
            gpu_name = "unknown"
        print(f"Device: cuda ({gpu_name})")
        return "cuda"
    print("Device: cpu")
    return "cpu"


def num_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def qv_names_from_state_dict(state_dict: Dict[str, torch.Tensor]) -> List[str]:
    return sorted([name for name in state_dict if (Q_SUBSTR in name or V_SUBSTR in name)])


def load_classifier_model(model_id: str) -> torch.nn.Module:
    print(f"Loading {model_id} ...")
    model = AutoModelForSequenceClassification.from_pretrained(model_id)
    print(f"Loaded {model_id}")
    return model


def load_base_fft_and_lora_merged() -> LoadBundle:
    base = load_classifier_model(BASE_MODEL_ID)
    fft_sst2 = load_classifier_model(FFT_SST2_MODEL_ID)

    lora_load_mode = "direct_transformers_model"
    try:
        lora_model = load_classifier_model(LORA_SST2_MODEL_ID)
        lora_merged = lora_model
    except Exception as exc:
        print(f"Direct LoRA load failed: {type(exc).__name__}: {exc}")
        print("Trying PEFT adapter load + merge_and_unload ...")
        from peft import PeftModel

        base_for_lora = load_classifier_model(BASE_MODEL_ID)
        peft_model = PeftModel.from_pretrained(base_for_lora, LORA_SST2_MODEL_ID)
        lora_merged = peft_model.merge_and_unload()
        lora_load_mode = "peft_adapter_merged"

    return LoadBundle(base=base, fft_sst2=fft_sst2, lora_merged=lora_merged, lora_load_mode=lora_load_mode)


def common_qv_names(*state_dicts: Dict[str, torch.Tensor]) -> List[str]:
    if not state_dicts:
        return []
    sets = [set(qv_names_from_state_dict(sd)) for sd in state_dicts]
    return sorted(list(set.intersection(*sets)))


def build_qv_deltas(
    base_state_dict: Dict[str, torch.Tensor],
    tuned_state_dict: Dict[str, torch.Tensor],
    names: List[str],
) -> Dict[str, torch.Tensor]:
    deltas: Dict[str, torch.Tensor] = {}
    for name in names:
        base_tensor = base_state_dict[name].detach().cpu().float()
        tuned_tensor = tuned_state_dict[name].detach().cpu().float()
        if base_tensor.shape != tuned_tensor.shape:
            continue
        deltas[name] = tuned_tensor - base_tensor
    return deltas


def split_qv(name: str) -> Tuple[str, str]:
    if Q_SUBSTR in name:
        return "query", name
    return "value", name
