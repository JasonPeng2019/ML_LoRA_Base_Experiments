import torch
from transformers import AutoModelForSequenceClassification

BASE = "roberta-base"
FFT = "rambodazimi/roberta-base-finetuned-FFT-SST2"
LORA = "rambodazimi/roberta-base-finetuned-LoRA-SST2"

def qv_names(sd):
    return sorted([n for n in sd if "attention.self.query.weight" in n or "attention.self.value.weight" in n])

print("Device:", "cuda" if torch.cuda.is_available() else "cpu")

print("\n[1] Loading base...")
base = AutoModelForSequenceClassification.from_pretrained(BASE)
base_sd = base.state_dict()
print("base loaded")

print("\n[2] Loading full-FT...")
fft = AutoModelForSequenceClassification.from_pretrained(FFT)
fft_sd = fft.state_dict()
print("full-FT loaded")

print("\n[3] Trying LoRA as normal model...")
lora_sd = None
try:
    lora = AutoModelForSequenceClassification.from_pretrained(LORA)
    lora_sd = lora.state_dict()
    print("LoRA loaded as normal model")
except Exception as e:
    print("Normal load failed:", type(e).__name__)
    print("[4] Trying PEFT load...")
    from peft import PeftModel
    peft_model = PeftModel.from_pretrained(base, LORA)
    lora_sd = peft_model.state_dict()
    print("LoRA loaded via PEFT")

base_qv = qv_names(base_sd)
fft_qv = qv_names(fft_sd)
lora_qv = qv_names(lora_sd)

print("\n[5] Matrix counts")
print("base q/v:", len(base_qv))
print("fft  q/v:", len(fft_qv))
print("lora q/v:", len(lora_qv))

common_base_fft = sorted(set(base_qv) & set(fft_qv))
common_all = sorted(set(base_qv) & set(fft_qv) & set(lora_qv))

print("\n[6] Comparable matrices")
print("base vs fft:", len(common_base_fft))
print("base vs fft vs lora:", len(common_all))

if common_base_fft:
    print("\nExample base-vs-fft matrix:", common_base_fft[0])
if common_all:
    print("Example all-3 matrix:", common_all[0])

print("\nSMOKE TEST DONE")
