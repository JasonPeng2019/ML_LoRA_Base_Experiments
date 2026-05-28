from transformers import AutoModelForSequenceClassification
from peft import PeftModel

BASE = "roberta-base"
LORA = "rambodazimi/roberta-base-finetuned-LoRA-SST2"

def qv_names(sd):
    return sorted([n for n in sd if "attention.self.query.weight" in n or "attention.self.value.weight" in n])

base = AutoModelForSequenceClassification.from_pretrained(BASE)
base_for_lora = AutoModelForSequenceClassification.from_pretrained(BASE)

peft_model = PeftModel.from_pretrained(base_for_lora, LORA)
merged_lora = peft_model.merge_and_unload()   # key step

base_sd = base.state_dict()
lora_sd = merged_lora.state_dict()

print("base q/v:", len(qv_names(base_sd)))
print("merged lora q/v:", len(qv_names(lora_sd)))
print("common:", len(set(qv_names(base_sd)) & set(qv_names(lora_sd))))
print("DONE")
