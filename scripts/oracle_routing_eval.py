from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_dataset
from peft import PeftModel
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

from pipeline_common import BASE_MODEL_ID, ensure_dirs


@dataclass(frozen=True)
class TaskSpec:
    name: str
    display_name: str
    glue_config: str
    split: str
    text_a_field: str
    text_b_field: Optional[str]
    metrics: Tuple[str, ...]
    lora_model_id: str
    fft_model_id: str


@dataclass
class TaskPayload:
    task: str
    text_a: List[str]
    text_b: Optional[List[str]]
    labels: List[int]
    num_labels: int


TASK_REGISTRY: Dict[str, TaskSpec] = {
    "sst2": TaskSpec(
        name="sst2",
        display_name="SST-2",
        glue_config="sst2",
        split="validation",
        text_a_field="sentence",
        text_b_field=None,
        metrics=("accuracy",),
        lora_model_id="rambodazimi/roberta-base-finetuned-LoRA-SST2",
        fft_model_id="rambodazimi/roberta-base-finetuned-FFT-SST2",
    ),
    "mrpc": TaskSpec(
        name="mrpc",
        display_name="MRPC",
        glue_config="mrpc",
        split="validation",
        text_a_field="sentence1",
        text_b_field="sentence2",
        metrics=("accuracy", "f1"),
        lora_model_id="rambodazimi/roberta-base-finetuned-LoRA-MRPC",
        fft_model_id="rambodazimi/roberta-base-finetuned-FFT-MRPC",
    ),
    "rte": TaskSpec(
        name="rte",
        display_name="RTE",
        glue_config="rte",
        split="validation",
        text_a_field="sentence1",
        text_b_field="sentence2",
        metrics=("accuracy",),
        lora_model_id="rambodazimi/roberta-base-finetuned-LoRA-RTE",
        fft_model_id="rambodazimi/roberta-base-finetuned-FFT-RTE",
    ),
    "mnli": TaskSpec(
        name="mnli",
        display_name="MNLI (matched)",
        glue_config="mnli",
        split="validation_matched",
        text_a_field="premise",
        text_b_field="hypothesis",
        metrics=("accuracy",),
        lora_model_id="rambodazimi/roberta-base-finetuned-LoRA-MNLI",
        fft_model_id="rambodazimi/roberta-base-finetuned-FFT-MNLI",
    ),
    "qnli": TaskSpec(
        name="qnli",
        display_name="QNLI",
        glue_config="qnli",
        split="validation",
        text_a_field="question",
        text_b_field="sentence",
        metrics=("accuracy",),
        lora_model_id="rambodazimi/roberta-base-finetuned-LoRA-QNLI",
        fft_model_id="rambodazimi/roberta-base-finetuned-FFT-QNLI",
    ),
    "qqp": TaskSpec(
        name="qqp",
        display_name="QQP",
        glue_config="qqp",
        split="validation",
        text_a_field="question1",
        text_b_field="question2",
        metrics=("accuracy", "f1"),
        lora_model_id="rambodazimi/roberta-base-finetuned-LoRA-QQP",
        fft_model_id="rambodazimi/roberta-base-finetuned-FFT-QQP",
    ),
}

DEFAULT_TASKS = ["sst2", "mrpc", "rte", "mnli", "qnli", "qqp"]


def parse_tasks(raw_tasks: str) -> List[str]:
    tasks = [x.strip().lower() for x in raw_tasks.split(",") if x.strip()]
    unknown = [t for t in tasks if t not in TASK_REGISTRY]
    if unknown:
        raise ValueError(f"Unknown tasks: {unknown}. Supported: {sorted(TASK_REGISTRY)}")
    return tasks


def resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        chosen = "cuda" if torch.cuda.is_available() else "cpu"
    elif device_arg == "cuda":
        chosen = "cuda" if torch.cuda.is_available() else "cpu"
        if chosen != "cuda":
            print("Requested cuda but it is unavailable. Falling back to cpu.")
    else:
        chosen = "cpu"

    if chosen == "cuda":
        print(f"Device: cuda ({torch.cuda.get_device_name(0)})")
    else:
        print("Device: cpu")
    return chosen


def auto_batch_size(device: str) -> int:
    if device != "cuda":
        return 16

    mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    if mem_gb >= 20:
        return 32
    if mem_gb >= 12:
        return 16
    return 8


def load_task_payload(spec: TaskSpec, max_examples: int) -> TaskPayload:
    dataset = load_dataset("glue", spec.glue_config, split=spec.split)
    if max_examples > 0:
        keep = min(max_examples, len(dataset))
        dataset = dataset.select(range(keep))

    labels = [int(v) for v in dataset["label"]]
    text_a = [str(v) for v in dataset[spec.text_a_field]]
    text_b = [str(v) for v in dataset[spec.text_b_field]] if spec.text_b_field else None
    num_labels = int(dataset.features["label"].num_classes)

    return TaskPayload(task=spec.name, text_a=text_a, text_b=text_b, labels=labels, num_labels=num_labels)


def load_lora_model(spec: TaskSpec, num_labels: int, device: str) -> torch.nn.Module:
    config = AutoConfig.from_pretrained(BASE_MODEL_ID, num_labels=num_labels)
    base_model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL_ID, config=config)
    peft_model = PeftModel.from_pretrained(base_model, spec.lora_model_id)
    merged = peft_model.merge_and_unload()
    return merged.to(device).eval()


def load_fft_model(spec: TaskSpec, device: str) -> torch.nn.Module:
    model = AutoModelForSequenceClassification.from_pretrained(spec.fft_model_id)
    return model.to(device).eval()


def predict_labels(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    payload: TaskPayload,
    spec: TaskSpec,
    batch_size: int,
    device: str,
) -> List[int]:
    preds: List[int] = []
    total = len(payload.labels)

    with torch.no_grad():
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            if spec.text_b_field is None:
                encoded = tokenizer(
                    payload.text_a[start:end],
                    truncation=True,
                    padding=True,
                    max_length=256,
                    return_tensors="pt",
                )
            else:
                encoded = tokenizer(
                    payload.text_a[start:end],
                    payload.text_b[start:end],
                    truncation=True,
                    padding=True,
                    max_length=256,
                    return_tensors="pt",
                )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            logits = model(**encoded).logits
            preds.extend(logits.argmax(dim=-1).detach().cpu().tolist())

    return preds


def compute_metrics(y_true: Sequence[int], y_pred: Sequence[int], metric_names: Sequence[str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if "accuracy" in metric_names:
        out["accuracy"] = float(accuracy_score(y_true, y_pred))
    if "f1" in metric_names:
        out["f1"] = float(f1_score(y_true, y_pred, average="binary"))
    return out


def primary_metric(metrics: Dict[str, float]) -> float:
    if "accuracy" in metrics and "f1" in metrics:
        return float((metrics["accuracy"] + metrics["f1"]) / 2.0)
    return float(metrics.get("accuracy", 0.0))


def evaluate_per_task_track(
    models: Dict[str, torch.nn.Module],
    tokenizer: AutoTokenizer,
    payloads: Dict[str, TaskPayload],
    specs: Dict[str, TaskSpec],
    batch_size: int,
    device: str,
) -> Dict[str, Dict[str, float]]:
    per_task: Dict[str, Dict[str, float]] = {}
    for task, payload in payloads.items():
        spec = specs[task]
        preds = predict_labels(models[task], tokenizer, payload, spec, batch_size=batch_size, device=device)
        metrics = compute_metrics(payload.labels, preds, spec.metrics)
        metrics["primary_metric"] = primary_metric(metrics)
        per_task[task] = metrics
    return per_task


def build_mixed_examples(payloads: Dict[str, TaskPayload]) -> List[Dict[str, object]]:
    mixed: List[Dict[str, object]] = []
    for task, payload in payloads.items():
        for i, label in enumerate(payload.labels):
            mixed.append(
                {
                    "task": task,
                    "text_a": payload.text_a[i],
                    "text_b": None if payload.text_b is None else payload.text_b[i],
                    "label": int(label),
                }
            )
    return mixed


def route_indices_by_task(chunk: Sequence[Dict[str, object]]) -> Dict[str, List[int]]:
    task_to_idxs: Dict[str, List[int]] = {}
    for idx, row in enumerate(chunk):
        task = str(row["task"])
        task_to_idxs.setdefault(task, []).append(idx)
    return task_to_idxs


def evaluate_oracle_routed_lora(
    lora_models: Dict[str, torch.nn.Module],
    tokenizer: AutoTokenizer,
    payloads: Dict[str, TaskPayload],
    specs: Dict[str, TaskSpec],
    batch_size: int,
    device: str,
) -> Dict[str, Dict[str, float]]:
    mixed = build_mixed_examples(payloads)

    y_true: Dict[str, List[int]] = {task: [] for task in payloads}
    y_pred: Dict[str, List[int]] = {task: [] for task in payloads}

    with torch.no_grad():
        for start in range(0, len(mixed), batch_size):
            end = min(start + batch_size, len(mixed))
            chunk = mixed[start:end]

            # Oracle router: dispatch each example to the task-specific expert.
            task_to_idxs = route_indices_by_task(chunk)

            for task, idxs in task_to_idxs.items():
                spec = specs[task]
                model = lora_models[task]

                text_a = [str(chunk[i]["text_a"]) for i in idxs]
                labels = [int(chunk[i]["label"]) for i in idxs]

                if spec.text_b_field is None:
                    encoded = tokenizer(
                        text_a,
                        truncation=True,
                        padding=True,
                        max_length=256,
                        return_tensors="pt",
                    )
                else:
                    text_b = [str(chunk[i]["text_b"]) for i in idxs]
                    encoded = tokenizer(
                        text_a,
                        text_b,
                        truncation=True,
                        padding=True,
                        max_length=256,
                        return_tensors="pt",
                    )

                encoded = {k: v.to(device) for k, v in encoded.items()}
                logits = model(**encoded).logits
                preds = logits.argmax(dim=-1).detach().cpu().tolist()

                y_true[task].extend(labels)
                y_pred[task].extend([int(p) for p in preds])

    per_task: Dict[str, Dict[str, float]] = {}
    for task, spec in specs.items():
        metrics = compute_metrics(y_true[task], y_pred[task], spec.metrics)
        metrics["primary_metric"] = primary_metric(metrics)
        per_task[task] = metrics

    return per_task


def aggregate_metrics(
    per_task: Dict[str, Dict[str, float]],
    sample_counts: Dict[str, int],
) -> Dict[str, float]:
    tasks = list(per_task.keys())
    total_n = float(sum(sample_counts[t] for t in tasks))

    macro_accuracy = float(np.mean([per_task[t]["accuracy"] for t in tasks]))
    weighted_accuracy = float(
        sum(per_task[t]["accuracy"] * sample_counts[t] for t in tasks) / max(total_n, 1.0)
    )

    primary_vals = [per_task[t]["primary_metric"] for t in tasks]
    macro_primary = float(np.mean(primary_vals))
    weighted_primary = float(sum(per_task[t]["primary_metric"] * sample_counts[t] for t in tasks) / max(total_n, 1.0))

    out = {
        "macro_accuracy": macro_accuracy,
        "weighted_accuracy": weighted_accuracy,
        "macro_primary": macro_primary,
        "weighted_primary": weighted_primary,
    }

    f1_tasks = [t for t in tasks if "f1" in per_task[t]]
    if f1_tasks:
        f1_n = float(sum(sample_counts[t] for t in f1_tasks))
        out["macro_f1"] = float(np.mean([per_task[t]["f1"] for t in f1_tasks]))
        out["weighted_f1"] = float(
            sum(per_task[t]["f1"] * sample_counts[t] for t in f1_tasks) / max(f1_n, 1.0)
        )

    return out


def make_oracle_plot(
    task_order: List[str],
    specs: Dict[str, TaskSpec],
    oracle_metrics: Dict[str, Dict[str, float]],
    lora_metrics: Dict[str, Dict[str, float]],
    fft_metrics: Dict[str, Dict[str, float]],
    out_path: Path,
) -> None:
    labels = [specs[t].display_name for t in task_order]
    x = np.arange(len(task_order))

    oracle = [oracle_metrics[t]["primary_metric"] for t in task_order]
    lora = [lora_metrics[t]["primary_metric"] for t in task_order]
    fft = [fft_metrics[t]["primary_metric"] for t in task_order]

    width = 0.25
    plt.figure(figsize=(11, 5.4))
    plt.bar(x - width, oracle, width=width, label="Oracle-routed LoRA", color="#1D39C4")
    plt.bar(x, lora, width=width, label="Per-task LoRA", color="#389E0D")
    plt.bar(x + width, fft, width=width, label="Per-task Full-FT", color="#D46B08")

    plt.title("Oracle-Routed Multi-Task Performance (Primary Metric)")
    plt.ylabel("Score")
    plt.ylim(0.0, 1.02)
    plt.xticks(x, labels, rotation=20, ha="right")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)


def write_oracle_table(
    task_order: List[str],
    specs: Dict[str, TaskSpec],
    sample_counts: Dict[str, int],
    oracle_metrics: Dict[str, Dict[str, float]],
    lora_metrics: Dict[str, Dict[str, float]],
    fft_metrics: Dict[str, Dict[str, float]],
    out_path: Path,
) -> None:
    row_end = " \\\\"
    lines = [
        "\\begin{tabular}{lrrrrrr}",
        "\\toprule",
        "Task & N & Oracle Primary & LoRA Primary & FFT Primary & Oracle Acc & Oracle F1" + row_end,
        "\\midrule",
    ]

    for task in task_order:
        spec = specs[task]
        row = (
            f"{spec.display_name} & {sample_counts[task]} & "
            f"{oracle_metrics[task]['primary_metric']:.4f} & "
            f"{lora_metrics[task]['primary_metric']:.4f} & "
            f"{fft_metrics[task]['primary_metric']:.4f} & "
            f"{oracle_metrics[task]['accuracy']:.4f} & "
            f"{oracle_metrics[task].get('f1', float('nan')):.4f}"
        )
        # Avoid rendering "nan" in non-F1 tasks.
        row_text = (row + row_end).replace("nan", "-")
        lines.append(row_text)

    lines.extend(["\\bottomrule", "\\end{tabular}"])
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_consistency_report(
    oracle_metrics: Dict[str, Dict[str, float]],
    lora_metrics: Dict[str, Dict[str, float]],
) -> Dict[str, object]:
    per_task = {}
    max_primary_diff = 0.0
    for task in oracle_metrics:
        diffs = {}
        for metric_name, value in oracle_metrics[task].items():
            if metric_name not in lora_metrics[task]:
                continue
            diff = abs(value - lora_metrics[task][metric_name])
            diffs[metric_name] = float(diff)
            if metric_name == "primary_metric":
                max_primary_diff = max(max_primary_diff, diff)
        per_task[task] = diffs
    return {
        "oracle_vs_per_task_lora_abs_diff": per_task,
        "max_primary_diff": float(max_primary_diff),
    }


def run_oracle_routing_eval(
    tasks: List[str],
    max_examples_per_task: int,
    batch_size: Optional[int],
    device_arg: str,
    write_legacy_outputs: bool = True,
) -> Dict[str, object]:
    ensure_dirs()
    t0 = time.time()

    specs = {t: TASK_REGISTRY[t] for t in tasks}
    device = resolve_device(device_arg)
    bs = batch_size if batch_size and batch_size > 0 else auto_batch_size(device)
    print(f"Using batch_size={bs}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)

    t_payload = time.time()
    payloads: Dict[str, TaskPayload] = {
        task: load_task_payload(specs[task], max_examples=max_examples_per_task) for task in tasks
    }
    payload_runtime = time.time() - t_payload

    sample_counts = {task: len(payloads[task].labels) for task in tasks}
    label_counts = {task: payloads[task].num_labels for task in tasks}

    t_lora_load = time.time()
    lora_models: Dict[str, torch.nn.Module] = {
        task: load_lora_model(specs[task], payloads[task].num_labels, device) for task in tasks
    }
    lora_load_runtime = time.time() - t_lora_load

    t_oracle = time.time()
    oracle_per_task = evaluate_oracle_routed_lora(
        lora_models, tokenizer, payloads, specs, batch_size=bs, device=device
    )
    oracle_runtime = time.time() - t_oracle

    t_lora_eval = time.time()
    per_task_lora = evaluate_per_task_track(
        lora_models, tokenizer, payloads, specs, batch_size=bs, device=device
    )
    per_task_lora_runtime = time.time() - t_lora_eval

    # Release LoRA models before loading FFT baselines.
    del lora_models
    if device == "cuda":
        torch.cuda.empty_cache()

    t_fft = time.time()
    per_task_fft: Dict[str, Dict[str, float]] = {}
    for task in tasks:
        model = load_fft_model(specs[task], device=device)
        per_task_fft[task] = evaluate_per_task_track(
            {task: model},
            tokenizer,
            {task: payloads[task]},
            {task: specs[task]},
            batch_size=bs,
            device=device,
        )[task]
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
    per_task_fft_runtime = time.time() - t_fft

    oracle_agg = aggregate_metrics(oracle_per_task, sample_counts)
    lora_agg = aggregate_metrics(per_task_lora, sample_counts)
    fft_agg = aggregate_metrics(per_task_fft, sample_counts)

    consistency = build_consistency_report(oracle_per_task, per_task_lora)

    fig_out = Path("figures/oracle_routing_per_task.png")
    make_oracle_plot(tasks, specs, oracle_per_task, per_task_lora, per_task_fft, fig_out)

    table_out = Path("tables/oracle_routing_results.tex")
    write_oracle_table(tasks, specs, sample_counts, oracle_per_task, per_task_lora, per_task_fft, table_out)

    if write_legacy_outputs:
        legacy_fig = Path("figures/router_demo.png")
        legacy_fig.write_bytes(fig_out.read_bytes())
        legacy_table = Path("tables/router_demo_metrics.tex")
        legacy_table.write_text(table_out.read_text(encoding="utf-8"), encoding="utf-8")

    total_runtime = time.time() - t0
    results = {
        "config": {
            "tasks": tasks,
            "max_examples_per_task": max_examples_per_task,
            "batch_size": bs,
            "device_request": device_arg,
            "device_used": device,
        },
        "sample_counts": sample_counts,
        "label_counts": label_counts,
        "tracks": {
            "oracle_lora": {
                "per_task": oracle_per_task,
                "aggregate": oracle_agg,
            },
            "per_task_lora": {
                "per_task": per_task_lora,
                "aggregate": lora_agg,
            },
            "per_task_fft": {
                "per_task": per_task_fft,
                "aggregate": fft_agg,
            },
        },
        "consistency": consistency,
        "runtime_sec": {
            "load_payloads": payload_runtime,
            "load_lora_models": lora_load_runtime,
            "oracle_eval": oracle_runtime,
            "per_task_lora_eval": per_task_lora_runtime,
            "per_task_fft_eval": per_task_fft_runtime,
            "total": total_runtime,
        },
        "artifacts": {
            "metrics_json": "outputs/oracle_routing_metrics.json",
            "results_table": "tables/oracle_routing_results.tex",
            "results_figure": "figures/oracle_routing_per_task.png",
            "legacy_router_table": "tables/router_demo_metrics.tex",
            "legacy_router_figure": "figures/router_demo.png",
        },
    }

    json_out = Path("outputs/oracle_routing_metrics.json")
    json_out.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"Wrote {json_out}")
    print(f"Wrote {table_out}")
    print(f"Wrote {fig_out}")
    print("Oracle-routing consistency max primary diff:", f"{consistency['max_primary_diff']:.6f}")

    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run measured oracle-routed multi-task LoRA evaluation.")
    parser.add_argument(
        "--tasks",
        type=str,
        default=",".join(DEFAULT_TASKS),
        help="Comma-separated task names. Default: sst2,mrpc,rte,mnli,qnli,qqp",
    )
    parser.add_argument(
        "--max-examples-per-task",
        type=int,
        default=2000,
        help="Max examples from each task validation split.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="Batch size for inference. 0 means auto.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Device selection policy.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    tasks = parse_tasks(args.tasks)
    run_oracle_routing_eval(
        tasks=tasks,
        max_examples_per_task=args.max_examples_per_task,
        batch_size=args.batch_size,
        device_arg=args.device,
        write_legacy_outputs=True,
    )


if __name__ == "__main__":
    main()
