from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from oracle_routing_eval import aggregate_metrics, compute_metrics, primary_metric


def test_compute_metrics_accuracy_only() -> None:
    y_true = [1, 0, 1, 1]
    y_pred = [1, 0, 0, 1]
    metrics = compute_metrics(y_true, y_pred, ["accuracy"])
    assert metrics["accuracy"] == 0.75


def test_compute_metrics_accuracy_and_f1() -> None:
    y_true = [1, 1, 0, 0]
    y_pred = [1, 0, 0, 0]
    metrics = compute_metrics(y_true, y_pred, ["accuracy", "f1"])
    assert abs(metrics["accuracy"] - 0.75) < 1e-9
    assert abs(metrics["f1"] - (2.0 / 3.0)) < 1e-9


def test_primary_metric_uses_average_for_acc_f1_tasks() -> None:
    m = {"accuracy": 0.8, "f1": 0.6}
    assert primary_metric(m) == 0.7


def test_aggregate_metrics_macro_and_weighted() -> None:
    per_task = {
        "sst2": {"accuracy": 0.9, "primary_metric": 0.9},
        "mrpc": {"accuracy": 0.8, "f1": 0.7, "primary_metric": 0.75},
    }
    sample_counts = {"sst2": 100, "mrpc": 50}
    agg = aggregate_metrics(per_task, sample_counts)
    assert abs(agg["macro_accuracy"] - 0.85) < 1e-9
    assert abs(agg["weighted_accuracy"] - ((0.9 * 100 + 0.8 * 50) / 150)) < 1e-9
    assert abs(agg["macro_primary"] - 0.825) < 1e-9
    assert abs(agg["weighted_primary"] - ((0.9 * 100 + 0.75 * 50) / 150)) < 1e-9
    assert abs(agg["macro_f1"] - 0.7) < 1e-9
