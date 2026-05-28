from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from oracle_routing_eval import DEFAULT_TASKS, TASK_REGISTRY, parse_tasks, route_indices_by_task


def test_default_tasks_exist_in_registry() -> None:
    for task in DEFAULT_TASKS:
        assert task in TASK_REGISTRY


def test_mnli_schema_is_three_way_nli() -> None:
    mnli = TASK_REGISTRY["mnli"]
    assert mnli.split == "validation_matched"
    assert mnli.text_a_field == "premise"
    assert mnli.text_b_field == "hypothesis"


def test_parse_tasks_validates_input() -> None:
    tasks = parse_tasks("sst2,mrpc,rte")
    assert tasks == ["sst2", "mrpc", "rte"]


def test_oracle_router_groups_indices_by_task() -> None:
    chunk = [
        {"task": "sst2", "text_a": "a", "text_b": None, "label": 1},
        {"task": "mrpc", "text_a": "b", "text_b": "c", "label": 0},
        {"task": "sst2", "text_a": "d", "text_b": None, "label": 0},
        {"task": "qqp", "text_a": "e", "text_b": "f", "label": 1},
        {"task": "mrpc", "text_a": "g", "text_b": "h", "label": 1},
    ]
    grouped = route_indices_by_task(chunk)
    assert grouped == {"sst2": [0, 2], "mrpc": [1, 4], "qqp": [3]}
