from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.integration
@pytest.mark.skipif(os.environ.get("RUN_INTEGRATION") != "1", reason="set RUN_INTEGRATION=1 to run integration test")
def test_oracle_routing_smoke_end_to_end(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "oracle_routing_eval.py"

    cmd = [
        sys.executable,
        str(script),
        "--tasks",
        "sst2,mrpc,rte,mnli,qnli,qqp",
        "--max-examples-per-task",
        "16",
        "--batch-size",
        "8",
        "--device",
        "auto",
    ]
    subprocess.run(cmd, check=True, cwd=tmp_path)

    required = [
        tmp_path / "outputs" / "oracle_routing_metrics.json",
        tmp_path / "tables" / "oracle_routing_results.tex",
        tmp_path / "figures" / "oracle_routing_per_task.png",
        tmp_path / "figures" / "router_demo.png",
        tmp_path / "tables" / "router_demo_metrics.tex",
    ]
    for path in required:
        assert path.exists(), f"missing {path}"
        assert path.stat().st_size > 0, f"empty {path}"
