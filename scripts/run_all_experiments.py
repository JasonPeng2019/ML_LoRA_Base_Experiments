from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REQUIRED = [
    "figures/sst2_svd_energy.png",
    "figures/sst2_lora_ft_overlap.png",
    "figures/oracle_routing_per_task.png",
    "tables/oracle_routing_results.tex",
    "outputs/oracle_routing_metrics.json",
]


def run(cmd: list[str]) -> None:
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def validate_outputs() -> None:
    missing = []
    for rel in REQUIRED:
        path = Path(rel)
        if not path.exists() or path.stat().st_size == 0:
            missing.append(rel)
    if missing:
        raise RuntimeError(f"Missing or empty artifacts: {missing}")
    print("All required artifacts are present and non-empty.")


def main() -> None:
    py = sys.executable
    run([py, "scripts/download_and_verify.py"])
    run([py, "scripts/compute_svd.py"])
    run([py, "scripts/compute_overlap.py"])
    run([py, "scripts/oracle_routing_eval.py"])
    run([py, "scripts/write_slide_snippets.py"])
    validate_outputs()


if __name__ == "__main__":
    main()
