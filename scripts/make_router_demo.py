from __future__ import annotations

from oracle_routing_eval import DEFAULT_TASKS, run_oracle_routing_eval


def main() -> None:
    # Backward-compatible entrypoint now runs measured oracle-routed evaluation.
    run_oracle_routing_eval(
        tasks=DEFAULT_TASKS,
        max_examples_per_task=2000,
        batch_size=0,
        device_arg="auto",
        write_legacy_outputs=True,
    )


if __name__ == "__main__":
    main()
