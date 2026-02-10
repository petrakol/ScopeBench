from __future__ import annotations

import argparse
from pathlib import Path

from scopebench.runtime.guard import evaluate_from_files
from scopebench.tracing.otel import init_tracing


def iter_example_pairs(examples_dir: Path):
    for contract_file in sorted(examples_dir.glob("*.contract.yaml")):
        stem = contract_file.name.replace(".contract.yaml", "")
        plan_file = examples_dir / f"{stem}.plan.yaml"
        if plan_file.exists():
            yield contract_file, plan_file


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Replay example plan/contract pairs and emit OTel spans."
    )
    parser.add_argument(
        "--examples-dir",
        default="examples",
        help="Directory containing *.contract.yaml and *.plan.yaml files",
    )
    parser.add_argument(
        "--enable-console",
        action="store_true",
        help="Enable OpenTelemetry ConsoleSpanExporter output",
    )
    args = parser.parse_args()

    init_tracing(enable_console=args.enable_console)

    examples_dir = Path(args.examples_dir)
    if not examples_dir.exists():
        raise SystemExit(f"examples directory not found: {examples_dir}")

    for contract_file, plan_file in iter_example_pairs(examples_dir):
        result = evaluate_from_files(str(contract_file), str(plan_file))
        aggregate_max = max(result.aggregate.as_dict().values())
        print(
            f"{contract_file.stem}: decision={result.policy.decision.value} "
            f"max_axis={aggregate_max:.3f} steps={result.aggregate.n_steps}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
