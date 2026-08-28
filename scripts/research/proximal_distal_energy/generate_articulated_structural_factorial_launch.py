"""Bind the structural-factorial plan to one immutable execution revision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.research.proximal_distal_energy.articulated_structural_factorial_runner import (
    build_launch_manifest,
)

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PLAN = ROOT / (
    "docs/research/proximal_distal_energy_transfer/data/"
    "articulated_structural_factorial_plan.json"
)
DEFAULT_OUTPUT = ROOT / (
    "docs/research/proximal_distal_energy_transfer/data/"
    "articulated_structural_factorial_launch.json"
)


def main() -> None:
    """Write one deterministic launch identity without executing outcomes."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--execution-revision", required=True)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    if not isinstance(plan, dict):
        raise ValueError("plan manifest must be a mapping")
    launch = build_launch_manifest(
        plan=plan, execution_revision=args.execution_revision
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(launch, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


if __name__ == "__main__":
    main()
