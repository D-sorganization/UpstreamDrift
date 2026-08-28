"""CLI for serial execution of the prospective structural factorial."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Sequence
import json
from pathlib import Path

from scripts.research.proximal_distal_energy.articulated_structural_factorial_runtime_audit import (
    validate_runtime_audit,
)


def launch_structural_factorial(
    *,
    plan_path: Path,
    launch_path: Path,
    runtime_audit_path: Path,
    checkpoint_dir: Path,
) -> dict[str, object]:
    """Run only after an immutable launch-specific runtime audit passes."""

    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    launch = json.loads(launch_path.read_text(encoding="utf-8"))
    runtime_audit = json.loads(runtime_audit_path.read_text(encoding="utf-8"))
    if (
        not isinstance(plan, dict)
        or not isinstance(launch, dict)
        or not isinstance(runtime_audit, dict)
    ):
        raise ValueError("plan, launch, and runtime audit must be mappings")
    runtime_identity = validate_runtime_audit(
        plan=plan, launch=launch, audit=runtime_audit
    )
    from scripts.research.proximal_distal_energy.articulated_structural_factorial_evaluator import (
        evaluate_structural_case,
    )
    from scripts.research.proximal_distal_energy.articulated_structural_factorial_runner import (
        run_serial_cases,
    )

    checkpoints = run_serial_cases(
        plan=plan,
        launch=launch,
        checkpoint_dir=checkpoint_dir,
        evaluator=lambda case: evaluate_structural_case(case, plan),
    )
    counts = Counter(checkpoint.status for checkpoint in checkpoints)
    return {
        "checkpoint_dir": str(checkpoint_dir.resolve()),
        "case_count": len(checkpoints),
        "status_counts": dict(sorted(counts.items())),
        "resumed_count": sum(checkpoint.resumed for checkpoint in checkpoints),
        "runtime_identity_sha256": runtime_identity,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Execute the exact plan and launch manifests supplied by the reviewer."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--launch", type=Path, required=True)
    parser.add_argument("--runtime-audit", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    result = launch_structural_factorial(
        plan_path=args.plan,
        launch_path=args.launch,
        runtime_audit_path=args.runtime_audit,
        checkpoint_dir=args.checkpoint_dir,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["launch_structural_factorial", "main"]
