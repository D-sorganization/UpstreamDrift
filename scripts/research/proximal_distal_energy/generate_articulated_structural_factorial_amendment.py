"""Write a retention-only amendment after the legacy workflow is terminal."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
import os
from pathlib import Path
from typing import Any

from scripts.research.proximal_distal_energy.articulated_structural_factorial_plan import (
    StructuralFactorialPlan,
    StructuralFactorialPlanAmendment,
)


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _load_canonical_base(path: Path) -> StructuralFactorialPlan:
    raw = _mapping(json.loads(path.read_text(encoding="utf-8")), name="base plan")
    identity = _mapping(raw.get("identity"), name="base plan identity")
    authority_sha256 = _mapping(
        identity.get("authority_sha256"), name="base plan authority_sha256"
    )
    plan = StructuralFactorialPlan(
        design_authority_revision=str(identity.get("design_authority_revision")),
        authority_sha256={
            str(key): str(value) for key, value in authority_sha256.items()
        },
    )
    if raw != plan.to_manifest():
        raise ValueError("base plan is not the canonical unamended v1.2 manifest")
    return plan


def _write_atomic(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, sort_keys=False, ensure_ascii=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def main(argv: Sequence[str] | None = None) -> None:
    """Require explicit outcome-blind confirmation and write atomically."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-plan", type=Path, required=True)
    parser.add_argument("--legacy-execution-revision", required=True)
    parser.add_argument("--legacy-runtime-audit-run-id", type=int, required=True)
    parser.add_argument("--terminal-workflow-run-id", type=int, required=True)
    parser.add_argument("--terminal-conclusion", required=True)
    parser.add_argument("--legacy-prefix-case-stop-exclusive", type=int, required=True)
    parser.add_argument("--legacy-prefix-manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--confirm-no-scientific-outcome-inspection",
        action="store_true",
        required=True,
        help="Confirm the operational correction precedes outcome inspection.",
    )
    args = parser.parse_args(argv)
    plan = _load_canonical_base(args.base_plan)
    amendment = StructuralFactorialPlanAmendment(
        legacy_execution_revision=args.legacy_execution_revision,
        legacy_runtime_audit_run_id=args.legacy_runtime_audit_run_id,
        terminal_workflow_run_id=args.terminal_workflow_run_id,
        terminal_conclusion=args.terminal_conclusion,
        legacy_prefix_case_stop_exclusive=args.legacy_prefix_case_stop_exclusive,
        legacy_prefix_manifest_sha256=args.legacy_prefix_manifest_sha256,
        detected_before_scientific_outcome_inspection=True,
    )
    _write_atomic(args.output, plan.to_amended_manifest(amendment))


if __name__ == "__main__":  # pragma: no cover
    main()


__all__ = ["main"]
