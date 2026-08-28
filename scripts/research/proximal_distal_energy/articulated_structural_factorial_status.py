"""Validate and report a partial structural-factorial campaign."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from scripts.research.proximal_distal_energy.articulated_structural_factorial_runner import (
    build_registered_cases,
    load_available_checkpoints,
)


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def structural_factorial_status(
    *,
    plan: Mapping[str, object],
    launch: Mapping[str, object],
    checkpoint_dir: Path,
) -> dict[str, object]:
    """Return an identity-validated progress snapshot with no outcome inference."""

    cases = build_registered_cases(plan)
    checkpoints = load_available_checkpoints(
        plan=plan, launch=launch, checkpoint_dir=checkpoint_dir
    )
    status_counts = Counter(checkpoint.status for checkpoint in checkpoints)
    completed_paths = {
        checkpoint.path.with_suffix(".npz").name
        for checkpoint in checkpoints
        if checkpoint.status == "completed"
    }
    visible_npz = {path.name for path in checkpoint_dir.glob("case-*.npz")}
    inflight = sorted(visible_npz - completed_paths)
    completed_keys = {checkpoint.case.case_key for checkpoint in checkpoints}
    next_case = next(
        (case.case_key for case in cases if case.case_key not in completed_keys), None
    )
    latest_paths = [checkpoint.path for checkpoint in checkpoints]
    latest_timestamp = (
        max(path.stat().st_mtime for path in latest_paths) if latest_paths else None
    )
    return {
        "schema_version": "articulated-structural-factorial-status/1.0.0",
        "classification": "execution_progress_not_scientific_evidence",
        "execution_revision": launch.get("execution_revision"),
        "registered_case_count": len(cases),
        "retained_checkpoint_count": len(checkpoints),
        "status_counts": dict(sorted(status_counts.items())),
        "validated_completed_sidecar_count": len(completed_paths),
        "inflight_or_orphan_sidecar_count": len(inflight),
        "missing_case_count": len(cases) - len(checkpoints),
        "fraction_complete": len(checkpoints) / len(cases),
        "next_case_key": next_case,
        "latest_checkpoint_utc": (
            datetime.fromtimestamp(latest_timestamp, tz=timezone.utc).isoformat()
            if latest_timestamp is not None
            else None
        ),
        "complete": len(checkpoints) == len(cases),
        "promotion_eligible": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Read explicit manifests and print one validated JSON snapshot."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--launch", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    plan = _mapping(json.loads(args.plan.read_text(encoding="utf-8")), name="plan")
    launch = _mapping(
        json.loads(args.launch.read_text(encoding="utf-8")), name="launch"
    )
    result = structural_factorial_status(
        plan=plan, launch=launch, checkpoint_dir=args.checkpoint_dir
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["structural_factorial_status"]
