"""CLI for the serial preregistered rigid-refinement extension."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Sequence
import json
from pathlib import Path


def launch_rigid_refinement(
    *, plan_path: Path, execution_revision: str, checkpoint_dir: Path
) -> dict[str, object]:
    """Load a manifest and execute or resume its atomic serial cases."""

    manifest = json.loads(plan_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("manifest must be a mapping")
    from scripts.research.proximal_distal_energy.articulated_forward_smoke_evaluator import (
        run_registered_rigid_smoke,
    )

    checkpoints = run_registered_rigid_smoke(
        manifest=manifest,
        execution_revision=execution_revision,
        checkpoint_dir=checkpoint_dir,
    )
    counts = Counter(checkpoint.status for checkpoint in checkpoints)
    return {
        "checkpoint_dir": str(checkpoint_dir.resolve()),
        "case_count": len(checkpoints),
        "status_counts": dict(sorted(counts.items())),
        "resumed_count": sum(checkpoint.resumed for checkpoint in checkpoints),
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Run the extension from explicit immutable paths and identity."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--execution-revision", required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    result = launch_rigid_refinement(
        plan_path=args.plan,
        execution_revision=args.execution_revision,
        checkpoint_dir=args.checkpoint_dir,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through CLI
    raise SystemExit(main())


__all__ = ["launch_rigid_refinement", "main"]
