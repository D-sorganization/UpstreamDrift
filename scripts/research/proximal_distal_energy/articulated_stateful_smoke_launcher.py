"""CLI that qualifies native imports before loading numerical dependencies."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Sequence
from importlib import import_module
import json
from pathlib import Path
from typing import Any


def preload_registered_native_modules(
    engines: Sequence[str],
) -> dict[str, str]:
    """Load MuJoCo before NumPy and retain typed import failures for evaluation."""

    if not engines or any(engine not in {"mujoco", "pinocchio"} for engine in engines):
        raise ValueError("engines must contain registered native engine names")
    ordered = (["mujoco"] if "mujoco" in engines else []) + [
        engine for engine in engines if engine != "mujoco"
    ]
    outcomes: dict[str, str] = {}
    for engine in ordered:
        try:
            import_module(engine)
        except (ImportError, OSError) as error:
            outcomes[engine] = f"unavailable: {error}"
        else:
            outcomes[engine] = "loaded"
    return outcomes


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return value


def launch_stateful_campaign(
    *, plan_path: Path, execution_revision: str, checkpoint_dir: Path
) -> dict[str, object]:
    """Preload native modules, then execute the immutable serial campaign."""

    manifest = _mapping(json.loads(plan_path.read_text(encoding="utf-8")), "manifest")
    design = _mapping(manifest.get("design"), "design")
    engines = design.get("engines")
    if not isinstance(engines, list) or any(
        not isinstance(engine, str) for engine in engines
    ):
        raise ValueError("design.engines must be a string list")
    preload = preload_registered_native_modules(engines)
    from scripts.research.proximal_distal_energy.articulated_stateful_smoke_evaluator import (
        run_registered_stateful_smoke,
    )

    checkpoints = run_registered_stateful_smoke(
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
        "native_preload": preload,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Run the stateful campaign from explicit immutable paths and identity."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--execution-revision", required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    result = launch_stateful_campaign(
        plan_path=args.plan,
        execution_revision=args.execution_revision,
        checkpoint_dir=args.checkpoint_dir,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through CLI
    raise SystemExit(main())


__all__ = [
    "launch_stateful_campaign",
    "main",
    "preload_registered_native_modules",
]
