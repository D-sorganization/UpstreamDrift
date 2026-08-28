"""CLI for serial execution of the prospective structural factorial."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Sequence
import hashlib
import json
import os
from pathlib import Path

from scripts.research.proximal_distal_energy.articulated_structural_factorial_runtime_audit import (
    validate_runtime_audit,
)

_SESSION_SCHEMA = "articulated-structural-factorial-session/1.0.0"


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _expected_execution_session(
    *,
    plan_path: Path,
    launch_path: Path,
    runtime_audit_path: Path,
    launch: dict[str, object],
    runtime_identity: str,
) -> dict[str, object]:
    return {
        "schema_version": _SESSION_SCHEMA,
        "execution_revision": launch.get("execution_revision"),
        "plan_file_sha256": _file_sha256(plan_path),
        "launch_file_sha256": _file_sha256(launch_path),
        "runtime_audit_file_sha256": _file_sha256(runtime_audit_path),
        "runtime_identity_sha256": runtime_identity,
    }


def validate_execution_session(
    *,
    plan_path: Path,
    launch_path: Path,
    runtime_audit_path: Path,
    launch: dict[str, object],
    runtime_identity: str,
    checkpoint_dir: Path,
) -> Path:
    """Return the session path only when its complete file identity matches."""

    session_path = checkpoint_dir / "execution-session.json"
    try:
        observed = json.loads(session_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("execution session identity is unreadable") from exc
    expected = _expected_execution_session(
        plan_path=plan_path,
        launch_path=launch_path,
        runtime_audit_path=runtime_audit_path,
        launch=launch,
        runtime_identity=runtime_identity,
    )
    if observed != expected:
        raise ValueError("execution session identity does not match this launch")
    return session_path


def _bind_execution_session(
    *,
    plan_path: Path,
    launch_path: Path,
    runtime_audit_path: Path,
    launch: dict[str, object],
    runtime_identity: str,
    checkpoint_dir: Path,
) -> Path:
    """Atomically bind an empty or matching directory to one execution identity."""

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    session_path = checkpoint_dir / "execution-session.json"
    expected = _expected_execution_session(
        plan_path=plan_path,
        launch_path=launch_path,
        runtime_audit_path=runtime_audit_path,
        launch=launch,
        runtime_identity=runtime_identity,
    )
    if session_path.exists():
        return validate_execution_session(
            plan_path=plan_path,
            launch_path=launch_path,
            runtime_audit_path=runtime_audit_path,
            launch=launch,
            runtime_identity=runtime_identity,
            checkpoint_dir=checkpoint_dir,
        )
    if any(checkpoint_dir.glob("case-*")):
        raise ValueError("populated checkpoint directory lacks an execution session")
    temporary = session_path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(expected, stream, indent=2, sort_keys=True, ensure_ascii=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, session_path)
    return session_path


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
    session_path = _bind_execution_session(
        plan_path=plan_path,
        launch_path=launch_path,
        runtime_audit_path=runtime_audit_path,
        launch=launch,
        runtime_identity=runtime_identity,
        checkpoint_dir=checkpoint_dir,
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
        "execution_session_path": str(session_path.resolve()),
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


__all__ = ["launch_structural_factorial", "main", "validate_execution_session"]
