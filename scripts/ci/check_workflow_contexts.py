#!/usr/bin/env python3
"""Reject GitHub Actions contexts that are unavailable before job scheduling."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover - exercised only in underprovisioned CI
    yaml = None  # type: ignore[assignment]

WORKFLOW_DIR = Path(".github") / "workflows"
RUNNER_CONTEXT_RE = re.compile(r"\$\{\{\s*runner\.")


@dataclass(frozen=True)
class WorkflowContextViolation:
    """A workflow expression that GitHub cannot resolve at that YAML location."""

    path: Path
    job_id: str
    env_name: str
    value: str


def _workflow_paths(workflow_dir: Path) -> list[Path]:
    if not workflow_dir.exists():
        return []
    return sorted(
        path for path in workflow_dir.rglob("*") if path.suffix in {".yml", ".yaml"}
    )


def _load_workflow(path: Path) -> dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML is required to parse workflow files")

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: workflow root must be a mapping")
    return data


def _job_env_violations(
    path: Path, job_id: str, job: dict[str, Any]
) -> list[WorkflowContextViolation]:
    env = job.get("env")
    if not isinstance(env, dict):
        return []

    violations: list[WorkflowContextViolation] = []
    for env_name, value in env.items():
        if (
            isinstance(env_name, str)
            and isinstance(value, str)
            and RUNNER_CONTEXT_RE.search(value)
        ):
            violations.append(
                WorkflowContextViolation(
                    path=path,
                    job_id=job_id,
                    env_name=env_name,
                    value=value,
                )
            )
    return violations


def find_violations(workflow_paths: list[Path]) -> list[WorkflowContextViolation]:
    """Return job-level env expressions that use unavailable runner context."""
    violations: list[WorkflowContextViolation] = []
    for path in workflow_paths:
        data = _load_workflow(path)
        jobs = data.get("jobs") or {}
        if not isinstance(jobs, dict):
            continue
        for job_id, job in jobs.items():
            if isinstance(job_id, str) and isinstance(job, dict):
                violations.extend(_job_env_violations(path, job_id, job))
    return violations


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "workflows",
        nargs="*",
        type=Path,
        help="Workflow files to scan. Defaults to every workflow file.",
    )
    parser.add_argument(
        "--workflow-dir",
        type=Path,
        default=WORKFLOW_DIR,
        help="Directory scanned when no workflow file is supplied.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Print violations and return non-zero when workflow contexts are invalid."""
    args = parse_args(argv)
    workflow_paths = args.workflows or _workflow_paths(args.workflow_dir)
    violations = find_violations(workflow_paths)
    for violation in violations:
        print(
            f"{violation.path}:{violation.job_id}: job env {violation.env_name} "
            f"uses unavailable runner context: {violation.value}"
        )
    return 1 if violations else 0


if __name__ == "__main__":
    raise SystemExit(main())
