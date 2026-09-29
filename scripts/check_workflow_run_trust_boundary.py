#!/usr/bin/env python3
"""Guard privileged workflow_run paths from executing PR-controlled code."""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

WORKFLOWS_DIR = Path(".github/workflows")
DISPATCH_ONLY_RE = re.compile(r"github\.event_name\s*==\s*['\"]workflow_dispatch['\"]")
WORKFLOW_RUN_RE = re.compile(r"\bworkflow_run\b")
PR_CODE_RUN_PATTERNS = (
    re.compile(r"\bpip\s+install\b"),
    re.compile(r"\bpython(?:3)?\s+scripts/mypy_autofix_agent\.py\b"),
    re.compile(r"\bruff\s+check\s+--fix\b"),
    re.compile(r"\bruff\s+format\b"),
    re.compile(r"\bautoflake\b"),
    re.compile(r"\bgit\s+(?:commit|push)\b"),
)
PR_CODE_ACTIONS = (
    "actions/checkout",
    "actions/setup-python",
)


@dataclass(frozen=True)
class Violation:
    """A workflow_run trust-boundary violation."""

    path: Path
    job: str
    step: str
    pattern: str
    detail: str


def _event_names(on_block: Any) -> set[str]:
    if isinstance(on_block, str):
        return {on_block}
    if isinstance(on_block, list):
        return {str(item) for item in on_block}
    if isinstance(on_block, dict):
        return {str(key) for key in on_block}
    return set()


def _job_can_run_on_workflow_run(job: dict[str, Any]) -> bool:
    condition = str(job.get("if", ""))
    return not (
        DISPATCH_ONLY_RE.search(condition) and not WORKFLOW_RUN_RE.search(condition)
    )


def _step_name(step: dict[str, Any], index: int) -> str:
    return str(step.get("name") or step.get("uses") or f"step {index}")


def _action_name(uses: str) -> str:
    return uses.split("@", 1)[0]


def default_workflows(workflows_dir: Path = WORKFLOWS_DIR) -> list[Path]:
    """Return every workflow file the guard scans when no path is given.

    ``find_violations`` returns no violations for a workflow that does not
    trigger on ``workflow_run``, so scanning the whole directory is equivalent
    to scanning the privileged subset without hardcoding a filename.
    """
    if not workflows_dir.is_dir():
        return []
    return sorted(
        path
        for path in workflows_dir.iterdir()
        if path.is_file() and path.suffix in {".yml", ".yaml"}
    )


def find_violations(path: Path) -> list[Violation]:
    """Return violations where workflow_run-capable jobs can execute PR code."""
    workflow = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(workflow, dict):
        raise ValueError(f"workflow is not a YAML mapping: {path}")

    on_block = workflow.get("on", workflow.get(True))
    if "workflow_run" not in _event_names(on_block):
        return []

    jobs = workflow.get("jobs", {})
    if not isinstance(jobs, dict):
        raise ValueError(f"workflow jobs are not a YAML mapping: {path}")

    violations: list[Violation] = []
    for job_name, job in jobs.items():
        if not isinstance(job, dict) or not _job_can_run_on_workflow_run(job):
            continue

        steps = job.get("steps", [])
        if not isinstance(steps, list):
            continue

        for index, step in enumerate(steps, start=1):
            if not isinstance(step, dict):
                continue

            step_name = _step_name(step, index)
            uses = str(step.get("uses", ""))
            action = _action_name(uses)
            if action in PR_CODE_ACTIONS:
                violations.append(
                    Violation(path, str(job_name), step_name, action, uses)
                )

            run = str(step.get("run", ""))
            for pattern in PR_CODE_RUN_PATTERNS:
                match = pattern.search(run)
                if match:
                    violations.append(
                        Violation(path, str(job_name), step_name, match.group(0), run)
                    )

    return violations


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "workflows",
        nargs="*",
        type=Path,
        help="Workflow files to scan. Defaults to every repository workflow.",
    )
    return parser.parse_args()


def main() -> int:
    """Print workflow_run trust-boundary violations."""
    args = parse_args()
    paths = args.workflows or default_workflows()
    violations: list[Violation] = []
    for path in paths:
        violations.extend(find_violations(path))
    for violation in violations:
        print(
            f"{violation.path}: job {violation.job!r}, step {violation.step!r}: "
            f"{violation.pattern}: {violation.detail.splitlines()[0]}",
            file=sys.stderr,
        )
    return 1 if violations else 0


if __name__ == "__main__":
    raise SystemExit(main())
