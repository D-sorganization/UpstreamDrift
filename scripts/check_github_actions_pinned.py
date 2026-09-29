#!/usr/bin/env python3
"""Audit GitHub Actions workflow references for commit SHA pinning."""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

WORKFLOW_GLOB_PATTERNS = ("*.yml", "*.yaml")
USES_PATTERN = re.compile(r"^\s*(?:-\s*)?uses:\s*['\"]?([^'\"\s#]+)")
COMMIT_SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")


@dataclass(frozen=True)
class UnpinnedAction:
    """An external action reference that is not pinned to a commit SHA."""

    path: Path
    line_number: int
    reference: str
    reason: str


def iter_workflow_files(workflow_dir: Path) -> list[Path]:
    """Return all workflow YAML files under the workflow directory."""
    return sorted(
        path
        for pattern in WORKFLOW_GLOB_PATTERNS
        for path in workflow_dir.rglob(pattern)
        if path.is_file()
    )


def split_action_reference(reference: str) -> tuple[str, str | None]:
    """Split a workflow uses reference into action and ref components."""
    if reference.startswith("./"):
        return reference, None
    if "@" not in reference:
        return reference, None
    action, ref = reference.rsplit("@", 1)
    return action, ref


def audit_workflow_file(path: Path) -> list[UnpinnedAction]:
    """Find unpinned external action references in a workflow file."""
    findings: list[UnpinnedAction] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        match = USES_PATTERN.match(line)
        if not match:
            continue

        reference = match.group(1)
        action, ref = split_action_reference(reference)
        if action.startswith("./"):
            continue

        if ref is None:
            findings.append(
                UnpinnedAction(path, line_number, reference, "missing commit ref")
            )
            continue

        if COMMIT_SHA_PATTERN.fullmatch(ref) is None:
            findings.append(
                UnpinnedAction(path, line_number, reference, "ref is not a full SHA")
            )

    return findings


def audit_workflows(workflow_dir: Path) -> list[UnpinnedAction]:
    """Find unpinned external action references in all workflow files."""
    if not workflow_dir.exists():
        raise FileNotFoundError(f"workflow directory does not exist: {workflow_dir}")

    findings: list[UnpinnedAction] = []
    for path in iter_workflow_files(workflow_dir):
        findings.extend(audit_workflow_file(path))
    return findings


def build_parser() -> argparse.ArgumentParser:
    """Build the command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Fail if GitHub Actions are not pinned to commit SHAs."
    )
    parser.add_argument(
        "--workflow-dir",
        type=Path,
        default=Path(".github/workflows"),
        help="Directory containing GitHub Actions workflow YAML files.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the GitHub Actions pinning audit."""
    args = build_parser().parse_args(argv)
    findings = audit_workflows(args.workflow_dir)
    if findings:
        for finding in findings:
            print(
                f"{finding.path}:{finding.line_number}: {finding.reason}: "
                f"{finding.reference}",
                file=sys.stderr,
            )
        return 1

    print("All external GitHub Actions are pinned to commit SHAs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
