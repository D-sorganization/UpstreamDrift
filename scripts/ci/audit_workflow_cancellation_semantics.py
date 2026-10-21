#!/usr/bin/env python3
"""Audit GitHub Actions concurrency for merge-queue-safe cancellation (UD-CI-032).

Postcondition: returns a list of human-readable violation strings; an empty list
means every audited workflow complies with fleet cancellation semantics.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

READONLY_QUEUE_BRANCH_IGNORE = "gh-readonly-queue/**"
PR_ONLY_CANCEL_EXPRESSION = "${{ github.event_name == 'pull_request' }}"

# Workflows that legitimately keep long-running publishes from being interrupted.
CANCEL_FALSE_ALLOWLIST = frozenset(
    {
        "approve-same-repo-runs.yml",
        "release.yml",
    }
)


def _workflow_on(data: dict[str, Any]) -> dict[str, Any]:
    """Return the workflow trigger mapping (PyYAML may parse ``on`` as True)."""
    on = data.get("on", data.get(True))
    if not isinstance(on, dict):
        return {}
    return on


def _push_branches(on: dict[str, Any]) -> list[str] | None:
    """Return branch filters for a ``push`` trigger, or None when not branch-scoped."""
    push = on.get("push")
    if not isinstance(push, dict):
        return None
    branches = push.get("branches")
    if branches is None:
        return None
    if isinstance(branches, str):
        return [branches]
    if isinstance(branches, list):
        return [str(item) for item in branches]
    return None


def _has_readonly_queue_ignore(on: dict[str, Any]) -> bool:
    push = on.get("push")
    if not isinstance(push, dict):
        return False
    ignore = push.get("branches-ignore") or []
    if isinstance(ignore, str):
        ignore = [ignore]
    return any(READONLY_QUEUE_BRANCH_IGNORE in str(item) for item in ignore)


def audit_workflow_file(path: Path, data: dict[str, Any]) -> list[str]:
    """Return violation messages for one workflow document."""
    if not isinstance(data, dict):
        return [f"{path.name}: workflow root is not a mapping"]

    on = _workflow_on(data)
    events = set(on.keys())
    concurrency = data.get("concurrency")
    if not isinstance(concurrency, dict):
        return [f"{path.name}: missing concurrency block"]

    cancel = concurrency.get("cancel-in-progress")
    group = str(concurrency.get("group", ""))
    violations: list[str] = []

    push_branches = _push_branches(on)
    has_branch_push = push_branches is not None
    has_merge_group = "merge_group" in events

    if path.name in CANCEL_FALSE_ALLOWLIST:
        if cancel is not False:
            violations.append(
                f"{path.name}: cancel-in-progress must be false (protected publish lane)"
            )
        return violations

    if cancel is None:
        violations.append(f"{path.name}: concurrency.cancel-in-progress is required")
        return violations

    needs_pr_only_cancel = has_merge_group or has_branch_push
    if needs_pr_only_cancel:
        if cancel is True:
            violations.append(
                f"{path.name}: cancel-in-progress must be PR-scoped "
                f"({PR_ONLY_CANCEL_EXPRESSION!r}) so merge_group and main runs "
                "are never cancelled"
            )
        elif str(cancel).strip() != PR_ONLY_CANCEL_EXPRESSION:
            violations.append(
                f"{path.name}: cancel-in-progress must be exactly "
                f"{PR_ONLY_CANCEL_EXPRESSION!r}, got {cancel!r}"
            )
    elif cancel is not True and str(cancel).strip() != PR_ONLY_CANCEL_EXPRESSION:
        violations.append(
            f"{path.name}: cancel-in-progress must be true or the PR-only expression"
        )

    if has_branch_push and not _has_readonly_queue_ignore(on):
        violations.append(
            f"{path.name}: push.branches must include "
            f"branches-ignore: [{READONLY_QUEUE_BRANCH_IGNORE!r}]"
        )

    if has_merge_group and "github.event_name" not in group:
        violations.append(
            f"{path.name}: merge_group workflows must include github.event_name "
            "in the concurrency group"
        )

    return violations


def audit_workflow_paths(paths: list[Path]) -> list[str]:
    """Audit an explicit list of workflow files."""
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required for workflow cancellation audit") from exc

    violations: list[str] = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        data = yaml.safe_load(text)
        violations.extend(audit_workflow_file(path, data))
    return violations


def iter_workflow_files(workflow_dir: Path) -> list[Path]:
    """Return active workflow YAML files directly under ``.github/workflows``."""
    return sorted(
        path
        for pattern in ("*.yml", "*.yaml")
        for path in workflow_dir.glob(pattern)
        if path.is_file()
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workflow-dir",
        type=Path,
        default=Path(".github/workflows"),
        help="Directory containing workflow YAML files",
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="Optional workflow files to audit (default: entire workflow dir)",
    )
    args = parser.parse_args(argv)

    workflow_dir = args.workflow_dir
    if args.paths:
        paths = [path for path in args.paths if path.is_file()]
    else:
        if not workflow_dir.is_dir():
            print(f"workflow directory does not exist: {workflow_dir}", file=sys.stderr)
            return 2
        paths = iter_workflow_files(workflow_dir)

    violations = audit_workflow_paths(paths)
    if violations:
        for message in violations:
            print(message, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
