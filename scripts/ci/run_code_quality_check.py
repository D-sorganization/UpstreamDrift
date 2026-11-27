#!/usr/bin/env python3
"""Run the code-quality checker and return its status unchanged (#15765)."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CHECKER = REPO_ROOT / "src/tools/code_quality_check.py"
ZERO_SHA = "0" * 40
GIT_TIMEOUT_SECONDS = 120
CHECKER_TIMEOUT_SECONDS = 1200


def diff_base_for_event(
    event_name: str,
    *,
    base_ref: str,
    before_sha: str,
    merge_group_base_sha: str,
) -> str | None:
    """Return a review diff base, or None when the caller audits the tree.

    Preconditions: ``event_name`` is a non-empty string.
    Postcondition: None selects a full audit; a string is a git revision.
    """
    if not isinstance(event_name, str) or not event_name.strip():
        raise ValueError("event_name must be a non-empty string")
    if event_name == "pull_request":
        if not isinstance(base_ref, str) or not base_ref.strip():
            raise ValueError("pull_request requires base_ref")
        return f"origin/{base_ref.strip()}"
    if event_name == "push":
        if isinstance(before_sha, str) and before_sha.strip() not in {"", ZERO_SHA}:
            return before_sha.strip()
        return None
    if event_name == "merge_group":
        if (
            not isinstance(merge_group_base_sha, str)
            or not merge_group_base_sha.strip()
        ):
            raise ValueError("merge_group requires merge_group_base_sha")
        return merge_group_base_sha.strip()
    return None


def python_paths(changed_paths: list[str]) -> list[str]:
    """Return Python paths from ``changed_paths``.

    Preconditions: ``changed_paths`` is a list of strings.
    """
    if not isinstance(changed_paths, list):
        raise TypeError("changed_paths must be a list")
    selected: list[str] = []
    for path in changed_paths:
        if not isinstance(path, str):
            raise TypeError("changed path must be a string")
        if path.endswith(".py"):
            selected.append(path)
    return selected


def changed_python_files(base: str) -> list[str]:
    """Return added or modified Python paths between ``base`` and HEAD.

    Preconditions: ``base`` is a non-empty git revision.
    """
    if not isinstance(base, str) or not base.strip():
        raise ValueError("base must be a non-empty git revision")
    completed = subprocess.run(
        [
            "git",
            "diff",
            "--name-only",
            "--diff-filter=ACMRT",
            base,
            "HEAD",
            "--",
            "*.py",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=GIT_TIMEOUT_SECONDS,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "git diff failed")
    return [line for line in completed.stdout.splitlines() if line.endswith(".py")]


def run_checker(paths: list[str] | None, checker: Path | None = None) -> int:
    """Return the checker process status.

    Preconditions: ``paths`` is None or a list; ``checker`` is a Path or None.
    Postcondition: an empty path list returns 0 and does not launch a process.
    """
    if paths is not None and not isinstance(paths, list):
        raise TypeError("paths must be a list or None")
    if checker is not None and not isinstance(checker, Path):
        raise TypeError("checker must be a Path")
    if paths is not None and not paths:
        return 0
    command = [sys.executable, str(checker or CHECKER)]
    if paths is not None:
        command.extend(paths)
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=os.environ | {"PYTHONPATH": str(REPO_ROOT)},
        timeout=CHECKER_TIMEOUT_SECONDS,
        check=False,
    )
    return int(completed.returncode)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse runner arguments, defaulting review metadata from the environment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event", default=os.environ.get("EVENT_NAME", ""))
    parser.add_argument("--base-ref", default=os.environ.get("BASE_REF", ""))
    parser.add_argument("--before-sha", default=os.environ.get("BEFORE_SHA", ""))
    parser.add_argument(
        "--merge-group-base-sha",
        default=os.environ.get("MERGE_GROUP_BASE_SHA", ""),
    )
    parser.add_argument("--changed-file", action="append", default=None)
    parser.add_argument("--checker", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Check changed Python files, or the whole tree outside review events."""
    args = _parse_args(argv)
    if args.changed_file is not None:
        return run_checker(python_paths(args.changed_file), checker=args.checker)
    base = diff_base_for_event(
        args.event or "workflow_dispatch",
        base_ref=args.base_ref,
        before_sha=args.before_sha,
        merge_group_base_sha=args.merge_group_base_sha,
    )
    if base is None:
        return run_checker(None, checker=args.checker)
    return run_checker(changed_python_files(base), checker=args.checker)


if __name__ == "__main__":
    raise SystemExit(main())
