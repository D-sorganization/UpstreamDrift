#!/usr/bin/env python3
"""Run the code-quality checker and propagate its exit status (#15765).

Review events check changed Python files. Schedule and manual runs audit
the tree. A non-zero checker status is returned unchanged.
"""

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
    """Return the git base for a review diff, or None for a full audit.

    Preconditions:
        ``event_name`` is a non-empty string.
    Postconditions:
        None means the caller audits every Python file. A string is a git
        revision the caller diffs against HEAD.
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
    """Return the Python files from a list of repository paths.

    Preconditions:
        ``changed_paths`` is a list of strings.
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
    """Return Python paths changed between ``base`` and HEAD.

    Preconditions:
        ``base`` is a non-empty git revision.
    Postconditions:
        The list contains added, copied, modified, renamed, and type-changed
        Python files only.
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
        detail = completed.stderr.strip() or "git diff failed"
        raise RuntimeError(detail)
    return [line for line in completed.stdout.splitlines() if line.endswith(".py")]


def run_checker(paths: list[str] | None, checker: Path | None = None) -> int:
    """Run the code-quality checker and return its exit status.

    Preconditions:
        ``paths`` is None for a full audit, or a list of file paths.
        ``checker`` is the checker script, or None for the repository checker.
    Postconditions:
        The return value is the checker process status. An empty list
        returns 0 without launching the checker.
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


def main(argv: list[str] | None = None) -> int:
    """Check changed Python files, or the whole tree outside review events."""
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
    args = parser.parse_args(argv)
    if args.changed_file is not None:
        return run_checker(python_paths(args.changed_file), checker=args.checker)
    event_name = args.event or "workflow_dispatch"
    base = diff_base_for_event(
        event_name,
        base_ref=args.base_ref,
        before_sha=args.before_sha,
        merge_group_base_sha=args.merge_group_base_sha,
    )
    if base is None:
        return run_checker(None, checker=args.checker)
    return run_checker(changed_python_files(base), checker=args.checker)


if __name__ == "__main__":
    raise SystemExit(main())
