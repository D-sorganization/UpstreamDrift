#!/usr/bin/env python3
"""Fail when workflow pytest invocations reference missing tracked test files."""

from __future__ import annotations

import argparse
import re
import shlex
from pathlib import Path

DEFAULT_WORKFLOWS = tuple(sorted(Path(".github/workflows").glob("*.yml")))
TEST_PATH_RE = re.compile(r"^tests/.+\.py$")


def _workflow_test_paths(path: Path) -> set[str]:
    paths: set[str] = set()
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip().rstrip("\\")
        if "tests/" not in line:
            continue
        try:
            parts = shlex.split(line, posix=True)
        except ValueError:
            parts = line.split()
        for part in parts:
            candidate = part.strip("'\"")
            if "$" in candidate or "*" in candidate:
                continue
            if TEST_PATH_RE.match(candidate):
                paths.add(candidate)
    return paths


def missing_workflow_test_paths(workflows: list[Path]) -> list[str]:
    failures: list[str] = []
    for workflow in workflows:
        for rel_path in sorted(_workflow_test_paths(workflow)):
            if not Path(rel_path).exists():
                failures.append(f"{workflow}: missing pytest path {rel_path}")
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workflows", nargs="*", type=Path)
    args = parser.parse_args(argv)
    workflows = args.workflows or list(DEFAULT_WORKFLOWS)
    failures = missing_workflow_test_paths(workflows)
    if failures:
        print("\n".join(failures))
        return 1
    print("Workflow pytest paths exist.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
