#!/usr/bin/env python3
"""Enforce optional-stack pytest skip policy from JUnit results."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import defusedxml.ElementTree as ElementTree


class OptionalStackPolicyError(RuntimeError):
    """Raised when an optional-stack test lane did not validate real behavior."""


def summarize_junit(junit_path: Path) -> dict[str, int]:
    """Return pass/skip/fail/error testcase counts for a JUnit XML report."""
    if not junit_path.exists():
        raise FileNotFoundError(f"JUnit report not found: {junit_path}")

    counts = {"passed": 0, "skipped": 0, "failed": 0, "error": 0, "matched": 0}
    root = ElementTree.parse(junit_path).getroot()
    for testcase in root.iter("testcase"):
        counts["matched"] += 1
        if testcase.find("skipped") is not None:
            counts["skipped"] += 1
        elif testcase.find("failure") is not None:
            counts["failed"] += 1
        elif testcase.find("error") is not None:
            counts["error"] += 1
        else:
            counts["passed"] += 1
    return counts


def check_optional_stack_skip_policy(
    *,
    junit_path: Path,
    dependency_available: bool,
    pytest_exit_code: int,
) -> dict[str, int]:
    """Return JUnit counts or raise when optional-stack coverage is false-green."""
    if pytest_exit_code not in {0, 5}:
        raise OptionalStackPolicyError(
            "optional-stack pytest run failed with exit code "
            f"{pytest_exit_code}; see test output above"
        )

    if pytest_exit_code == 5:
        if dependency_available:
            raise OptionalStackPolicyError(
                "Pinocchio is available, but pytest collected zero ecosystem tests"
            )
        return {"passed": 0, "skipped": 0, "failed": 0, "error": 0, "matched": 0}

    counts = summarize_junit(junit_path)
    if dependency_available and counts["passed"] < 1:
        raise OptionalStackPolicyError(
            "Pinocchio is available, but no ecosystem testcase passed: "
            f"passed={counts['passed']} skipped={counts['skipped']} "
            f"failed={counts['failed']} error={counts['error']} "
            f"matched={counts['matched']}"
        )
    return counts


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def _append_summary(path: Path, counts: dict[str, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            "- Passed: {passed} | Failed: {failed} | Skipped: {skipped} | "
            "Matched: {matched}\n".format(**counts)
        )


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--junit", required=True, type=Path)
    parser.add_argument("--available", required=True, type=_parse_bool)
    parser.add_argument("--pytest-exit-code", required=True, type=int)
    parser.add_argument("--summary-file", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = build_parser().parse_args(argv)
    try:
        counts = check_optional_stack_skip_policy(
            junit_path=args.junit,
            dependency_available=args.available,
            pytest_exit_code=args.pytest_exit_code,
        )
    except (FileNotFoundError, OptionalStackPolicyError) as exc:
        print(f"::error::{exc}", file=sys.stderr)  # noqa: T201 - CI signal
        return 1

    if args.summary_file is not None:
        _append_summary(args.summary_file, counts)

    print(  # noqa: T201 - intentional CI summary
        "Optional-stack skip policy satisfied: "
        f"passed={counts['passed']} skipped={counts['skipped']} "
        f"failed={counts['failed']} error={counts['error']} "
        f"matched={counts['matched']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
