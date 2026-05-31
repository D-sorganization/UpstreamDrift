#!/usr/bin/env python3
"""Fail a required parity gate when its assertions never executed (#6881).

A cross-engine parity job is only meaningful when its parity assertions
actually run. With optional dependencies installed best-effort, a JUnit report
can show every parity case *skipped* yet still exit zero, producing a green gate
that validated nothing.

This helper parses a JUnit XML report and fails (exit 1) when no test matching
the required name substring(s) ``passed`` — i.e. all matching cases were
skipped, errored, or absent. It is intended to run as a post-pytest step in a
*required* parity job. Local/default suites that legitimately skip optional
engines simply do not invoke this script.

Usage
-----
    python scripts/ci/assert_required_parity_ran.py \
        --junit test-results/cross-engine-equivalence-junit.xml \
        --require-name test_jaxsim_pinocchio_free_body_dynamics_terms_match

The script is dependency-free (stdlib ``xml.etree`` only) so it runs in any CI
environment regardless of installed engines. It is also DRY: future engine
parity jobs reuse it by passing a different ``--require-name``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from xml.etree import ElementTree


class ParityGateError(RuntimeError):
    """Raised when a required parity assertion did not execute and pass."""


def _matches(name: str, classname: str, required: tuple[str, ...]) -> bool:
    """Return True when a testcase name/classname matches any required token."""
    haystack = f"{classname}.{name}"
    return any(token in haystack for token in required)


def summarize_required_cases(
    junit_path: Path, required: tuple[str, ...]
) -> dict[str, int]:
    """Return pass/skip/fail/error counts for testcases matching ``required``.

    Raises
    ------
    FileNotFoundError
        If the JUnit report is missing (a required job must emit one).
    """
    if not junit_path.exists():
        raise FileNotFoundError(f"JUnit report not found: {junit_path}")

    tree = ElementTree.parse(junit_path)
    counts = {"passed": 0, "skipped": 0, "failed": 0, "error": 0, "matched": 0}

    for testcase in tree.iter("testcase"):
        name = testcase.get("name", "")
        classname = testcase.get("classname", "")
        if not _matches(name, classname, required):
            continue
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


def assert_parity_ran(junit_path: Path, required: tuple[str, ...]) -> dict[str, int]:
    """Assert at least one required parity case passed.

    Raises
    ------
    ParityGateError
        When no matching case passed (all skipped/failed/errored/absent).
    """
    if not required:
        raise ValueError("at least one --require-name token must be provided")

    counts = summarize_required_cases(junit_path, required)
    if counts["passed"] < 1:
        raise ParityGateError(
            "Required parity assertions did not run. "
            f"Matched {counts['matched']} case(s) for {list(required)}: "
            f"passed={counts['passed']} skipped={counts['skipped']} "
            f"failed={counts['failed']} error={counts['error']}. "
            "A required parity gate must not pass on skips alone."
        )
    return counts


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns process exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--junit",
        required=True,
        type=Path,
        help="Path to the JUnit XML report emitted by pytest --junitxml.",
    )
    parser.add_argument(
        "--require-name",
        action="append",
        required=True,
        dest="require_names",
        help="Substring a required testcase name/classname must contain. "
        "May be passed multiple times.",
    )
    args = parser.parse_args(argv)

    try:
        counts = assert_parity_ran(args.junit, tuple(args.require_names))
    except (ParityGateError, FileNotFoundError) as exc:
        print(f"::error::{exc}", file=sys.stderr)  # noqa: T201 - CI stderr signal
        return 1

    print(  # noqa: T201 - intentional CI summary to stdout
        f"Required parity gate satisfied: passed={counts['passed']} "
        f"skipped={counts['skipped']} matched={counts['matched']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
