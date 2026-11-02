#!/usr/bin/env python3
"""Fail CI when a required pytest JUnit testcase did not pass."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from importlib.metadata import version
from pathlib import Path

import defusedxml.ElementTree as ET


def require_test_passed(
    junit_xml: Path,
    testcase_name: str,
) -> int:
    """Return zero only when at least one matching testcase passed."""
    if not junit_xml.exists():
        print(f"JUnit XML not found: {junit_xml}", file=sys.stderr)
        return 1

    root = ET.parse(junit_xml).getroot()
    matches = [
        testcase
        for testcase in root.findall(".//testcase")
        if testcase.attrib.get("name", "").startswith(testcase_name)
    ]
    if not matches:
        print(f"Required testcase not found: {testcase_name}", file=sys.stderr)
        return 1

    passed = [testcase for testcase in matches if not list(testcase)]
    if not passed:
        print(
            f"Required testcase did not pass: {testcase_name} "
            f"({len(matches)} matching cases)",
            file=sys.stderr,
        )
        return 1

    print(f"Required testcase passed: {testcase_name} ({len(passed)} cases)")
    return 0


def require_junit_cases_and_write_receipt(
    junit_xml: Path,
    classname: str,
    case_names: list[str],
    receipt_path: Path,
    source_revision: str,
    runner_os: str,
) -> int:
    if not case_names or len(case_names) != len(set(case_names)):
        return _fail("Required JUnit case names must be non-empty and unique.")
    if not junit_xml.exists():
        return _fail(f"JUnit XML not found: {junit_xml}")
    root = ET.parse(junit_xml).getroot()
    matches = [
        n
        for n in root.iter("testcase")
        if n.get("classname") == classname and n.get("name") in case_names
    ]
    if len(matches) != len(case_names) or {n.get("name") for n in matches} != set(
        case_names
    ):
        return _fail("Required JUnit cases are missing or ambiguous.")
    failed = {"failure", "error", "skipped"}
    if any(c.tag.rsplit("}", 1)[-1] in failed for n in matches for c in n):
        return _fail("A required JUnit case did not pass.")
    receipt = {
        "source_revision": source_revision,
        "python": sys.version,
        "platform": platform.platform(),
        "runner_os": runner_os,
        "packages": {name: version(name) for name in ("jaxsim", "pin")},
        "executed_case_count": len(matches),
        "executed_cases": [f"{classname}::{node.get('name')}" for node in matches],
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return 0


def _fail(message: str) -> int:
    print(message, file=sys.stderr)
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Require at least one named JUnit testcase to pass."
    )
    parser.add_argument("junit_xml", type=Path)
    parser.add_argument("testcase_name", nargs="?")
    parser.add_argument("--required-classname")
    parser.add_argument("--required-case", action="append", default=[])
    parser.add_argument("--receipt-output", type=Path)
    parser.add_argument("--source-revision")
    args = parser.parse_args()
    if args.required_case:
        if not all(
            (args.required_classname, args.receipt_output, args.source_revision)
        ):
            parser.error(
                "required cases need classname, receipt output, and source revision"
            )
        return require_junit_cases_and_write_receipt(
            args.junit_xml,
            args.required_classname,
            args.required_case,
            args.receipt_output,
            args.source_revision,
            os.environ.get("RUNNER_OS", "unknown"),
        )
    if not args.testcase_name:
        parser.error("testcase_name is required unless --required-case is used")
    return require_test_passed(
        args.junit_xml,
        args.testcase_name,
    )


if __name__ == "__main__":
    raise SystemExit(main())
