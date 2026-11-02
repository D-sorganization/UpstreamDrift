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
    *,
    exact: bool = False,
    classname: str | None = None,
) -> int:
    """Return zero only when a matching testcase passed.

    Prefix matching remains available for legacy parameterized checks. Exact
    node-name matching prevents a similarly named testcase from satisfying a
    required-case guard.
    """
    if not junit_xml.exists():
        print(f"JUnit XML not found: {junit_xml}", file=sys.stderr)
        return 1

    if exact and not classname:
        print("Exact testcase checks require a JUnit classname.", file=sys.stderr)
        return 1

    root = ET.parse(junit_xml).getroot()
    matches = [
        testcase
        for testcase in root.findall(".//testcase")
        if (
            testcase.attrib.get("name", "") == testcase_name
            and testcase.attrib.get("classname", "") == classname
            if exact
            else testcase.attrib.get("name", "").startswith(testcase_name)
        )
    ]
    if not matches:
        identity = f"{classname}::{testcase_name}" if exact else testcase_name
        print(f"Required testcase not found: {identity}", file=sys.stderr)
        return 1
    if exact and len(matches) != 1:
        identity = f"{classname}::{testcase_name}"
        print(
            f"Required testcase is ambiguous: {identity} "
            f"({len(matches)} matching cases)",
            file=sys.stderr,
        )
        return 1

    failed_states = {"failure", "error", "skipped"}
    passed = [
        testcase
        for testcase in matches
        if not any(child.tag.rsplit("}", 1)[-1] in failed_states for child in testcase)
    ]
    if not passed:
        identity = f"{classname}::{testcase_name}" if exact else testcase_name
        print(
            f"Required testcase did not pass: {identity} "
            f"({len(matches)} matching cases)",
            file=sys.stderr,
        )
        return 1

    identity = f"{classname}::{testcase_name}" if exact else testcase_name
    print(f"Required testcase passed: {identity} ({len(passed)} cases)")
    return 0


def require_junit_cases_and_write_receipt(
    junit_xml: Path,
    classname: str,
    case_names: list[str],
    receipt_path: Path,
    source_revision: str,
    runner_os: str,
    packages: dict[str, str] | None = None,
) -> int:
    """Require each exact unique case to pass, then receipt the actual JUnit IDs."""
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
    packages = packages or {name: version(name) for name in ("jaxsim", "pin")}
    receipt = {
        "source_revision": source_revision,
        "python": sys.version,
        "platform": platform.platform(),
        "runner_os": runner_os,
        "packages": packages,
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
    parser.add_argument(
        "--exact",
        action="store_true",
        help="Match the full JUnit testcase name instead of a prefix.",
    )
    parser.add_argument(
        "--classname",
        help="Require the exact JUnit classname (required with --exact).",
    )
    parser.add_argument("--required-classname")
    parser.add_argument("--required-case", action="append", default=[])
    parser.add_argument("--receipt-output", type=Path)
    parser.add_argument("--source-revision")
    parser.add_argument("--package-version", action="append", default=[])
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
            dict(item.split("=", 1) for item in args.package_version),
        )
    if not args.testcase_name:
        parser.error("testcase_name is required unless --required-case is used")
    return require_test_passed(
        args.junit_xml,
        args.testcase_name,
        exact=args.exact,
        classname=args.classname,
    )


if __name__ == "__main__":
    raise SystemExit(main())
