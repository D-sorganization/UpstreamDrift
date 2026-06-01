#!/usr/bin/env python3
"""Fail CI when a required pytest JUnit testcase did not pass."""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def require_test_passed(junit_xml: Path, testcase_name: str) -> int:
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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Require at least one named JUnit testcase to pass."
    )
    parser.add_argument("junit_xml", type=Path)
    parser.add_argument("testcase_name")
    args = parser.parse_args()
    return require_test_passed(args.junit_xml, args.testcase_name)


if __name__ == "__main__":
    raise SystemExit(main())
