#!/usr/bin/env python3
"""MATLAB Quality Check Script.

This script runs comprehensive quality checks on MATLAB code following the project's
requirements. It decomposes complex analysis into orthogonal components.

Refactored to address Orthogonality and God function violations (Pragmatic Programmer).
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import timezone, datetime
from pathlib import Path
from typing import cast

from src.shared.python.logging_config import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)

MATLAB_SCRIPT_TIMEOUT_SECONDS: int = 300
MIN_DOCSTRING_LENGTH: int = 3


class MATLABQualityChecker:
    """Comprehensive MATLAB code quality checker."""

    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root
        self.matlab_dir = project_root / "matlab"
        self.results: dict[str, object] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_files": 0,
            "issues": [],
            "passed": True,
            "summary": "",
            "checks": {},
        }

    def _check_docstring_and_args(
        self, lines: list[str], i: int, file_name: str, issues: list[str]
    ) -> None:
        """Check for docstrings and arguments block in functions."""
        # Check docstring
        has_doc = False
        for j in range(i, min(i + 5, len(lines))):
            next_l = lines[j].strip()
            if next_l and not next_l.startswith("%"):
                break
            if next_l.startswith("%") and len(next_l) > MIN_DOCSTRING_LENGTH:
                has_doc = True
                break
        if not has_doc:
            issues.append(f"{file_name} (line {i}): Missing function docstring")

        # Check arguments block
        has_args = False
        for j in range(i, min(i + 15, len(lines))):
            l_check = lines[j].strip()
            if not l_check.startswith("%") and re.match(r"\barguments\b", l_check):
                has_args = True
                break
        if not has_args:
            issues.append(f"{file_name} (line {i}): Missing arguments validation block")

    def _check_banned_patterns(
        self, line: str, i: int, file_name: str, issues: list[str]
    ) -> None:
        """Check for banned placeholders and template patterns."""
        banned = [
            (r"\bTODO\b", "TODO placeholder found"),
            (r"\bFIXME\b", "FIXME placeholder found"),
            (r"\bHACK\b", "HACK comment found"),
            (r"\bXXX\b", "XXX comment found"),
            (r"<[A-Z_][A-Z0-9_]*>", "Angle bracket placeholder found"),
            (r"\{\{.*?\}\}", "Template placeholder found"),
        ]
        for pattern, msg in banned:
            if re.search(pattern, line):
                issues.append(f"{file_name} (line {i}): {msg}")

    def _check_anti_patterns(
        self, line: str, i: int, file_name: str, issues: list[str]
    ) -> None:
        """Check for common MATLAB anti-patterns."""
        anti = [
            (r"\beval\s*\(", "Avoid eval() - security/performance risk"),
            (r"\bassignin\s*\(", "Avoid assignin() - violates encapsulation"),
            (r"\bevalin\s*\(", "Avoid evalin() - violates encapsulation"),
            (r"\bglobal\s+\w+", "Global variable usage - use arguments instead"),
            (r"\bexist\s*\(", "Consider validation/try-catch instead of exist()"),
        ]
        for pattern, msg in anti:
            if re.search(pattern, line):
                issues.append(f"{file_name} (line {i}): {msg}")

        # load without output
        if (
            re.search(r"^\s*load\s+\w+", line)
            or re.search(r"^\s*load\s*\([^)]+\)", line)
        ) and "=" not in line:
            issues.append(
                f"{file_name} (line {i}): load without output - use 'data = load(...)'"
            )

    def _check_magic_numbers(
        self, line: str, i: int, file_name: str, issues: list[str]
    ) -> None:
        """Check for unexplained magic numbers."""
        acceptable = {
            "0",
            "0.0",
            "1",
            "1.0",
            "2",
            "2.0",
            "3",
            "3.0",
            "4",
            "4.0",
            "5",
            "5.0",
            "10",
            "10.0",
            "0.5",
            "0.1",
        }
        matches = re.findall(r"(?<![.\w])(?:\d+\.\d+|\d+)(?![.\w])", line)
        for num in matches:
            if num not in acceptable:
                # Check for trailing comment explaining it
                if "%" not in line or line.find(num) > line.find("%"):
                    issues.append(
                        f"{file_name} (line {i}): Magic number {num} - define as named constant with unit/source"
                    )

    def _check_function_unsafe(
        self, line: str, i: int, file_name: str, issues: list[str]
    ) -> None:
        """Check for dangerous commands inside functions."""
        unsafe = [
            (r"\bclear\s+(all|global)\b", "Avoid 'clear all/global' in functions"),
            (r"\bclear\b(?!\s+\w+)", "Avoid 'clear' in functions"),
            (r"\bclc\b", "Avoid 'clc' in functions"),
            (r"\bclose\s+all\b", "Avoid 'close all' in functions"),
            (r"\baddpath\s*\(", "Avoid addpath in functions"),
        ]
        for pattern, msg in unsafe:
            if re.search(pattern, line, re.IGNORECASE):
                issues.append(f"{file_name} (line {i}): {msg}")

    def _analyze_matlab_file(self, file_path: Path) -> list[str]:
        """Analyze a single MATLAB file (Decomposed for Orthogonality)."""
        issues: list[str] = []
        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
            lines = content.splitlines()
            in_func, nesting = False, 0

            for i, line in enumerate(lines, 1):
                stripped = line.strip()
                if not stripped:
                    continue
                is_comment = stripped.startswith("%")

                # Track function scope
                if not is_comment:
                    if re.match(
                        r"\b(function|if|for|while|switch|try|parfor|classdef|arguments|properties|methods|events)\b",
                        stripped,
                    ):
                        if stripped.startswith("function"):
                            in_func = True
                        nesting += 1
                    if re.match(r"\bend\b", stripped):
                        nesting = max(0, nesting - 1)
                        if nesting == 0:
                            in_func = False

                    # Function-specific checks
                    if stripped.startswith("function"):
                        self._check_docstring_and_args(lines, i, file_path.name, issues)

                # General pattern checks
                self._check_banned_patterns(stripped, i, file_path.name, issues)
                if not is_comment:
                    self._check_anti_patterns(stripped, i, file_path.name, issues)
                    self._check_magic_numbers(line, i, file_path.name, issues)
                    if in_func:
                        self._check_function_unsafe(stripped, i, file_path.name, issues)

        except Exception as e:
            issues.append(f"{file_path.name}: Failed analysis - {e}")
        return issues

    def run_all_checks(self) -> dict[str, object]:
        """Run all quality checks."""
        if not self.matlab_dir.exists():
            return {"passed": True, "summary": "No MATLAB dir found", "issues": []}

        m_files = list(self.matlab_dir.rglob("*.m"))
        all_issues = []
        for f in m_files:
            all_issues.extend(self._analyze_matlab_file(f))

        self.results.update(
            {
                "total_files": len(m_files),
                "issues": all_issues,
                "passed": len(all_issues) == 0,
                "summary": f"{'[PASS]' if not all_issues else '[FAIL]'} Checked {len(m_files)} files",
            }
        )
        return self.results


def main() -> None:
    parser = argparse.ArgumentParser(description="MATLAB Quality Checker")
    parser.add_argument("--project-root", type=str, default=".")
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    checker = MATLABQualityChecker(root)
    results = checker.run_all_checks()

    for issue in cast(list[str], results["issues"]):
        logger.error(issue)

    sys.exit(0 if results["passed"] else 1)


if __name__ == "__main__":
    main()
