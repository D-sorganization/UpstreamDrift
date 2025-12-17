#!/usr/bin/env python3
"""
MATLAB Quality Check Script

This script runs comprehensive quality checks on MATLAB code following the project's
.cursorrules.md requirements. It can be run from the command line and integrates
with the project's quality control system.

Usage:
    python scripts/matlab_quality_check.py [--strict] [--output-format json|text]
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import ClassVar  # noqa: ICN003

# Constants
# 300 seconds (5 minutes) - allows time for large codebases
MATLAB_SCRIPT_TIMEOUT_SECONDS: int = 300
MIN_DOCSTRING_LENGTH: int = 3  # Minimum length for a valid docstring comment

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class MATLABQualityChecker:
    """Comprehensive MATLAB code quality checker."""

    # Class constants for magic number checking
    ACCEPTABLE_NUMBERS: ClassVar[set[str]] = {
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
        "100",
        "100.0",
        "1000",
        "1000.0",
        "0.5",
        "0.1",
        "0.01",
        "0.001",
        "0.0001",  # Common tolerances
    }

    KNOWN_CONSTANTS: ClassVar[dict[str, str]] = {
        "3.14159": "pi constant [dimensionless] - mathematical constant",
        "3.1416": "pi constant [dimensionless] - mathematical constant",
        "3.14": "pi constant [dimensionless] - mathematical constant",
        "1.5708": "pi/2 constant [dimensionless] - mathematical constant",
        "1.57": "pi/2 constant [dimensionless] - mathematical constant",
        "0.7854": "pi/4 constant [dimensionless] - mathematical constant",
        "0.785": "pi/4 constant [dimensionless] - mathematical constant",
        "9.81": "gravitational acceleration [m/s²] - approximate standard gravity",
        "9.8": "gravitational acceleration [m/s²] - approximate standard gravity",
        "9.807": "gravitational acceleration [m/s²] - approximate standard gravity",
    }

    # Pre-compile regex for magic numbers
    MAGIC_NUMBER_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"(?<![.\w])(?:\d+\.\d+|\d+)(?![.\w])",
    )

    def __init__(self, project_root: Path) -> None:
        """Initialize the MATLAB quality checker.

        Args:
            project_root: Path to the project root directory
        """
        self.project_root = project_root
        self.matlab_dir = project_root / "matlab"
        self.matlab_optimized_dir = project_root / "matlab_optimized"
        self.results = {
            "timestamp": datetime.now(UTC).isoformat(),
            "total_files": 0,
            "issues": [],
            "passed": True,
            "summary": "",
            "checks": {},
        }

    def check_matlab_files_exist(self) -> bool:
        """Check if MATLAB files exist in the project.

        Returns:
            True if MATLAB files are found, False otherwise
        """
        m_files: list[Path] = []
        if self.matlab_dir.exists():
            m_files.extend(self.matlab_dir.rglob("*.m"))
        if self.matlab_optimized_dir.exists():
            m_files.extend(self.matlab_optimized_dir.rglob("*.m"))

        self.results["total_files"] = len(m_files)

        if len(m_files) == 0:
            logger.info("No MATLAB files found (skipping MATLAB checks)")
            return False

        logger.info("Found %d MATLAB files", len(m_files))
        return True

    def run_matlab_quality_checks(self) -> dict[str, object]:  # noqa: PLR0911
        """Run MATLAB quality checks using the MATLAB script.

        Returns:
            Dictionary containing quality check results
        """
        try:
            # Check if we can run MATLAB from command line
            matlab_script = self.matlab_dir / "matlab_quality_config.m"
            if not matlab_script.exists():
                # Config script not found - fall back to static analysis (primary use case)
                logger.info(
                    "MATLAB quality config script not found, using static analysis",
                )
                return self._static_matlab_analysis()

            # Try to run MATLAB quality checks
            # Note: This requires MATLAB to be installed and accessible from command line
            try:
                # First, try to run the MATLAB script directly if possible
                return self._run_matlab_script(matlab_script)
            except (subprocess.SubprocessError, OSError) as e:
                logger.warning("Could not run MATLAB script directly: %s", e)
                # Fall back to static analysis
                return self._static_matlab_analysis()

        except (OSError, ValueError) as e:
            logger.exception("Error running MATLAB quality checks")
            return {"error": str(e)}

    def _run_matlab_script(self, script_path: Path) -> dict[str, object]:
        """Attempt to run MATLAB script from command line.

        Args:
            script_path: Path to the MATLAB script

        Returns:
            Dictionary containing script results
        """
        try:
            # Try different ways to run MATLAB
            commands = [
                ["matlab", "-batch", f"run('{script_path}')"],
                [
                    "matlab",
                    "-nosplash",
                    "-nodesktop",
                    "-batch",
                    f"run('{script_path}')",
                ],
                ["octave", "--no-gui", "--eval", f"run('{script_path}')"],
            ]

            for cmd in commands:
                try:
                    logger.info("Trying command: %s", " ".join(cmd))
                    # Note: cmd is a controlled list from our code, not user input (S603)
                    result = subprocess.run(  # noqa: S603
                        cmd,
                        capture_output=True,
                        text=True,
                        cwd=self.matlab_dir,
                        timeout=MATLAB_SCRIPT_TIMEOUT_SECONDS,
                        check=False,
                    )

                    if result.returncode == 0:
                        logger.info("MATLAB quality checks completed successfully")
                        return {
                            "success": True,
                            "output": result.stdout,
                            "method": "matlab_script",
                            "passed": True,
                        }
                    logger.warning(
                        "Command failed with return code %d",
                        result.returncode,
                    )
                    logger.debug("stderr: %s", result.stderr)

                except (subprocess.TimeoutExpired, FileNotFoundError):
                    continue

            # If all commands fail, fall back to static analysis
            logger.info("All MATLAB commands failed, falling back to static analysis")
            return self._static_matlab_analysis()

        except (OSError, ValueError) as e:
            logger.exception("Error running MATLAB script")
            return {"error": str(e)}

    def _static_matlab_analysis(self) -> dict[str, object]:
        """Perform static analysis of MATLAB files without running MATLAB.

        Returns:
            Dictionary containing static analysis results
        """
        logger.info("Performing static MATLAB file analysis")

        issues = []
        total_files = 0

        # Analyze each MATLAB file in both directories
        for matlab_path in [self.matlab_dir, self.matlab_optimized_dir]:
            if matlab_path.exists():
                for m_file in matlab_path.rglob("*.m"):
                    total_files += 1
                    file_issues = self._analyze_matlab_file(m_file)
                    issues.extend(file_issues)

        self.results["total_files"] = total_files
        self.results["issues"] = issues
        self.results["passed"] = len(issues) == 0

        return {
            "success": True,
            "method": "static_analysis",
            "total_files": total_files,
            "issues": issues,
            "passed": len(issues) == 0,
        }

    def _analyze_matlab_file(  # noqa: C901, PLR0912, PLR0915
        self,
        file_path: Path,
    ) -> list[str]:
        """Analyze a single MATLAB file for quality issues.

        Args:
            file_path: Path to the MATLAB file

        Returns:
            List of quality issues found
        """
        issues = []

        try:
            with file_path.open(encoding="utf-8", errors="ignore") as f:
                content = f.read()
                lines = content.split("\n")

            # Track if we're in a function and nesting level
            # Use separate function nesting to handle classdef files correctly
            in_function = False
            nesting_level = 0
            function_nesting_level = 0  # Track function nesting separately

            # Check for basic quality issues
            for i, line in enumerate(lines, 1):
                line_stripped = line.strip()

                # Skip empty lines
                if not line_stripped:
                    continue

                # Skip comment-only lines for most checks (but check comments for banned patterns)
                is_comment = line_stripped.startswith("%")

                # Track function scope by monitoring nesting level and function entry/exit
                if not is_comment:
                    # Check for keywords that increase nesting
                    # Note: arguments, properties, methods, events also have 'end'
                    if re.match(
                        r"\b(function|if|for|while|switch|try|parfor|classdef|arguments|properties|methods|events)\b",
                        line_stripped,
                    ):
                        if line_stripped.startswith("function"):
                            in_function = True
                            function_nesting_level += 1
                        nesting_level += 1

                    # Check for 'end' keyword that decreases nesting
                    if re.match(r"\bend\b", line_stripped):
                        nesting_level -= 1
                        # Track function exit separately from general nesting level
                        # In classdef files, nesting_level may not reach zero when exiting
                        # a function, but function_nesting_level tracks function-specific
                        # nesting
                        if function_nesting_level > 0:
                            function_nesting_level -= 1
                            if function_nesting_level == 0:
                                in_function = False

                        if nesting_level <= 0:
                            nesting_level = 0  # Prevent negative nesting
                            # Also reset function tracking if we've exited all nesting
                            in_function = False
                            function_nesting_level = 0

                # Check for function definition (for docstring and arguments validation)
                # Use word boundary to match MATLAB keyword, not just prefix
                if re.match(r"\bfunction\b", line_stripped) and not is_comment:
                    # Check if next non-empty line has docstring
                    # Handle multi-line function definitions with ... continuation
                    has_docstring = False

                    # Track continuation for docstring check
                    func_line_code_ds = line_stripped.split("%")[0].strip()
                    in_continuation_ds = func_line_code_ds.endswith("...")

                    for j in range(i, min(i + 10, len(lines))):
                        next_line = lines[j].strip()

                        # Handle continuation
                        if in_continuation_ds:
                            next_line_code = next_line.split("%")[0].strip()
                            in_continuation_ds = next_line_code.endswith("...")
                            continue

                        # Skip continuation lines (...)
                        if next_line == "...":
                            continue

                        # Break on non-comment, non-continuation code lines
                        if next_line and not next_line.startswith("%") and next_line != "...":
                            break
                        if (
                            next_line.startswith("%")
                            and len(next_line) > MIN_DOCSTRING_LENGTH
                        ):
                            has_docstring = True
                            break

                    if not has_docstring:
                        issues.append(
                            f"{file_path.name} (line {i}): Missing function docstring",
                        )

                    # Check for arguments validation block
                    # Skip comment lines to avoid false positives
                    # Arguments block must appear immediately after function signature
                    has_arguments = False

                    # Check if function definition line ends with continuation
                    # Strip comment first
                    func_line_code = line_stripped.split("%")[0].strip()
                    in_continuation = func_line_code.endswith("...")

                    for j in range(i, min(i + 50, len(lines))):
                        line_check = lines[j].strip()

                        # Handle continuation from previous line
                        if in_continuation:
                            line_code = line_check.split("%")[0].strip()
                            in_continuation = line_code.endswith("...")
                            continue

                        # Skip comment lines
                        if line_check.startswith("%"):
                            continue
                        # Skip continuation lines (...)
                        if line_check == "...":
                            continue
                        # Break on non-empty, non-comment code lines (arguments must be immediate)
                        if line_check and not line_check.startswith("%"):
                            # Check if 'arguments' is at start of line (MATLAB keyword requirement)
                            # Line is already stripped, so re.match anchors to start automatically
                            # Word boundary ensures it's not part of a larger word
                            # (e.g., data.arguments)
                            if re.match(r"arguments\b", line_check):
                                has_arguments = True
                            break

                    if not has_arguments:
                        issues.append(
                            f"{file_path.name} (line {i}): Missing arguments validation block",
                        )

                # Check for banned patterns (in comments and code)
                banned_patterns = [
                    (r"\bTODO\b", "TODO placeholder found"),
                    (r"\bFIXME\b", "FIXME placeholder found"),
                    (r"\bHACK\b", "HACK comment found"),
                    (r"\bXXX\b", "XXX comment found"),
                    (r"<[A-Z_][A-Z0-9_]*>", "Angle bracket placeholder found"),
                    (r"\{\{.*?\}\}", "Template placeholder found"),
                ]

                for pattern, message in banned_patterns:
                    if re.search(pattern, line_stripped):
                        issues.append(f"{file_path.name} (line {i}): {message}")

                # Skip further checks for comment lines
                if is_comment:
                    continue

                # Check for common MATLAB anti-patterns
                if re.search(r"\beval\s*\(", line_stripped):
                    issues.append(
                        f"{file_path.name} (line {i}): Avoid using eval() - "
                        "potential security risk and performance issue",
                    )

                if re.search(r"\bassignin\s*\(", line_stripped):
                    issues.append(
                        f"{file_path.name} (line {i}): "
                        "Avoid using assignin() - violates encapsulation",
                    )

                if re.search(r"\bevalin\s*\(", line_stripped):
                    issues.append(
                        f"{file_path.name} (line {i}): "
                        "Avoid using evalin() - violates encapsulation",
                    )

                # Check for global variables (often code smell)
                if re.search(r"\bglobal\s+\w+", line_stripped):
                    issues.append(
                        f"{file_path.name} (line {i}): "
                        "Global variable usage - consider passing as argument",
                    )

                # Check for load without output (loads into workspace)
                # Match both command syntax (load file.mat) and function syntax (load('file.mat'))
                if (
                    re.search(r"^\s*load\s+\w+", line_stripped)
                    or re.search(r"^\s*load\s*\([^)]+\)", line_stripped)
                ) and "=" not in line_stripped:
                    issues.append(
                        f"{file_path.name} (line {i}): load without output variable - "
                        "use 'data = load(...)' instead",
                    )

                # Check for magic numbers (but allow common values and known constants)
                # Matches both integer and floating-point literals (e.g., 3.14, 42, 0.5)
                # that are not part of scientific notation, array indices, or embedded in words.
                # Uses lookbehind/lookahead to avoid matching numbers adjacent to dots or
                # word characters. This helps flag "magic numbers" in code while avoiding
                # false positives from common patterns.

                # Check if there is a comment and where it starts
                # Note: This is a simple check and doesn't account for % inside strings
                # but matches the original implementation's behavior
                comment_idx = line_stripped.find("%")

                # Track reported numbers for this line to avoid duplicates
                reported_nums = set()

                for match in self.MAGIC_NUMBER_PATTERN.finditer(line_stripped):
                    num = match.group()

                    # If already reported for this line, skip
                    if num in reported_nums:
                        continue

                    # Check if match is in comment
                    if comment_idx != -1 and match.start() >= comment_idx:
                        continue

                    reported_nums.add(num)

                    # Check if it's a known constant
                    if num in self.KNOWN_CONSTANTS:
                        issues.append(
                            f"{file_path.name} (line {i}): Magic number {num} "
                            f"({self.KNOWN_CONSTANTS[num]}) - define as named constant",
                        )
                    elif num not in self.ACCEPTABLE_NUMBERS:
                        issues.append(
                            f"{file_path.name} (line {i}): Magic number {num} "
                            "should be defined as constant with units and source",
                        )

                # Check for clear/clc/close all in functions (bad practice)
                if in_function:
                    # Check for clear without variable (dangerous) or clear all/global
                    # (very dangerous)
                    if re.search(
                        r"\bclear\s+(all|global)\b",
                        line_stripped,
                        re.IGNORECASE,
                    ):
                        issues.append(
                            f"{file_path.name} (line {i}): Avoid 'clear all' or "
                            "'clear global' in functions - clears all variables, "
                            "functions, and MEX links",
                        )
                    elif re.search(r"\bclear\b(?!\s+\w+)", line_stripped):
                        issues.append(
                            f"{file_path.name} (line {i}): Avoid 'clear' in functions - "
                            "can clear function variables",
                        )
                    if re.search(r"\bclc\b", line_stripped):
                        issues.append(
                            f"{file_path.name} (line {i}): Avoid 'clc' in functions - "
                            "affects user's workspace",
                        )
                    if re.search(r"\bclose\s+all\b", line_stripped):
                        issues.append(
                            f"{file_path.name} (line {i}): Avoid 'close all' in functions - "
                            "closes user's figures",
                        )

                # Check for exist() usage (often code smell, prefer try/catch or validation)
                if re.search(r"\bexist\s*\(", line_stripped):
                    issues.append(
                        f"{file_path.name} (line {i}): Consider using validation or "
                        "try/catch instead of exist()",
                    )

                # Check for addpath in functions (should be in startup.m or managed externally)
                if in_function and re.search(r"\baddpath\s*\(", line_stripped):
                    issues.append(
                        f"{file_path.name} (line {i}): Avoid addpath in functions - "
                        "manage paths externally",
                    )

        except (OSError, UnicodeDecodeError, ValueError) as e:
            issues.append(f"{file_path.name}: Could not analyze file - {e!s}")

        return issues

    def run_all_checks(self) -> dict[str, object]:
        """Run all MATLAB quality checks.

        Returns:
            Dictionary containing all quality check results
        """
        logger.info("Starting MATLAB quality checks")

        # Check if MATLAB files exist
        if not self.check_matlab_files_exist():
            self.results["passed"] = True
            self.results["summary"] = "[SKIP] No MATLAB files to check - passed"
            return self.results

        # Run MATLAB quality checks
        matlab_results = self.run_matlab_quality_checks()

        if "error" in matlab_results:
            self.results["passed"] = False
            self.results["summary"] = (
                f"MATLAB quality checks failed: {matlab_results['error']}"
            )
            # Type ignore: dict[str, object] allows string keys with object values
            self.results["checks"]["matlab"] = matlab_results  # type: ignore[index]
        else:
            # Type ignore: dict[str, object] allows string keys with object values
            self.results["checks"]["matlab"] = matlab_results  # type: ignore[index]
            if matlab_results.get("passed", False):
                self.results["summary"] = (
                    f"[PASS] MATLAB quality checks PASSED "
                    f"({self.results['total_files']} files checked)"
                )
            else:
                self.results["passed"] = False
                self.results["summary"] = (
                    f"[FAIL] MATLAB quality checks FAILED "
                    f"({self.results['total_files']} files checked)"
                )

        return self.results


def main() -> None:  # noqa: PLR0915
    """Main entry point for the MATLAB quality check script."""
    parser = argparse.ArgumentParser(description="MATLAB Code Quality Checker")
    parser.add_argument("--strict", action="store_true", help="Enable strict mode")
    parser.add_argument(
        "--output-format",
        choices=["json", "text"],
        default="text",
        help="Output format (default: text)",
    )
    parser.add_argument(
        "--project-root",
        type=str,
        default=".",
        help="Project root directory (default: current directory)",
    )

    args = parser.parse_args()

    # Get project root
    project_root = Path(args.project_root).resolve()
    if not project_root.exists():
        logger.error("Project root does not exist: %s", project_root)
        sys.exit(1)

    # Initialize and run quality checks
    checker = MATLABQualityChecker(project_root)
    results = checker.run_all_checks()

    # Output results
    if args.output_format == "json":
        print(json.dumps(results, indent=2, default=str))  # noqa: T201
    else:
        print("\n" + "=" * 60)  # noqa: T201
        print("MATLAB QUALITY CHECK RESULTS")  # noqa: T201
        print("=" * 60)  # noqa: T201
        print(f"Timestamp: {results.get('timestamp', 'N/A')}")  # noqa: T201
        print(f"Total Files: {results.get('total_files', 0)}")  # noqa: T201
        print(  # noqa: T201
            f"Status: {'PASSED' if results.get('passed', False) else 'FAILED'}",
        )
        print(f"Summary: {results.get('summary', 'N/A')}")  # noqa: T201

        issues_raw = results.get("issues", [])
        issues: list[str] = issues_raw if isinstance(issues_raw, list) else []
        if issues:
            print(f"\nIssues Found ({len(issues)}):")  # noqa: T201
            for i, issue in enumerate(issues, 1):
                print(f"  {i}. {issue}")  # noqa: T201

        print("\n" + "=" * 60)  # noqa: T201

    # Exit with appropriate code
    # In strict mode, fail if any issues are found; otherwise fail only if checks didn't pass
    passed = bool(results.get("passed", False))
    has_issues = bool(results.get("issues"))

    exit_code = (
        (0 if (passed and not has_issues) else 1)
        if args.strict
        else (0 if passed else 1)
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
