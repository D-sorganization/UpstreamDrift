"""Process Isolation Tests for Strict Physics Engine Protocols.

This module executes strict unit tests for aggressive engines (Drake, Pinocchio)
in separate Python processes to avoid 'numpy' corruption caused by incompatible
C-extension mocking/reloading within a single pytest session (Issue #496).
"""

import subprocess
import sys
from pathlib import Path

import pytest

# Paths to the isolated test files
ISOLATED_TESTS_DIR = Path(__file__).parent / "isolated"
TEST_DRAKE_STRICT = ISOLATED_TESTS_DIR / "test_drake_strict.py"
TEST_PINOCCHIO_STRICT = ISOLATED_TESTS_DIR / "test_pinocchio_strict.py"


class TestProcessIsolationStrict:
    """Run specific strict tests in isolated subprocesses."""

    def run_isolated_test(self, test_file: Path):
        """Helper to run pytest on a single file in a subprocess."""
        cmd = [sys.executable, "-m", "pytest", str(test_file), "-v", "--no-cov"]

        # Capture output to help debugging if it fails
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,  # We check returncode manually for better error reporting
        )

        if result.returncode != 0:
            pytest.fail(
                f"Isolated test {test_file.name} failed with exit code {result.returncode}.\n"
                f"--- STDOUT ---\n{result.stdout}\n"
                f"--- STDERR ---\n{result.stderr}"
            )

    def test_drake_strict_isolated(self):
        """Run Drake strict tests in an isolated process to prevent numpy corruption."""
        if not TEST_DRAKE_STRICT.exists():
            pytest.fail(f"Test file not found: {TEST_DRAKE_STRICT}")
        self.run_isolated_test(TEST_DRAKE_STRICT)

    def test_pinocchio_strict_isolated(self):
        """Run Pinocchio strict tests in an isolated process to prevent numpy corruption."""
        if not TEST_PINOCCHIO_STRICT.exists():
            pytest.fail(f"Test file not found: {TEST_PINOCCHIO_STRICT}")
        self.run_isolated_test(TEST_PINOCCHIO_STRICT)
