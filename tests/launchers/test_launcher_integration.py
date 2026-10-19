#!/usr/bin/env python3
"""
Integration tests for Golf Modeling Suite launcher functionality.

Tests the actual functionality without complex PyQt6 mocking.
"""

import subprocess
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


class TestLauncherCommands(unittest.TestCase):
    """Test launcher command functionality."""

    def test_engine_launch_commands(self) -> None:
        """--engine choices are accepted by the canonical launcher CLI."""
        engines = ["mujoco", "drake", "pinocchio"]

        for engine in engines:
            with self.subTest(engine=engine):
                parse_result = subprocess.run(
                    [
                        sys.executable,
                        "-c",
                        (
                            "import launch_golf_suite as lg, sys; "
                            f"sys.argv = ['launch_golf_suite.py', '--engine', '{engine}']; "
                            "args = lg.parse_arguments(); "
                            f"assert args.engine == '{engine}'"
                        ),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    cwd=REPO_ROOT,
                )
                self.assertEqual(
                    parse_result.returncode,
                    0,
                    msg=parse_result.stderr,
                )
                self.assertNotIn("Traceback", parse_result.stderr)


if __name__ == "__main__":
    # Run tests with detailed output
    unittest.main(verbosity=2, buffer=True)
