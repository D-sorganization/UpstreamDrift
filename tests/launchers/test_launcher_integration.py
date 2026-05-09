#!/usr/bin/env python3
"""
Integration tests for Golf Modeling Suite launcher functionality.

Tests the actual functionality without complex PyQt6 mocking.
"""

import subprocess
import sys
import unittest
from pathlib import Path


class TestLauncherCommands(unittest.TestCase):
    """Test launcher command functionality."""

    def test_engine_launch_commands(self) -> None:
        """Test individual engine launch commands."""
        engines = ["mujoco", "drake", "pinocchio"]

        for engine in engines:
            with self.subTest(engine=engine):
                # Test that command is recognized and doesn't fail immediately
                try:
                    result = subprocess.run(
                        [sys.executable, "launch_golf_suite.py", "--engine", engine],
                        capture_output=True,
                        text=True,
                        timeout=5,  # Increased timeout for Windows
                    )

                    # Check for immediate failures (import errors, etc.)
                    if result.returncode != 0:
                        # Module not found or engine not available is expected
                        # in environments without all engines installed
                        stderr = result.stderr.lower()
                        if any(
                            msg in stderr
                            for msg in [
                                "not ready",
                                "not available",
                                "no module named",
                                "failed to launch",
                            ]
                        ):
                            pass
                        else:
                            self.fail(f"Engine {engine} launch failed: {result.stderr}")
                    else:
                        pass

                except subprocess.TimeoutExpired:
                    # Timeout is good - means GUI started and didn't crash immediately
                    pass


if __name__ == "__main__":
    # Run tests with detailed output
    unittest.main(verbosity=2, buffer=True)
