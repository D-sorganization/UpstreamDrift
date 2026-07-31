"""Regression tests to ensure all examples produce expected output.

This module verifies that:
- All examples exit successfully (code 0)
- Output examples produce non-empty output (not silent failures)
- Output examples contain expected unit suffixes (m, yd, N, kg, s, etc.)
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

SKIP_EXAMPLES = {"__init__.py"}  # not an example
CLI_OUTPUT_EXAMPLES = {
    "basic_flight_simulation.py"
}  # examples that must produce output
REQUIRED_UNIT_SUFFIXES = {"m", "yd", "N", "kg", "s", "rad", "rpm"}


def get_example_files() -> list[Path]:
    """Get all example files in the examples/ directory."""
    examples_dir = Path(__file__).parent.parent.parent / "examples"
    if not examples_dir.exists():
        return []
    return sorted(
        py_file
        for py_file in examples_dir.glob("*.py")
        if py_file.name not in SKIP_EXAMPLES
    )


@pytest.mark.parametrize("example_file", get_example_files(), ids=lambda p: p.name)
def test_example_produces_output(example_file: Path) -> None:
    """Test that an example runs successfully and produces output (if documented).

    CLI output examples (basic_flight_simulation.py) must produce output with units.
    Run-only demos (aerodynamics_demo.py, etc.) just need to exit successfully.
    """
    repo_root = Path(__file__).parent.parent.parent
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root)

    result = subprocess.run(
        [sys.executable, str(example_file)],
        capture_output=True,
        text=True,
        timeout=30,
        cwd=repo_root,
        env=env,
    )

    assert result.returncode == 0, (
        f"{example_file.name} failed with exit code {result.returncode}. "
        f"stderr: {result.stderr}"
    )

    # Only CLI output examples must produce output
    if example_file.name in CLI_OUTPUT_EXAMPLES:
        output = result.stdout + result.stderr
        assert (
            output.strip()
        ), f"{example_file.name} produced no output (silent failure)"

        has_unit = any(suffix in output for suffix in REQUIRED_UNIT_SUFFIXES)
        assert has_unit, (
            f"{example_file.name} output missing unit suffixes. "
            f"Expected at least one of: {REQUIRED_UNIT_SUFFIXES}. "
            f"Got: {output[:200]}"
        )
