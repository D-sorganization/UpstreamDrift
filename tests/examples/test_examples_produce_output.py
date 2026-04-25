"""Regression tests to ensure all examples produce expected output.

This module verifies that:
- All examples exit successfully (code 0)
- Output is non-empty (not silent failures)
- Output contains expected unit suffixes (m, yd, N, kg, s, etc.)
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


SKIP_EXAMPLES = {"__init__.py"}  # not an example
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
    """Test that an example runs successfully and produces non-empty output."""
    result = subprocess.run(
        ["python3", str(example_file)],
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, (
        f"{example_file.name} failed with exit code {result.returncode}. "
        f"stderr: {result.stderr}"
    )

    output = result.stdout + result.stderr
    assert output.strip(), f"{example_file.name} produced no output (silent failure)"

    has_unit = any(suffix in output for suffix in REQUIRED_UNIT_SUFFIXES)
    assert has_unit, (
        f"{example_file.name} output missing unit suffixes. "
        f"Expected at least one of: {REQUIRED_UNIT_SUFFIXES}. "
        f"Got: {output[:200]}"
    )


@pytest.mark.unit
def test_basic_flight_simulation_output() -> None:
    """Regression test for basic_flight_simulation.py output."""
    example = (
        Path(__file__).parent.parent.parent / "examples" / "basic_flight_simulation.py"
    )
    if not example.exists():
        pytest.skip(f"Example file not found: {example}")

    result = subprocess.run(
        ["python3", str(example)],
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, f"Example failed: {result.stderr}"
    output = result.stdout

    assert "t (s)" in output, "Missing trajectory time column header"
    assert "x (m)" in output, "Missing trajectory x-position header"
    assert "z (m)" in output, "Missing trajectory height header"
    assert "|v| (m/s)" in output, "Missing velocity column header"
    assert "Carry distance:" in output, "Missing carry distance output"
    assert "m (" in output and "yd)" in output, "Missing meters/yards output"
    assert "Physics:" in output, "Missing physics explanation"
    assert "drag" in output.lower(), "Physics output should mention drag"
    assert "lift" in output.lower(), "Physics output should mention lift"
