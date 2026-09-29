"""Smoke tests ensuring the docs/examples scripts stay runnable.

Each example under ``docs/examples/`` is an executable end-to-end script wired
into ``docs/examples/index.rst``. Running them here keeps the documentation
honest: if a public API changes and breaks an example, this test fails.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
EXAMPLES_DIR = REPO_ROOT / "docs" / "examples"

EXAMPLE_SCRIPTS = sorted(EXAMPLES_DIR.glob("*.py"))


def test_examples_directory_is_populated() -> None:
    """At least three runnable examples must exist (issue #7063)."""
    assert len(EXAMPLE_SCRIPTS) >= 3, (
        "docs/examples/ should contain >= 3 runnable example scripts; "
        f"found {len(EXAMPLE_SCRIPTS)}"
    )


@pytest.mark.unit
@pytest.mark.parametrize("script", EXAMPLE_SCRIPTS, ids=lambda p: p.name)
def test_example_runs_successfully(script: Path) -> None:
    """Each example script must execute and exit ``0``."""
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, (
        f"{script.name} failed (exit {result.returncode}):\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
