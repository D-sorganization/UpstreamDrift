"""Test for repo hygiene: vendor submodule must be clean."""

import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent.parent
VENDOR_DIR = ROOT / "vendor" / "ud-tools"


def test_vendor_submodule_clean() -> None:
    if not VENDOR_DIR.exists():
        pytest.skip("Vendor submodule not checked out")

    result = subprocess.run(
        ["git", "status", "--porcelain=v2"],
        cwd=str(VENDOR_DIR),
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        pytest.skip("Git is not available or vendor is not a git repository")

    # If there's any output, the submodule is not clean
    output = result.stdout.strip()
    if output:
        pytest.fail(
            f"vendor/ud-tools has uncommitted changes:\n{output}\n"
            "Working-tree edits inside the submodule are forbidden in PRs."
        )
