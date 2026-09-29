"""Smoke tests for the upstream-physics Rust crate artifact."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parents[3]
CRATE_DIR = REPO_ROOT / "rust_core" / "upstream-physics"
pytestmark = pytest.mark.smoke


def test_crate_artifact_metadata_is_valid() -> None:
    """Cargo must accept the release crate metadata before publishing."""
    import json

    result = subprocess.run(
        ["cargo", "metadata", "--format-version", "1", "--no-deps"],
        cwd=CRATE_DIR,
        check=True,
        capture_output=True,
        text=True,
    )
    metadata = json.loads(result.stdout)
    assert "packages" in metadata, "Cargo metadata should contain a 'packages' key"
    assert len(metadata["packages"]) > 0, (
        "Cargo metadata should list at least one package"
    )
