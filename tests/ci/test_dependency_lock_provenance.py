"""Contracts for deterministic pip-tools lockfile provenance (issue #9120)."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
CANONICAL_COMPILE_COMMAND = "make sync-deps"
LOCKFILES = ("requirements.lock", "requirements-dev.lock")
pytestmark = pytest.mark.unit


def _sync_deps_dry_run(*, offline: bool) -> bytes:
    """Return the repository recipe without executing dependency resolution."""

    environment = os.environ.copy()
    for variable in ("PIP_FIND_LINKS", "PIP_INDEX_URL", "PIP_NO_INDEX"):
        environment.pop(variable, None)
    if offline:
        environment.update(
            {
                "PIP_FIND_LINKS": "/nonexistent/offline-wheelhouse",
                "PIP_INDEX_URL": "https://invalid.example.invalid/simple",
                "PIP_NO_INDEX": "1",
            }
        )

    result = subprocess.run(
        ["make", "--no-print-directory", "--dry-run", "sync-deps"],
        cwd=REPO_ROOT,
        env=environment,
        check=True,
        capture_output=True,
    )
    return result.stdout


def test_sync_deps_recipe_is_environment_invariant() -> None:
    normal = _sync_deps_dry_run(offline=False)
    offline = _sync_deps_dry_run(offline=True)

    assert offline == normal
    assert normal.count(b'CUSTOM_COMPILE_COMMAND="make sync-deps"') == 2


def test_committed_lockfiles_use_canonical_generated_header() -> None:
    for relative_path in LOCKFILES:
        header = (REPO_ROOT / relative_path).read_bytes().splitlines()[:8]
        assert b"#    make sync-deps" in header, relative_path
