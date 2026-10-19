"""Repo-hygiene guard: licensed_assets/ must stay ignored in .gitignore.

SimTK spine models and other licensed model files are symlinked or placed in
licensed_assets/ and must never be tracked in the repository (#13258).
"""

from __future__ import annotations

import shutil
import subprocess  # nosec B404 - git invocation, no shell, fixed args
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_GITIGNORE = _REPO_ROOT / ".gitignore"


def _gitignore_entries() -> set[str]:
    return {
        line.strip()
        for line in _GITIGNORE.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def test_gitignore_ignores_licensed_assets() -> None:
    """licensed_assets/ must stay ignored in root .gitignore (#13258)."""
    entries = _gitignore_entries()
    assert "licensed_assets/" in entries or "licensed_assets" in entries


def test_no_tracked_files_in_licensed_assets() -> None:
    """No file under licensed_assets/ may ever be committed to git."""
    git = shutil.which("git")
    if git is None or not (_REPO_ROOT / ".git").exists():
        pytest.skip("git checkout not available")
    result = subprocess.run(  # nosec B603 - fixed argv, no shell
        [git, "ls-files", "licensed_assets"],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    tracked = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    assert not tracked, f"Found tracked files in licensed_assets: {tracked}"
