"""Regression tests for the root `.gitignore` env-file contract.

Issue: #5920
"""

from __future__ import annotations

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_GITIGNORE = _REPO_ROOT / ".gitignore"


def _gitignore_entries() -> set[str]:
    return {
        line.strip()
        for line in _GITIGNORE.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def test_gitignore_ignores_env_files() -> None:
    """Secrets-bearing env files must stay ignored at the repo root."""
    entries = _gitignore_entries()
    assert ".env" in entries
    assert ".env.*" in entries


def test_gitignore_keeps_example_env_files_trackable() -> None:
    """Documented example env files must remain unignored for onboarding."""
    entries = _gitignore_entries()
    assert "!.env.example" in entries
    assert "!.env.docker.example" in entries
    assert "!.env.*.example" in entries
