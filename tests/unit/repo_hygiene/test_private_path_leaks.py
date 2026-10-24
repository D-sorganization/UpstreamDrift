"""Repo-hygiene guard for private path / hostname leak ratchet."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CHECKER = _REPO_ROOT / "scripts" / "ci" / "check_private_path_leaks.py"
_PATTERNS = _REPO_ROOT / "scripts" / "config" / "private_path_guard_patterns.json"


def test_guard_patterns_file_is_well_formed() -> None:
    data = json.loads(_PATTERNS.read_text(encoding="utf-8"))
    assert data.get("patterns"), "at least one committed pattern is required"
    for entry in data["patterns"]:
        assert entry.get("id")
        assert entry.get("regex")


def test_private_path_leak_ratchet_passes() -> None:
    """CI entrypoint must exit 0 at or below the recorded baseline."""
    if not _CHECKER.is_file():
        pytest.skip("checker not present on this ref")
    completed = subprocess.run(
        [sys.executable, str(_CHECKER)],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
