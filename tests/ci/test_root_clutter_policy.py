from __future__ import annotations

from pathlib import Path

from scripts.check_root_clutter import ALLOWLIST

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_repo_root_contains_only_allowlisted_files() -> None:
    disallowed = sorted(
        entry.name
        for entry in REPO_ROOT.iterdir()
        if entry.is_file()
        and not entry.name.startswith(".")
        and entry.name not in ALLOWLIST
    )

    assert disallowed == []


def test_sidekick_pyinstaller_spec_is_documented_root_file() -> None:
    assert "sidekick.spec" in ALLOWLIST
