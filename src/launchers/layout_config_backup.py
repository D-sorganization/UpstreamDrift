"""Shared launcher layout backup helpers."""

from __future__ import annotations

from pathlib import Path


def replace_existing_layout_backup(config_file: Path) -> Path | None:
    """Move an existing layout config to its stable backup path.

    ``Path.replace`` overwrites an existing ``.bak`` file on Windows and POSIX,
    unlike ``Path.rename`` on Windows.
    """
    if not config_file.exists():
        return None
    backup_path = config_file.with_suffix(".json.bak")
    config_file.replace(backup_path)
    return backup_path
