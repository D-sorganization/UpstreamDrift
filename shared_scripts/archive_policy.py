#!/usr/bin/env python3
"""Archive-directory policy shared by the handoff and development-log validators.

Files under ``docs/development/archive/`` hold history moved out of the live
handoff and development log (Repository_Management#1893). They are exempt from
the size, entry-count and section-format budgets that keep the live files small,
but they stay fail-closed on basic sanity: valid UTF-8 Markdown, no binary data.

Portable: standard library only, copied fleet-wide next to ``handoff_validator.py``.
"""

from __future__ import annotations

from pathlib import Path

ARCHIVE_PARTS = ("docs", "development", "archive")
ARCHIVE_RELATIVE_DIR = "/".join(ARCHIVE_PARTS)


def is_archive_path(path: str | Path) -> bool:
    """True when ``path`` lies inside a ``docs/development/archive/`` directory."""
    parts = [p for p in str(path).replace("\\", "/").split("/") if p]
    # The last part is the file name; only directory parts can match.
    directories = parts[:-1]
    size = len(ARCHIVE_PARTS)
    return any(
        tuple(directories[i : i + size]) == ARCHIVE_PARTS
        for i in range(len(directories) - size + 1)
    )


def archive_files(repo_root: Path) -> list[Path]:
    """Every regular file under the repository's archive directory, sorted."""
    base = repo_root / ARCHIVE_RELATIVE_DIR
    if not base.is_dir():
        return []
    return sorted(p for p in base.rglob("*") if p.is_file())


def check_archive_file(path: Path) -> list[tuple[int | None, str, str]]:
    """Cheap sanity checks: ``(line, kind, message)`` tuples, empty when fine."""
    if path.suffix.lower() != ".md":
        return [(None, "archive_not_markdown", f"{path.name} is not a .md file.")]
    try:
        raw = path.read_bytes()
    except OSError as exc:
        return [(None, "archive_unreadable", f"Could not read {path.name}: {exc}")]
    if b"\x00" in raw:
        return [(None, "archive_binary", f"{path.name} contains NUL bytes.")]
    try:
        raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        return [(None, "archive_invalid_utf8", f"{path.name} is not UTF-8: {exc}")]
    return []
