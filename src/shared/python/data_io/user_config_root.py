"""Single per-user configuration root for UpstreamDrift (issue #8907).

Before #8907 the launcher persisted user state under several independent
roots (``~/.golf_modeling_suite/`` -- a dead product name --,
``~/.upstreamdrift/`` and ``platformdirs("upstream-drift")/launcher``).
This module owns the one canonical root and the one-time, idempotent
migration of launcher-owned files out of the two legacy directories.

Every launcher writer resolves its path through :func:`user_config_path`
so no module does its own ``Path.home() / ...`` math.

Migration semantics
-------------------
* Only the launcher-owned names in :data:`LEGACY_MIGRATIONS` are copied.
  Legacy directories also hold files owned by other subsystems (chat
  sessions, anthropometric subjects, the Tools-owned
  ``mcp_servers.json`` contract) whose readers still point at the legacy
  location, so the legacy files are *copied*, never moved or deleted.
* An existing file in the new root is never overwritten.
* A marker file is written once every item succeeded; later runs are
  no-ops. A failed item leaves no marker so the next start retries.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path, PurePath

from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)

APP_DIR_NAME = "upstream-drift"
LAUNCHER_SUBDIR = "launcher"
MIGRATION_MARKER = ".legacy_dirs_migrated"

#: Legacy directory (relative to ``$HOME``) -> launcher-owned names to copy.
LEGACY_MIGRATIONS: dict[str, tuple[str, ...]] = {
    ".golf_modeling_suite": (
        "preferences.json",
        "recent_models.json",
        "library",
        "process_output.log",
        "launcher.log",
    ),
    ".upstreamdrift": ("onboarding_config.json",),
}


def user_config_dir() -> Path:
    """Return the canonical per-user launcher config directory.

    * Linux/macOS: ``~/.config/upstream-drift/launcher`` (via platformdirs)
    * Windows: ``%LOCALAPPDATA%/upstream-drift/.../launcher``

    The directory is not created; writers create it on first write.

    DbC postcondition: the result is absolute and is not a legacy root.
    """
    try:
        from platformdirs import user_config_dir as _platform_config_dir

        root = Path(_platform_config_dir(APP_DIR_NAME)) / LAUNCHER_SUBDIR
    except ImportError:
        # Graceful fallback if platformdirs is somehow unavailable at runtime
        if sys.platform == "win32":
            root = Path.home() / "AppData" / "Local" / "UpstreamDrift"
        else:
            root = Path.home() / ".config" / APP_DIR_NAME
        root = root / LAUNCHER_SUBDIR
    root = root.absolute()  # e.g. a drive-less "/tmp" on Windows
    assert root.is_absolute(), f"config root must be absolute: {root}"
    assert not set(root.parts) & set(LEGACY_MIGRATIONS), root
    return root


def user_config_path(*parts: str) -> Path:
    """Return ``user_config_dir() / parts`` -- the one path helper for writers.

    Raises:
        ValueError: if no part is given, or a part is absolute or climbs
            out of the root with ``..``.
    """
    if not parts:
        raise ValueError("user_config_path() needs at least one path part")
    relative = PurePath(*parts)
    if relative.anchor or ".." in relative.parts:
        raise ValueError(f"config path must stay inside the root: {relative}")
    return user_config_dir().joinpath(*parts)


def _copy_item(source: Path, dest: Path) -> None:
    """Copy one file or directory tree, never replacing an existing *dest*."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        shutil.copytree(source, dest)
    else:
        shutil.copy2(source, dest)


def _legacy_sources(home: Path) -> list[Path]:
    """Return the migratable legacy items that currently exist under *home*."""
    return [
        home / legacy_name / name
        for legacy_name, names in LEGACY_MIGRATIONS.items()
        for name in names
        if (home / legacy_name / name).exists()
    ]


def migrate_legacy_user_dirs(new_root: Path, home: Path | None = None) -> list[Path]:
    """Copy launcher-owned files from the legacy roots into *new_root* once.

    Args:
        new_root: The canonical root (normally :func:`user_config_dir`).
        home: Directory holding the legacy dot-directories; defaults to
            ``Path.home()``. Injectable for tests.

    Returns:
        The destination paths created by this call (empty when there was
        nothing to do or the marker was already present).

    Raises:
        ValueError: if *new_root* or *home* is not an absolute path.

    DbC postcondition: when the marker is written, every migratable legacy
    item has a counterpart in *new_root*, which is therefore authoritative.
    """
    home = Path.home() if home is None else home
    for label, value in (("new_root", new_root), ("home", home)):
        if not isinstance(value, Path) or not value.is_absolute():
            raise ValueError(f"{label} must be an absolute Path, got {value!r}")
    marker = new_root / MIGRATION_MARKER
    sources = _legacy_sources(home)
    if marker.exists() or not sources:
        return []

    created: list[Path] = []
    failed = False
    for source in sources:
        dest = new_root / source.name
        if dest.exists():
            continue  # never clobber: the new root is authoritative
        try:
            _copy_item(source, dest)
        except OSError as exc:
            failed = True
            logger.warning("Could not migrate %s to %s: %s", source, dest, exc)
            continue
        created.append(dest)
        logger.info("Migrated legacy user config %s -> %s", source, dest)

    if failed:
        return created  # no marker: retry on the next start
    pending = [s for s in sources if not (new_root / s.name).exists()]
    assert not pending, f"migration left items behind: {pending}"
    try:
        new_root.mkdir(parents=True, exist_ok=True)
        marker.write_text("issue #8907\n", encoding="utf-8")
    except OSError as exc:
        logger.warning("Could not write migration marker %s: %s", marker, exc)
    return created


__all__ = [
    "LEGACY_MIGRATIONS",
    "MIGRATION_MARKER",
    "migrate_legacy_user_dirs",
    "user_config_dir",
    "user_config_path",
]
