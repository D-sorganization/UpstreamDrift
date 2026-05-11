"""Code-map cold-rebuild and incremental update indexer.

Usage (programmatic)
--------------------
    from shared.python.codemap.indexer import CodeMapIndex, rebuild

    idx = CodeMapIndex(repo_root=Path("."))
    idx.rebuild()          # full cold rebuild
    idx.update_file(path)  # incremental update for one file

Usage (module script)
---------------------
    python -m shared.python.codemap.indexer rebuild
"""

from __future__ import annotations

import contextlib
import logging
import sqlite3
import subprocess
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from .db import (
    delete_path,
    get_manifest,
    open_db,
    set_manifest,
    upsert_symbols,
)
from .parsers import parse_file

logger = logging.getLogger(__name__)

_SCHEMA_VERSION = "1.0"
_DB_RELPATH = ".codemap/index.db"
_MANIFEST_RELPATH = ".codemap/manifest.json"

# Extensions to skip during indexing
_SKIP_DIRS = {
    ".git",
    ".hg",
    ".svn",
    "__pycache__",
    ".mypy_cache",
    ".ruff_cache",
    "node_modules",
    ".venv",
    "venv",
    ".tox",
    "dist",
    "build",
    "target",
    ".codemap",
}


def _get_git_commit(repo_root: Path) -> str:
    """Return the current HEAD commit SHA, or 'unknown' if git is unavailable."""
    try:
        result = subprocess.run(  # noqa: S603
            ["git", "rev-parse", "HEAD"],  # noqa: S607
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip() or "unknown"
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"


def _iter_source_files(repo_root: Path) -> Iterator[Path]:
    """Walk *repo_root* and yield all indexable source files."""
    for path in repo_root.rglob("*"):
        if (
            path.is_file()
            and not any(part in _SKIP_DIRS for part in path.parts)
            and path.suffix.lower()
            in {
                ".py",
                ".rs",
                ".ts",
                ".tsx",
                ".js",
                ".jsx",
                ".md",
                ".mdx",
            }
        ):
            yield path


@dataclass
class RebuildStats:
    """Statistics from a full rebuild."""

    files_processed: int = 0
    symbols_indexed: int = 0
    duration_s: float = 0.0
    errors: int = 0
    skipped: int = 0
    extra: dict[str, object] = field(default_factory=dict)


class CodeMapIndex:
    """Manages the SQLite code-map index for a single repository.

    Args:
        repo_root: The root directory of the repository.
        db_path:   Optional override for the database path.
                   Defaults to ``<repo_root>/.codemap/index.db``.
    """

    def __init__(
        self,
        repo_root: Path,
        db_path: Path | None = None,
    ) -> None:
        self.repo_root = repo_root.resolve()
        self.db_path = (db_path or repo_root / _DB_RELPATH).resolve()
        self._conn: sqlite3.Connection | None = None

    # ── Connection lifecycle ────────────────────────────────────────────────────

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = open_db(self.db_path)
        return self._conn

    def close(self) -> None:
        """Close the underlying database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> CodeMapIndex:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # ── Rebuild ─────────────────────────────────────────────────────────────────

    def rebuild(self) -> RebuildStats:
        """Perform a full cold rebuild of the index.

        1. Drops and recreates the ``symbols`` table.
        2. Walks all source files in the repo.
        3. Parses and inserts symbols.
        4. Writes the manifest.

        Returns:
            :class:`RebuildStats` with timing and count information.
        """
        stats = RebuildStats()
        t0 = time.monotonic()

        # Drop and recreate symbols table for clean slate
        with contextlib.suppress(sqlite3.OperationalError):
            self.conn.execute("DROP TABLE IF EXISTS symbols")

        # Re-open to recreate the table
        self.close()
        self._conn = open_db(self.db_path)

        logger.info("Starting cold rebuild of code-map for %s", self.repo_root)

        for src_path in _iter_source_files(self.repo_root):
            try:
                rows = parse_file(src_path, self.repo_root)
                if rows:
                    upsert_symbols(self.conn, rows)
                    stats.symbols_indexed += len(rows)
                stats.files_processed += 1
            except Exception as exc:  # noqa: BLE001
                logger.warning("Error indexing %s: %s", src_path, exc)
                stats.errors += 1

        self.conn.commit()

        stats.duration_s = time.monotonic() - t0

        # Write manifest
        commit = _get_git_commit(self.repo_root)
        set_manifest(self.conn, "repo_root", str(self.repo_root))
        set_manifest(self.conn, "last_commit", commit)
        set_manifest(self.conn, "schema_version", _SCHEMA_VERSION)
        set_manifest(self.conn, "last_rebuild_s", stats.duration_s)
        set_manifest(self.conn, "symbols_count", stats.symbols_indexed)
        self.conn.commit()

        logger.info(
            "Code-map rebuild complete: %d files, %d symbols in %.2fs",
            stats.files_processed,
            stats.symbols_indexed,
            stats.duration_s,
        )
        return stats

    # ── Incremental update ──────────────────────────────────────────────────────

    def update_file(self, path: Path) -> int:
        """Re-index a single file (add/update/delete) incrementally.

        Args:
            path: Absolute or repo-relative path to the changed file.

        Returns:
            Number of symbols indexed (0 if file deleted or unsupported).
        """
        abs_path = path if path.is_absolute() else (self.repo_root / path)
        rel_str = abs_path.relative_to(self.repo_root).as_posix()

        delete_path(self.conn, rel_str)

        if not abs_path.exists():
            self.conn.commit()
            logger.debug("Removed index entries for deleted file: %s", rel_str)
            return 0

        rows = parse_file(abs_path, self.repo_root)
        if rows:
            upsert_symbols(self.conn, rows)

        self.conn.commit()
        logger.debug("Updated index for %s: %d symbols", rel_str, len(rows))
        return len(rows)

    # ── Manifest access ─────────────────────────────────────────────────────────

    def get_manifest_value(self, key: str) -> object:
        """Read a value from the index manifest."""
        return get_manifest(self.conn, key)


# ── Module-level convenience function ─────────────────────────────────────────


def rebuild(repo_root: Path | None = None) -> RebuildStats:
    """Rebuild the code-map index for *repo_root* (defaults to CWD)."""
    root = (repo_root or Path(".")).resolve()
    with CodeMapIndex(root) as idx:
        return idx.rebuild()
