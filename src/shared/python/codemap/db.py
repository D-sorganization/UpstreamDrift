"""SQLite schema + open/init helpers for the code map index.

Single-file index living at ``<repo>/.codemap/index.db``. Uses SQLite FTS5
(built into the stdlib ``sqlite3`` module on modern Python) for BM25 search
over symbol names + signatures + docstrings.

Source slices are NOT copied into the DB — only line ranges + a blake3 hash
so the indexer can skip unchanged symbols on incremental rebuilds.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA_VERSION = 1
"""Bump when any table definition changes incompatibly."""

DB_DIR_NAME = ".codemap"
DB_FILE_NAME = "index.db"
MANIFEST_FILE_NAME = "manifest.json"


def db_path(repo_root: Path) -> Path:
    """Return the canonical index DB path for ``repo_root``."""
    return Path(repo_root) / DB_DIR_NAME / DB_FILE_NAME


def manifest_path(repo_root: Path) -> Path:
    """Return the manifest JSON path."""
    return Path(repo_root) / DB_DIR_NAME / MANIFEST_FILE_NAME


def open_db(repo_root: Path) -> sqlite3.Connection:
    """Open (and lazily initialise) the index DB for ``repo_root``."""
    target = db_path(repo_root)
    target.parent.mkdir(parents=True, exist_ok=True)
    # Ensure .codemap/ is gitignored locally regardless of repo .gitignore state.
    gitignore = target.parent / ".gitignore"
    if not gitignore.exists():
        gitignore.write_text("*\n", encoding="utf-8")
    conn = sqlite3.connect(target)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.execute("PRAGMA foreign_keys = ON")
    init_schema(conn)
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    """Create tables if missing. Idempotent."""
    cur = conn.cursor()
    cur.executescript(
        """
        CREATE TABLE IF NOT EXISTS meta (
            key   TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS files (
            path        TEXT PRIMARY KEY,
            language    TEXT NOT NULL,
            hash        TEXT NOT NULL,
            mtime       REAL NOT NULL,
            size        INTEGER NOT NULL,
            imports     TEXT NOT NULL DEFAULT '[]',
            indexed_at  REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS symbols (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            path        TEXT NOT NULL,
            kind        TEXT NOT NULL,
            name        TEXT NOT NULL,
            qualified   TEXT NOT NULL,
            sig         TEXT NOT NULL DEFAULT '',
            docstring   TEXT NOT NULL DEFAULT '',
            start_line  INTEGER NOT NULL,
            end_line    INTEGER NOT NULL,
            calls_out   TEXT NOT NULL DEFAULT '[]',
            hash        TEXT NOT NULL,
            FOREIGN KEY (path) REFERENCES files(path) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_symbols_qualified ON symbols(qualified);
        CREATE INDEX IF NOT EXISTS idx_symbols_name      ON symbols(name);
        CREATE INDEX IF NOT EXISTS idx_symbols_path      ON symbols(path);
        CREATE INDEX IF NOT EXISTS idx_symbols_kind      ON symbols(kind);
        """,
    )

    # FTS5 virtual table — content-less, mirrors symbols(id) for BM25 search.
    # Triggers below keep the FTS index in sync. `co` is the FTS column
    # alias for `calls_out` to keep trigger bodies under the 88-char limit.
    cur.executescript(
        """
        CREATE VIRTUAL TABLE IF NOT EXISTS symbols_fts USING fts5(
            name, qualified, sig, docstring, co,
            content='symbols', content_rowid='id',
            tokenize='unicode61'
        );

        CREATE TRIGGER IF NOT EXISTS symbols_ai AFTER INSERT ON symbols BEGIN
            INSERT INTO symbols_fts(rowid, name, qualified, sig, docstring, co)
            VALUES (new.id, new.name, new.qualified,
                    new.sig, new.docstring, new.calls_out);
        END;

        CREATE TRIGGER IF NOT EXISTS symbols_ad AFTER DELETE ON symbols BEGIN
            INSERT INTO symbols_fts(
                symbols_fts, rowid, name, qualified, sig, docstring, co
            ) VALUES('delete', old.id, old.name, old.qualified,
                     old.sig, old.docstring, old.calls_out);
        END;

        CREATE TRIGGER IF NOT EXISTS symbols_au AFTER UPDATE ON symbols BEGIN
            INSERT INTO symbols_fts(
                symbols_fts, rowid, name, qualified, sig, docstring, co
            ) VALUES('delete', old.id, old.name, old.qualified,
                     old.sig, old.docstring, old.calls_out);
            INSERT INTO symbols_fts(rowid, name, qualified, sig, docstring, co)
            VALUES (new.id, new.name, new.qualified,
                    new.sig, new.docstring, new.calls_out);
        END;
        """,
    )

    cur.execute(
        "INSERT OR REPLACE INTO meta(key, value) VALUES(?, ?)",
        ("schema_version", str(SCHEMA_VERSION)),
    )
    conn.commit()


def get_schema_version(conn: sqlite3.Connection) -> int:
    """Return the schema version recorded in the meta table, or 0 if unknown."""
    row = conn.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
    if row is None:
        return 0
    try:
        return int(row[0])
    except (TypeError, ValueError):
        return 0


__all__ = [
    "DB_DIR_NAME",
    "DB_FILE_NAME",
    "MANIFEST_FILE_NAME",
    "SCHEMA_VERSION",
    "db_path",
    "get_schema_version",
    "init_schema",
    "manifest_path",
    "open_db",
]
