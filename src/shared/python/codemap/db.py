"""SQLite FTS5 schema + CRUD helpers for the code-map index.

Schema
------
symbols (FTS5 virtual table)
    kind            TEXT  — 'function', 'class', 'method', 'module', 'constant', etc.
    qualified_name  TEXT  — fully-qualified dotted name (e.g. 'my_pkg.utils.parse')
    path            TEXT  — repo-relative path (e.g. 'src/my_pkg/utils.py')
    line_start      INT
    line_end        INT
    signature       TEXT  — condensed first-line signature or declaration
    docstring       TEXT  — first paragraph of docstring (may be empty)
    imports         TEXT  — space-separated imported names (for call-site search)
    calls_out       TEXT  — space-separated called names (from AST)
    blake3_hash     TEXT  — per-file content hash for incremental rebuild

manifest (plain table)
    key             TEXT  PRIMARY KEY
    value           TEXT
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DDL_MANIFEST = """
CREATE TABLE IF NOT EXISTS manifest (
    key   TEXT PRIMARY KEY,
    value TEXT
);
"""

_DDL_SYMBOLS = """
CREATE VIRTUAL TABLE IF NOT EXISTS symbols USING fts5(
    kind,
    qualified_name,
    path UNINDEXED,
    line_start UNINDEXED,
    line_end UNINDEXED,
    signature,
    docstring,
    imports,
    calls_out,
    blake3_hash UNINDEXED,
    tokenize = 'unicode61'
);
"""

_DDL_SYMBOLS_FALLBACK = """
CREATE TABLE IF NOT EXISTS symbols (
    kind           TEXT,
    qualified_name TEXT,
    path           TEXT,
    line_start     INTEGER,
    line_end       INTEGER,
    signature      TEXT,
    docstring      TEXT,
    imports        TEXT,
    calls_out      TEXT,
    blake3_hash    TEXT
);
CREATE INDEX IF NOT EXISTS idx_symbols_path ON symbols(path);
CREATE INDEX IF NOT EXISTS idx_symbols_kind ON symbols(kind);
"""


@dataclass
class SymbolRow:
    """A single indexed symbol entry."""

    kind: str
    qualified_name: str
    path: str
    line_start: int
    line_end: int
    signature: str = ""
    docstring: str = ""
    imports: str = ""
    calls_out: str = ""
    blake3_hash: str = ""

    def as_tuple(self) -> tuple[Any, ...]:
        """Return a tuple of values in field-declaration order."""
        return tuple(getattr(self, f.name) for f in fields(self))


def open_db(db_path: Path) -> sqlite3.Connection:
    """Open (or create) the index database at *db_path*.

    Applies WAL mode for concurrent read performance and attempts to create
    the FTS5 virtual table, falling back to a plain table if the SQLite
    build lacks FTS5 support.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute(_DDL_MANIFEST)

    try:
        conn.execute(_DDL_SYMBOLS)
    except sqlite3.OperationalError as exc:
        if "fts5" in str(exc).lower():
            logger.warning("SQLite FTS5 not available — falling back to plain table")
            for stmt in _DDL_SYMBOLS_FALLBACK.split(";"):
                stmt = stmt.strip()
                if stmt:
                    conn.execute(stmt)
        else:
            raise

    conn.commit()
    return conn


def upsert_symbols(conn: sqlite3.Connection, rows: list[SymbolRow]) -> None:
    """Insert or replace symbol rows in bulk."""
    if not rows:
        return
    conn.execute("DELETE FROM symbols WHERE path = ?", (rows[0].path,))
    conn.executemany(
        "INSERT INTO symbols VALUES (?,?,?,?,?,?,?,?,?,?)",
        [r.as_tuple() for r in rows],
    )


def delete_path(conn: sqlite3.Connection, repo_rel_path: str) -> None:
    """Remove all symbols for a given *repo_rel_path*."""
    conn.execute("DELETE FROM symbols WHERE path = ?", (repo_rel_path,))


def set_manifest(conn: sqlite3.Connection, key: str, value: Any) -> None:
    """Write a manifest key/value (JSON-serialised)."""
    conn.execute(
        "INSERT OR REPLACE INTO manifest(key, value) VALUES (?, ?)",
        (key, json.dumps(value)),
    )


def get_manifest(conn: sqlite3.Connection, key: str) -> Any:
    """Read a manifest value (JSON-deserialised), or None if absent."""
    row = conn.execute("SELECT value FROM manifest WHERE key = ?", (key,)).fetchone()
    return json.loads(row[0]) if row else None


def search_fts(
    conn: sqlite3.Connection,
    query: str,
    *,
    limit: int = 20,
    kind: str | None = None,
) -> list[SymbolRow]:
    """Full-text search across qualified_name, signature, and docstring.

    Args:
        conn:   Open database connection.
        query:  FTS5 query string (e.g. 'parse*' or '"parse spec"').
        limit:  Maximum rows to return.
        kind:   Optional filter by symbol kind ('function', 'class', …).

    Returns:
        List of matching :class:`SymbolRow` objects, ranked by relevance.
    """
    try:
        if kind:
            rows = conn.execute(
                "SELECT kind,qualified_name,path,line_start,line_end,"
                "signature,docstring,imports,calls_out,blake3_hash "
                "FROM symbols WHERE symbols MATCH ? AND kind = ? LIMIT ?",
                (query, kind, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT kind,qualified_name,path,line_start,line_end,"
                "signature,docstring,imports,calls_out,blake3_hash "
                "FROM symbols WHERE symbols MATCH ? LIMIT ?",
                (query, limit),
            ).fetchall()
    except sqlite3.OperationalError:
        # Fallback for plain-table mode (no FTS)
        like_query = f"%{query}%"
        if kind:
            rows = conn.execute(
                "SELECT kind,qualified_name,path,line_start,line_end,"
                "signature,docstring,imports,calls_out,blake3_hash "
                "FROM symbols WHERE qualified_name LIKE ? AND kind = ? LIMIT ?",
                (like_query, kind, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT kind,qualified_name,path,line_start,line_end,"
                "signature,docstring,imports,calls_out,blake3_hash "
                "FROM symbols WHERE qualified_name LIKE ? LIMIT ?",
                (like_query, limit),
            ).fetchall()

    return [SymbolRow(*r) for r in rows]


def who_calls_db(
    conn: sqlite3.Connection, qualified_name: str, *, limit: int = 20
) -> list[SymbolRow]:
    """Find symbols whose *calls_out* includes *qualified_name*."""
    try:
        rows = conn.execute(
            "SELECT kind,qualified_name,path,line_start,line_end,"
            "signature,docstring,imports,calls_out,blake3_hash "
            "FROM symbols WHERE calls_out MATCH ? LIMIT ?",
            (qualified_name, limit),
        ).fetchall()
    except sqlite3.OperationalError:
        like_query = f"%{qualified_name}%"
        rows = conn.execute(
            "SELECT kind,qualified_name,path,line_start,line_end,"
            "signature,docstring,imports,calls_out,blake3_hash "
            "FROM symbols WHERE calls_out LIKE ? LIMIT ?",
            (like_query, limit),
        ).fetchall()
    return [SymbolRow(*r) for r in rows]
