"""In-process query API for the code-map index.

Provides fast, synchronous search functions that the chat backend
and tool-calling executor can call directly.

All functions accept an optional *db_path* override; if omitted they
resolve to ``<cwd>/.codemap/index.db``.
"""

from __future__ import annotations

import logging
from pathlib import Path

from .db import SymbolRow, open_db, search_fts, who_calls_db

logger = logging.getLogger(__name__)

_DEFAULT_DB = Path(".codemap/index.db")


def _resolve_db(db_path: Path | None) -> Path:
    return (db_path or _DEFAULT_DB).resolve()


def search(
    query: str,
    *,
    limit: int = 20,
    kind: str | None = None,
    db_path: Path | None = None,
) -> list[SymbolRow]:
    """Full-text search the code-map index.

    Args:
        query:   FTS5 query string (e.g. ``'parse*'``, ``'"fit_swing"'``).
        limit:   Maximum results.
        kind:    Optional filter (``'function'``, ``'class'``, …).
        db_path: Override DB path (useful in tests).

    Returns:
        Ordered list of :class:`~codemap.db.SymbolRow` objects.
    """
    db = _resolve_db(db_path)
    if not db.exists():
        logger.warning("Code-map index not found at %s — run 'codemap rebuild'", db)
        return []

    conn = open_db(db)
    try:
        return search_fts(conn, query, limit=limit, kind=kind)
    finally:
        conn.close()


def get_symbol(
    qualified_name: str,
    *,
    db_path: Path | None = None,
) -> SymbolRow | None:
    """Look up a symbol by its exact qualified name.

    Args:
        qualified_name: Fully-qualified dotted name (e.g. ``'my_pkg.utils.parse'``).
        db_path:        Override DB path.

    Returns:
        The matching :class:`~codemap.db.SymbolRow`, or ``None`` if not found.
    """
    db = _resolve_db(db_path)
    if not db.exists():
        return None

    conn = open_db(db)
    try:
        # Use exact match on qualified_name
        row = conn.execute(
            "SELECT kind,qualified_name,path,line_start,line_end,"
            "signature,docstring,imports,calls_out,blake3_hash "
            "FROM symbols WHERE qualified_name = ? LIMIT 1",
            (qualified_name,),
        ).fetchone()
        return SymbolRow(*row) if row else None
    finally:
        conn.close()


def who_calls(
    qualified_name: str,
    *,
    limit: int = 20,
    db_path: Path | None = None,
) -> list[SymbolRow]:
    """Find all symbols that call *qualified_name*.

    Uses the ``calls_out`` FTS column populated during indexing.

    Args:
        qualified_name: The name of the callee to search for.
        limit:          Maximum results.
        db_path:        Override DB path.

    Returns:
        List of :class:`~codemap.db.SymbolRow` objects for callers.
    """
    db = _resolve_db(db_path)
    if not db.exists():
        return []

    conn = open_db(db)
    try:
        return who_calls_db(conn, qualified_name, limit=limit)
    finally:
        conn.close()


def imports_of(
    qualified_name: str,
    *,
    limit: int = 20,
    db_path: Path | None = None,
) -> list[SymbolRow]:
    """Find symbols whose *imports* column includes *qualified_name*.

    Args:
        qualified_name: Name to look up in the imports column.
        limit:          Maximum results.
        db_path:        Override DB path.

    Returns:
        List of :class:`~codemap.db.SymbolRow` objects.
    """
    db = _resolve_db(db_path)
    if not db.exists():
        return []

    conn = open_db(db)
    try:
        try:
            rows = conn.execute(
                "SELECT kind,qualified_name,path,line_start,line_end,"
                "signature,docstring,imports,calls_out,blake3_hash "
                "FROM symbols WHERE imports MATCH ? LIMIT ?",
                (qualified_name, limit),
            ).fetchall()
        except Exception:  # noqa: BLE001
            like_q = f"%{qualified_name}%"
            rows = conn.execute(
                "SELECT kind,qualified_name,path,line_start,line_end,"
                "signature,docstring,imports,calls_out,blake3_hash "
                "FROM symbols WHERE imports LIKE ? LIMIT ?",
                (like_q, limit),
            ).fetchall()
        return [SymbolRow(*r) for r in rows]
    finally:
        conn.close()
