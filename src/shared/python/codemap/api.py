"""Public query API for the code map.

The in-app chat backend and the MCP server both call into this module.
Surface is stable per ``chat_codemap_design.md`` §5.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import threading
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from . import db as db_mod

# ---------------------------------------------------------------------------
# Pydantic models.
# ---------------------------------------------------------------------------


class Symbol(BaseModel):
    """A single indexed symbol."""

    path: str
    kind: str
    name: str
    qualified: str
    sig: str = ""
    docstring: str = ""
    start_line: int
    end_line: int
    calls_out: list[str] = Field(default_factory=list)


class Hit(BaseModel):
    """A search hit."""

    symbol: Symbol
    score: float
    snippet: str = ""


class RepoStats(BaseModel):
    """Summary stats for ``repo_summary()``."""

    repo_root: str
    files: int
    symbols: int
    languages: dict[str, int] = Field(default_factory=dict)
    db_size_bytes: int = 0
    last_indexed: float | None = None
    last_commit: str | None = None


# ---------------------------------------------------------------------------
# Repo-root discovery.
# ---------------------------------------------------------------------------


_DEFAULT_ROOT_LOCK = threading.Lock()
_DEFAULT_ROOT: Path | None = None


def discover_repo_root(start: str | os.PathLike[str] | None = None) -> Path:
    """Find the enclosing git repo root, or fall back to ``start``/CWD."""
    base = Path(start) if start else Path.cwd()
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=str(base),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        if out:
            return Path(out)
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        pass
    # Walk up looking for .codemap/ or .git/.
    cur = base.resolve()
    for parent in [cur, *cur.parents]:
        if (parent / ".codemap").is_dir() or (parent / ".git").exists():
            return parent
    return cur


def _resolve(repo_root: str | os.PathLike[str] | None) -> Path:
    if repo_root is not None:
        return Path(repo_root).resolve()
    global _DEFAULT_ROOT
    with _DEFAULT_ROOT_LOCK:
        if _DEFAULT_ROOT is None:
            _DEFAULT_ROOT = discover_repo_root()
        return _DEFAULT_ROOT


def _row_to_symbol(row: Any) -> Symbol:
    raw_calls = row["calls_out"] if "calls_out" in row.keys() else "[]"
    try:
        calls = json.loads(raw_calls) if raw_calls else []
    except (TypeError, ValueError):
        calls = []
    return Symbol(
        path=row["path"],
        kind=row["kind"],
        name=row["name"],
        qualified=row["qualified"],
        sig=row["sig"],
        docstring=row["docstring"],
        start_line=row["start_line"],
        end_line=row["end_line"],
        calls_out=list(calls),
    )


# ---------------------------------------------------------------------------
# Query helpers.
# ---------------------------------------------------------------------------


_FTS_SAFE = re.compile(r"[A-Za-z0-9_]+")


def _fts_query(raw: str) -> str:
    """Sanitise a free-text query into an FTS5 MATCH expression."""
    tokens = _FTS_SAFE.findall(raw)
    if not tokens:
        return ""
    # Use prefix queries so that "wgs" matches "wgs_reactor".
    return " ".join(f'"{tok}"*' for tok in tokens)


# ---------------------------------------------------------------------------
# Public API.
# ---------------------------------------------------------------------------


def search_code(
    query: str,
    *,
    k: int = 20,
    kind: str | None = None,
    repo_root: str | os.PathLike[str] | None = None,
) -> list[Hit]:
    """Free-text BM25 search across the symbol index."""
    repo = _resolve(repo_root)
    conn = db_mod.open_db(repo)
    try:
        fts = _fts_query(query)
        if not fts:
            return []
        sql = (
            "SELECT s.*, bm25(symbols_fts) AS score "
            "FROM symbols_fts JOIN symbols s ON s.id = symbols_fts.rowid "
            "WHERE symbols_fts MATCH ? "
        )
        params: list[Any] = [fts]
        if kind:
            sql += "AND s.kind = ? "
            params.append(kind)
        sql += "ORDER BY score LIMIT ?"
        params.append(int(k))
        rows = conn.execute(sql, params).fetchall()
        return [
            Hit(symbol=_row_to_symbol(r), score=float(r["score"]), snippet=r["sig"])
            for r in rows
        ]
    finally:
        conn.close()


def get_symbol(
    qualified_name: str,
    *,
    repo_root: str | os.PathLike[str] | None = None,
) -> Symbol | None:
    """Look up a symbol by fully-qualified name (exact match, then suffix match)."""
    repo = _resolve(repo_root)
    conn = db_mod.open_db(repo)
    try:
        row = conn.execute(
            "SELECT * FROM symbols WHERE qualified = ? LIMIT 1",
            (qualified_name,),
        ).fetchone()
        if row is None:
            row = conn.execute(
                "SELECT * FROM symbols WHERE qualified LIKE ? "
                "ORDER BY length(qualified) LIMIT 1",
                (f"%{qualified_name}",),
            ).fetchone()
        if row is None:
            return None
        return _row_to_symbol(row)
    finally:
        conn.close()


def who_calls(
    qualified_name: str,
    *,
    repo_root: str | os.PathLike[str] | None = None,
) -> list[Symbol]:
    """Return symbols whose ``calls_out`` mentions ``qualified_name`` (lexical)."""
    repo = _resolve(repo_root)
    short = qualified_name.rsplit(".", 1)[-1].rsplit("::", 1)[-1]
    conn = db_mod.open_db(repo)
    try:
        # Two-stage: FTS prefilter then exact JSON check.
        fts = _fts_query(short)
        candidates: list[Any] = []
        if fts:
            rows = conn.execute(
                "SELECT s.* FROM symbols_fts "
                "JOIN symbols s ON s.id = symbols_fts.rowid "
                "WHERE symbols_fts MATCH ? LIMIT 500",
                (fts,),
            ).fetchall()
            candidates = list(rows)
        out: list[Symbol] = []
        for row in candidates:
            try:
                calls = json.loads(row["calls_out"]) if row["calls_out"] else []
            except (TypeError, ValueError):
                continue
            for c in calls:
                if c == qualified_name or c.endswith(short):
                    out.append(_row_to_symbol(row))
                    break
        return out
    finally:
        conn.close()


def imports_of(
    path: str,
    *,
    repo_root: str | os.PathLike[str] | None = None,
) -> list[str]:
    """Return the imports declared by ``path``."""
    repo = _resolve(repo_root)
    conn = db_mod.open_db(repo)
    try:
        rel = path.replace(os.sep, "/")
        row = conn.execute(
            "SELECT imports FROM files WHERE path = ?", (rel,)
        ).fetchone()
        if row is None:
            return []
        try:
            data = json.loads(row["imports"])
            return list(data) if isinstance(data, list) else []
        except (TypeError, ValueError):
            return []
    finally:
        conn.close()


def neighbors(
    qualified_name: str,
    *,
    hops: int = 1,
    repo_root: str | os.PathLike[str] | None = None,
) -> list[Symbol]:
    """Return symbols within ``hops`` of ``qualified_name`` (calls / callers).

    Best-effort: a 1-hop neighbour is anything this symbol calls OR anything
    that calls this symbol.
    """
    repo = _resolve(repo_root)
    seen: dict[str, Symbol] = {}
    frontier = [qualified_name]
    for _ in range(max(1, hops)):
        next_frontier: list[str] = []
        for q in frontier:
            sym = get_symbol(q, repo_root=repo)
            if sym is None:
                continue
            seen.setdefault(sym.qualified, sym)
            # Outbound.
            for c in sym.calls_out:
                if c not in seen:
                    next_frontier.append(c)
            # Inbound.
            for caller in who_calls(q, repo_root=repo):
                if caller.qualified not in seen:
                    seen[caller.qualified] = caller
                    next_frontier.append(caller.qualified)
        frontier = next_frontier
        if not frontier:
            break
    seen.pop(qualified_name, None)
    return list(seen.values())


def repo_summary(
    *,
    repo_root: str | os.PathLike[str] | None = None,
) -> RepoStats:
    """Summary statistics for the index at ``repo_root``."""
    repo = _resolve(repo_root)
    conn = db_mod.open_db(repo)
    try:
        files = conn.execute("SELECT COUNT(*) AS n FROM files").fetchone()["n"]
        symbols = conn.execute("SELECT COUNT(*) AS n FROM symbols").fetchone()["n"]
        rows = conn.execute(
            "SELECT language, COUNT(*) AS n FROM files GROUP BY language"
        ).fetchall()
        languages = {r["language"]: r["n"] for r in rows}
        db_p = db_mod.db_path(repo)
        size = db_p.stat().st_size if db_p.exists() else 0
        manifest_p = db_mod.manifest_path(repo)
        last_indexed: float | None = None
        last_commit: str | None = None
        if manifest_p.exists():
            try:
                m = json.loads(manifest_p.read_text(encoding="utf-8"))
                last_indexed = m.get("last_indexed")
                last_commit = m.get("last_commit")
            except (OSError, json.JSONDecodeError):
                pass
        return RepoStats(
            repo_root=str(repo),
            files=files,
            symbols=symbols,
            languages=languages,
            db_size_bytes=size,
            last_indexed=last_indexed,
            last_commit=last_commit,
        )
    finally:
        conn.close()


__all__ = [
    "Hit",
    "RepoStats",
    "Symbol",
    "discover_repo_root",
    "get_symbol",
    "imports_of",
    "neighbors",
    "repo_summary",
    "search_code",
    "who_calls",
]
