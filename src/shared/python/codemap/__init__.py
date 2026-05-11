"""Repo-aware code map — tree-sitter symbol index + SQLite FTS5.

See ``chat_codemap_design.md`` for the full design. The public surface is
defined in :mod:`codemap.api`; this top-level module re-exports it for
convenience.
"""

from __future__ import annotations

from .api import (
    Hit,
    RepoStats,
    Symbol,
    discover_repo_root,
    get_symbol,
    imports_of,
    neighbors,
    repo_summary,
    search_code,
    who_calls,
)
from .indexer import RebuildStats, rebuild

__all__ = [
    "Hit",
    "RebuildStats",
    "RepoStats",
    "Symbol",
    "discover_repo_root",
    "get_symbol",
    "imports_of",
    "neighbors",
    "rebuild",
    "repo_summary",
    "search_code",
    "who_calls",
]

__version__ = "0.1.0"
