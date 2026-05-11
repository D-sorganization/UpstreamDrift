"""Code-map package: tree-sitter + SQLite FTS5 symbol index.

Provides a repo-aware code map so the in-app chat can answer
'where is X defined / called?' without re-scanning the tree per prompt.

Sub-modules
-----------
db          SQLite FTS5 schema + CRUD helpers
parsers     Tree-sitter language parsers
indexer     Cold-rebuild + incremental update logic
api         In-process query API (search, get_symbol, who_calls)
cli         Command-line entry points (codemap rebuild / search / …)
watcher     Watchdog + git-hook bridge for incremental updates
"""

from __future__ import annotations

__all__ = [
    "CodeMapIndex",
    "SymbolRow",
    "rebuild",
    "search",
    "get_symbol",
    "who_calls",
]

from .api import get_symbol, search, who_calls
from .db import SymbolRow
from .indexer import CodeMapIndex, rebuild
