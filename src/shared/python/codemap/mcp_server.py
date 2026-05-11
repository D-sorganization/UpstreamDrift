"""MCP server exposing the code map to external coding agents.

Tools exposed (matching the in-app chat surface):

    search_code(query, k=20, kind=None) -> list[Hit]
    get_symbol(qualified_name) -> Symbol | None
    who_calls(qualified_name) -> list[Symbol]
    imports_of(path) -> list[str]

The server discovers the target repo via the ``CODEMAP_REPO_ROOT`` env var,
falling back to the current working directory.

Console-script entry point: ``codemap-mcp``.
"""

from __future__ import annotations

import os
import sys
from typing import Any

from . import api as api_mod


def _repo_root() -> str | None:
    return os.environ.get("CODEMAP_REPO_ROOT") or None


def _build_fastmcp():
    """Build a FastMCP server. Falls back to None on ImportError."""
    try:
        from mcp.server.fastmcp import FastMCP  # type: ignore[import-not-found]
    except Exception:
        return None

    mcp = FastMCP("codemap")

    @mcp.tool()
    def search_code(
        query: str, k: int = 20, kind: str | None = None
    ) -> list[dict[str, Any]]:
        """Free-text BM25 search across the repo's symbol index.

        Args:
            query: Search terms (whitespace-separated).
            k: Max results.
            kind: Optional filter (function, class, method, struct, heading, ...).
        """
        hits = api_mod.search_code(query, k=k, kind=kind, repo_root=_repo_root())
        return [h.model_dump() for h in hits]

    @mcp.tool()
    def get_symbol(qualified_name: str) -> dict[str, Any] | None:
        """Look up a symbol by qualified name (e.g. ``ChatDockWidget._setup_ui``)."""
        sym = api_mod.get_symbol(qualified_name, repo_root=_repo_root())
        return sym.model_dump() if sym else None

    @mcp.tool()
    def who_calls(qualified_name: str) -> list[dict[str, Any]]:
        """Return symbols whose body references ``qualified_name`` (lexical)."""
        rows = api_mod.who_calls(qualified_name, repo_root=_repo_root())
        return [r.model_dump() for r in rows]

    @mcp.tool()
    def imports_of(path: str) -> list[str]:
        """Return the imports declared by the file at ``path`` (repo-relative)."""
        return api_mod.imports_of(path, repo_root=_repo_root())

    @mcp.tool()
    def repo_summary() -> dict[str, Any]:
        """Return summary statistics for the indexed repo."""
        return api_mod.repo_summary(repo_root=_repo_root()).model_dump()

    return mcp


def main(argv: list[str] | None = None) -> int:
    """Run the MCP server on stdio."""
    server = _build_fastmcp()
    if server is None:
        print(
            "codemap-mcp: the 'mcp' package is not installed. "
            "Install with: pip install mcp",
            file=sys.stderr,
        )
        return 2
    server.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
