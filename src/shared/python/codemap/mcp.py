"""Model Context Protocol (MCP) server for CodeMap."""

from __future__ import annotations

from typing import Any

from fastmcp import FastMCP

from .api import get_symbol, imports_of, search, who_calls

mcp = FastMCP("CodeMap")


@mcp.tool()
def mcp_search_code(query: str, limit: int = 20) -> list[dict[str, Any]]:
    """Full-text search the codebase index.

    Args:
        query: FTS5 query string (e.g. "my_function OR my_class").
        limit: Maximum number of results to return.
    """
    results = search(query, limit=limit)
    return [
        {
            "kind": r.kind,
            "qualified_name": r.qualified_name,
            "path": r.path,
            "line_start": r.line_start,
            "signature": r.signature,
        }
        for r in results
    ]


@mcp.tool()
def mcp_get_symbol(qualified_name: str) -> dict[str, Any] | None:
    """Get the definition and signature of a specific symbol.

    Args:
        qualified_name: Fully qualified name (e.g. 'my_pkg.mod.Class').
    """
    r = get_symbol(qualified_name)
    if not r:
        return None
    return {
        "kind": r.kind,
        "qualified_name": r.qualified_name,
        "path": r.path,
        "line_start": r.line_start,
        "signature": r.signature,
        "docstring": r.docstring,
    }


@mcp.tool()
def mcp_who_calls(qualified_name: str, limit: int = 20) -> list[dict[str, Any]]:
    """Find all callers of a specific symbol.

    Args:
        qualified_name: Fully qualified name of the callee.
        limit: Maximum number of results to return.
    """
    results = who_calls(qualified_name, limit=limit)
    return [
        {
            "kind": r.kind,
            "qualified_name": r.qualified_name,
            "path": r.path,
            "line_start": r.line_start,
            "signature": r.signature,
        }
        for r in results
    ]


@mcp.tool()
def mcp_imports_of(qualified_name: str, limit: int = 20) -> list[dict[str, Any]]:
    """Find all modules/symbols that import a specific symbol.

    Args:
        qualified_name: Fully qualified name.
        limit: Maximum number of results to return.
    """
    results = imports_of(qualified_name, limit=limit)
    return [
        {
            "kind": r.kind,
            "qualified_name": r.qualified_name,
            "path": r.path,
            "line_start": r.line_start,
            "signature": r.signature,
        }
        for r in results
    ]


def main() -> None:
    """Run the MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
