"""Codemap tools for AI integration with UpstreamDrift."""

from __future__ import annotations

import logging
from typing import Any

from src.shared.python.ai.tool_registry import ToolCategory, ToolRegistry

logger = logging.getLogger(__name__)


def _register_search_codebase_tool(registry: ToolRegistry) -> None:
    @registry.register(
        name="search_codebase",
        description=(
            "Full-text search across the codebase for symbols, functions, "
            "classes, and docstrings."
        ),
        category=ToolCategory.CONFIGURATION,
        expertise_level=3,
    )
    def search_codebase(query: str) -> dict[str, Any]:
        """Search the codebase for a query using the codemap index.

        Args:
            query: Search query.

        Returns:
            Formatted search results.
        """
        try:
            from src.shared.python.codemap.api import search

            results = search(query, limit=10)
            if not results:
                return {"success": True, "result": "No results found."}
            formatted = []
            for r in results:
                formatted.append(
                    f"[{r.kind}] {r.qualified_name} at {r.path}:{r.line_start}"
                )
            return {"success": True, "result": "\n".join(formatted)}
        except Exception as e:  # noqa: BLE001
            return {"success": False, "error": str(e)}


def _register_get_symbol_tool(registry: ToolRegistry) -> None:
    @registry.register(
        name="get_symbol",
        description=(
            "Get the definition and signature of a specific symbol by qualified name."
        ),
        category=ToolCategory.CONFIGURATION,
        expertise_level=3,
    )
    def get_symbol(qualified_name: str) -> dict[str, Any]:
        """Get details about a specific symbol.

        Args:
            qualified_name: Fully qualified name of the symbol.

        Returns:
            Symbol definition and details.
        """
        try:
            from src.shared.python.codemap.api import get_symbol as api_get_symbol

            result = api_get_symbol(qualified_name)
            if not result:
                return {
                    "success": True,
                    "result": f"Symbol {qualified_name} not found.",
                }
            formatted = f"[{result.kind}] {result.qualified_name} at {result.path}:{result.line_start}\n"
            if result.signature:
                formatted += f"Signature: {result.signature}\n"
            if result.docstring:
                formatted += f"Docstring: {result.docstring}"
            return {"success": True, "result": formatted}
        except Exception as e:  # noqa: BLE001
            return {"success": False, "error": str(e)}


def _register_find_callers_tool(registry: ToolRegistry) -> None:
    @registry.register(
        name="find_callers",
        description="Find all callers of a specific symbol.",
        category=ToolCategory.CONFIGURATION,
        expertise_level=3,
    )
    def find_callers(qualified_name: str) -> dict[str, Any]:
        """Find callers of a specific symbol.

        Args:
            qualified_name: Fully qualified name of the callee.

        Returns:
            List of formatted caller symbols.
        """
        try:
            from src.shared.python.codemap.api import who_calls

            results = who_calls(qualified_name, limit=10)
            if not results:
                return {
                    "success": True,
                    "result": f"No callers found for {qualified_name}.",
                }
            formatted = []
            for r in results:
                formatted.append(
                    f"[{r.kind}] {r.qualified_name} at {r.path}:{r.line_start}"
                )
            return {"success": True, "result": "\n".join(formatted)}
        except Exception as e:  # noqa: BLE001
            return {"success": False, "error": str(e)}


def create_codemap_tools_for_registry() -> list[dict[str, Any]]:
    """Return tool definitions for the registry to process."""
    # We can either return definitions to be registered, or register them directly.
    # We'll use the _register_* functions directly in register_codemap_tools.
    return []


def register_codemap_tools(registry: ToolRegistry) -> None:
    """Register all codemap tools with the registry."""
    _register_search_codebase_tool(registry)
    _register_get_symbol_tool(registry)
    _register_find_callers_tool(registry)
    logger.info("Registered codemap tools")
