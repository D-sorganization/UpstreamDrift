"""Codemap LLM-callable tools for the in-app AI chat.

This module registers the six code-map query functions described in
``chat_codemap_design.md`` Part 2 §5 as tools that the LLM can invoke via
``ToolRegistry``:

* ``search_code(query, k=20, kind=None)`` — BM25 full-text search.
* ``get_symbol(qualified_name)`` — exact / suffix lookup.
* ``who_calls(qualified_name)`` — inbound call sites (lexical).
* ``imports_of(path)`` — declared imports of a file.
* ``neighbors(qualified_name, hops=1)`` — 1-hop call-graph neighbours.
* ``repo_summary()`` — index statistics (file/symbol counts, DB size).

The handlers degrade gracefully when ``.codemap/index.db`` does not exist,
returning a friendly "code-map not yet built" message instead of raising.

API resolution order
--------------------
The canonical six-function surface is defined in
``chat_codemap_design.md`` §5 and implemented in the Tools repo at
``src/shared/python/codemap/api.py``. UpstreamDrift carries a
byte-identical copy under ``src.shared.python.codemap`` so the two are
fungible. We try the top-level ``codemap`` distribution first (the case
when Tools is installed as a sibling package) and fall through to the
in-tree shadow copy.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from src.shared.python.ai.tool_registry import ToolCategory, ToolRegistry

logger = logging.getLogger(__name__)

_INDEX_MISSING_HINT = (
    "Code-map index not yet built. Run `codemap rebuild` (or `make codemap`) "
    "from the repo root to create `.codemap/index.db`, then retry."
)


# ---------------------------------------------------------------------------
# API resolution.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _ApiHandle:
    """Bundle of the resolved codemap API callables (new or legacy)."""

    flavor: str  # "tools" or "local"
    search_code: Any
    get_symbol: Any
    who_calls: Any
    imports_of: Any
    neighbors: Any
    repo_summary: Any


_API_CACHE: dict[str, Any] = {"handle": None, "resolved": False}


def _resolve_api() -> _ApiHandle | None:
    """Return the active codemap API, preferring the new Tools surface."""
    if _API_CACHE["resolved"]:
        cached = _API_CACHE["handle"]
        return cached if isinstance(cached, _ApiHandle) else None
    handle = _try_tools_api() or _try_local_api()
    _API_CACHE["handle"] = handle
    _API_CACHE["resolved"] = True
    if handle is None:
        logger.warning("No codemap API available; chat code-map tools will be no-ops.")
    else:
        logger.info("Using codemap API flavor=%s", handle.flavor)
    return handle


def _try_tools_api() -> _ApiHandle | None:
    """Try to import the new ``codemap`` package from Tools.

    We require the full new surface (``search_code`` + ``neighbors`` +
    ``repo_summary``) before accepting this flavor; an in-tree shadow
    package that exposes only the legacy ``search``/``who_calls(limit=)``
    surface should not match here — fall through to ``_try_local_api``
    which adapts it.
    """
    try:
        from codemap import api as tools_api  # type: ignore[import-not-found]
    except ImportError:
        return None
    required = (
        "search_code",
        "get_symbol",
        "who_calls",
        "imports_of",
        "neighbors",
        "repo_summary",
    )
    if not all(hasattr(tools_api, n) for n in required):
        return None
    return _ApiHandle(
        flavor="tools",
        search_code=tools_api.search_code,
        get_symbol=tools_api.get_symbol,
        who_calls=tools_api.who_calls,
        imports_of=tools_api.imports_of,
        neighbors=tools_api.neighbors,
        repo_summary=tools_api.repo_summary,
    )


def _try_local_api() -> _ApiHandle | None:
    """Use the in-tree ``src.shared.python.codemap.api`` (canonical surface).

    After the consolidation PR (Closes #5206) this module is a byte-identical
    copy of the Tools canonical, so the six functions are imported directly.
    """
    try:
        from src.shared.python.codemap import api as local_api
    except ImportError:
        return None
    required = (
        "search_code",
        "get_symbol",
        "who_calls",
        "imports_of",
        "neighbors",
        "repo_summary",
    )
    if not all(hasattr(local_api, n) for n in required):
        return None
    return _ApiHandle(
        flavor="local",
        search_code=local_api.search_code,
        get_symbol=local_api.get_symbol,
        who_calls=local_api.who_calls,
        imports_of=local_api.imports_of,
        neighbors=local_api.neighbors,
        repo_summary=local_api.repo_summary,
    )


# ---------------------------------------------------------------------------
# Symbol / Hit formatters — produce plain text suitable for chat bubbles.
# ---------------------------------------------------------------------------


def _symbol_to_dict(sym: Any) -> dict[str, Any]:
    """Coerce a canonical ``codemap.api.Symbol`` into a dict.

    Accepts pydantic models (``model_dump``), plain dicts, and best-effort
    legacy ``SymbolRow`` shapes for forward/backward compatibility during a
    consolidation window.
    """
    if sym is None:
        return {}
    if hasattr(sym, "model_dump"):
        return dict(sym.model_dump())
    if isinstance(sym, dict):
        return dict(sym)
    # Legacy SymbolRow fall-through (qualified_name / signature / line_*).
    if hasattr(sym, "__dict__"):
        return {
            "path": getattr(sym, "path", ""),
            "kind": getattr(sym, "kind", ""),
            "name": getattr(sym, "qualified_name", "").rsplit(".", 1)[-1],
            "qualified": getattr(sym, "qualified_name", ""),
            "sig": getattr(sym, "signature", ""),
            "docstring": getattr(sym, "docstring", ""),
            "start_line": getattr(sym, "line_start", 0),
            "end_line": getattr(sym, "line_end", 0),
        }
    return {}


def _format_symbol_line(sym_dict: dict[str, Any]) -> str:
    """Render a single symbol as ``[kind] qualified  path:start-end  sig``."""
    kind = sym_dict.get("kind", "?")
    qual = sym_dict.get("qualified") or sym_dict.get("qualified_name") or "?"
    path = sym_dict.get("path", "?")
    s = sym_dict.get("start_line") or sym_dict.get("line_start") or 0
    e = sym_dict.get("end_line") or sym_dict.get("line_end") or 0
    sig = (sym_dict.get("sig") or sym_dict.get("signature") or "").strip()
    base = f"[{kind}] {qual}  {path}:{s}-{e}"
    return f"{base}  {sig}" if sig else base


def _format_hits(hits: list[Any]) -> str:
    if not hits:
        return "No results."
    lines: list[str] = []
    for h in hits:
        # Tools Hit has .symbol and .score; local SymbolRow is itself a symbol.
        sym = getattr(h, "symbol", h)
        lines.append(_format_symbol_line(_symbol_to_dict(sym)))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Handler implementations.
# ---------------------------------------------------------------------------


def _err_index_missing() -> dict[str, Any]:
    return {"success": False, "error": _INDEX_MISSING_HINT}


def _ok(result: Any) -> dict[str, Any]:
    return {"success": True, "result": result}


def _safe_call(fn: Any, *args: Any, **kwargs: Any) -> dict[str, Any]:
    """Run an API call, mapping exceptions / empty indexes to friendly results."""
    try:
        return _ok(fn(*args, **kwargs))
    except FileNotFoundError:
        return _err_index_missing()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Codemap tool failed: %s", exc)
        return {"success": False, "error": f"{type(exc).__name__}: {exc}"}


# ---------------------------------------------------------------------------
# Tool registration.
# ---------------------------------------------------------------------------


CODEMAP_TOOL_NAMES = (
    "search_code",
    "get_symbol",
    "who_calls",
    "imports_of",
    "neighbors",
    "repo_summary",
)


def _register_search_code(registry: ToolRegistry, api: _ApiHandle) -> None:
    @registry.register(
        name="search_code",
        description=(
            "Full-text BM25 search across the code-map index. Returns ranked "
            "symbol hits (functions, classes, methods, constants) matching the "
            "query. Use this to discover where something lives in the codebase."
        ),
        category=ToolCategory.CONFIGURATION,
        expertise_level=2,
    )
    def search_code(query: str, k: int = 20, kind: str | None = None) -> dict[str, Any]:
        """Search the code-map.

        Args:
            query: Free-text query (e.g. ``"ChatDockWidget"``, ``"fit_swing"``).
            k:     Maximum results to return.
            kind:  Optional symbol kind filter (``function`` / ``class`` / ...).
        """
        result = _safe_call(api.search_code, query, k=k, kind=kind)
        if result["success"]:
            hits = result["result"]
            result["result"] = _format_hits(hits)
            result["count"] = len(hits)
        return result


def _register_get_symbol(registry: ToolRegistry, api: _ApiHandle) -> None:
    @registry.register(
        name="get_symbol",
        description=(
            "Look up a symbol by its fully-qualified name (e.g. "
            "'src.shared.python.chat.ChatDockWidget'). Returns the file, line "
            "range, signature, and docstring."
        ),
        category=ToolCategory.CONFIGURATION,
        expertise_level=2,
    )
    def get_symbol(qualified_name: str) -> dict[str, Any]:
        """Get the definition of a single symbol."""
        result = _safe_call(api.get_symbol, qualified_name)
        if not result["success"]:
            return result
        sym = result["result"]
        if sym is None:
            return _ok(f"Symbol not found: {qualified_name}")
        d = _symbol_to_dict(sym)
        lines = [_format_symbol_line(d)]
        if d.get("docstring"):
            lines.append(f"\n{d['docstring']}")
        return _ok("\n".join(lines))


def _register_who_calls(registry: ToolRegistry, api: _ApiHandle) -> None:
    @registry.register(
        name="who_calls",
        description=(
            "Find symbols that call the given qualified name. Lexical match "
            "against the indexed ``calls_out`` column — useful for impact "
            "analysis before refactors."
        ),
        category=ToolCategory.CONFIGURATION,
        expertise_level=2,
    )
    def who_calls(qualified_name: str) -> dict[str, Any]:
        """Find callers of a symbol."""
        result = _safe_call(api.who_calls, qualified_name)
        if result["success"]:
            symbols = result["result"]
            result["result"] = _format_hits(symbols)
            result["count"] = len(symbols)
        return result


def _register_imports_of(registry: ToolRegistry, api: _ApiHandle) -> None:
    @registry.register(
        name="imports_of",
        description=(
            "Return the imports declared by a given file (repo-relative path). "
            "Useful for understanding a module's dependencies."
        ),
        category=ToolCategory.CONFIGURATION,
        expertise_level=2,
    )
    def imports_of(path: str) -> dict[str, Any]:
        """List imports declared by a file."""
        result = _safe_call(api.imports_of, path)
        if not result["success"]:
            return result
        items = result["result"] or []
        if not items:
            return _ok(f"No imports recorded for {path}.")
        # Tools API returns list[str]; local returns list[SymbolRow].
        if items and not isinstance(items[0], str):
            items = [_symbol_to_dict(s).get("qualified", "?") for s in items]
        return _ok("\n".join(items))


def _register_neighbors(registry: ToolRegistry, api: _ApiHandle) -> None:
    @registry.register(
        name="neighbors",
        description=(
            "Return symbols within N hops of the given qualified name in the "
            "call graph (callers + callees). Helpful for exploring the "
            "neighbourhood around a function before editing it."
        ),
        category=ToolCategory.CONFIGURATION,
        expertise_level=3,
    )
    def neighbors(qualified_name: str, hops: int = 1) -> dict[str, Any]:
        """Return N-hop call-graph neighbours."""
        result = _safe_call(api.neighbors, qualified_name, hops=hops)
        if result["success"]:
            symbols = result["result"]
            result["result"] = _format_hits(symbols)
            result["count"] = len(symbols)
        return result


def _register_repo_summary(registry: ToolRegistry, api: _ApiHandle) -> None:
    @registry.register(
        name="repo_summary",
        description=(
            "Return summary statistics for the code-map index: file count, "
            "symbol count, per-language breakdown, DB size, and last-indexed "
            "timestamp. Use this to confirm the index is up to date."
        ),
        category=ToolCategory.CONFIGURATION,
        expertise_level=1,
    )
    def repo_summary() -> dict[str, Any]:
        """Return code-map index statistics."""
        result = _safe_call(api.repo_summary)
        if not result["success"]:
            return result
        stats = result["result"]
        d = stats.model_dump() if hasattr(stats, "model_dump") else dict(stats)
        size_mb = d.get("db_size_bytes", 0) / (1024 * 1024)
        langs = d.get("languages") or {}
        lang_str = ", ".join(f"{k}={v}" for k, v in sorted(langs.items())) or "n/a"
        summary = (
            f"Repo: {d.get('repo_root', '?')}\n"
            f"Files: {d.get('files', 0)}\n"
            f"Symbols: {d.get('symbols', 0)}\n"
            f"Languages: {lang_str}\n"
            f"DB size: {size_mb:.2f} MB\n"
            f"Last indexed: {d.get('last_indexed') or 'unknown'}"
        )
        return _ok(summary)


def register_codemap_tools(registry: ToolRegistry) -> int:
    """Register all six codemap tools with the registry.

    Returns the count of tools registered. Returns ``0`` and logs a warning if
    no codemap API can be resolved (the chat continues to work without
    code-search ability).
    """
    api = _resolve_api()
    if api is None:
        return 0
    _register_search_code(registry, api)
    _register_get_symbol(registry, api)
    _register_who_calls(registry, api)
    _register_imports_of(registry, api)
    _register_neighbors(registry, api)
    _register_repo_summary(registry, api)
    logger.info(
        "Registered %d codemap tools (flavor=%s)",
        len(CODEMAP_TOOL_NAMES),
        api.flavor,
    )
    return len(CODEMAP_TOOL_NAMES)


__all__ = [
    "CODEMAP_TOOL_NAMES",
    "register_codemap_tools",
]
