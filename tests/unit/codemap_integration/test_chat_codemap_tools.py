"""Unit tests for the chat -> codemap tool wiring.

Verifies that the six ``codemap.api`` functions are registered with the
``ToolRegistry`` under the expected names, expose well-formed JSON-schema
for both OpenAI and Anthropic tool-calling, and produce sensible payloads
when invoked end-to-end against a stubbed API surface.

These tests are deliberately hermetic: they do **not** require a built
``.codemap/index.db``. The real API is monkey-patched out so we test the
registration and execution contract, not the underlying indexer (which is
covered by ``tests/unit/test_codemap.py`` and the upstream Tools package).
"""

from __future__ import annotations

from typing import Any

import pytest

from src.shared.python.ai.tool_registry import ToolRegistry
from src.shared.python.ai.tools import codemap_tools

EXPECTED_TOOLS = (
    "search_code",
    "get_symbol",
    "who_calls",
    "imports_of",
    "neighbors",
    "repo_summary",
)


# ---------------------------------------------------------------------------
# Stubs.
# ---------------------------------------------------------------------------


class _StubSymbol:
    """Minimal stand-in for ``codemap.api.Symbol`` (pydantic-shaped)."""

    def __init__(self, **kw: Any) -> None:
        self._d = {
            "path": "src/foo.py",
            "kind": "function",
            "name": "do_thing",
            "qualified": "foo.do_thing",
            "sig": "def do_thing(x: int) -> int",
            "docstring": "Do the thing.",
            "start_line": 10,
            "end_line": 20,
            "calls_out": [],
            **kw,
        }

    def model_dump(self) -> dict[str, Any]:
        return dict(self._d)


class _StubHit:
    def __init__(self, sym: _StubSymbol, score: float = -1.0) -> None:
        self.symbol = sym
        self.score = score
        self.snippet = sym._d["sig"]


class _StubStats:
    def __init__(self, **kw: Any) -> None:
        self._d = {
            "repo_root": "/tmp/repo",
            "files": 12,
            "symbols": 99,
            "languages": {"python": 12},
            "db_size_bytes": 1024,
            "last_indexed": 1234567890.0,
            **kw,
        }

    def model_dump(self) -> dict[str, Any]:
        return dict(self._d)


def _make_stub_api() -> codemap_tools._ApiHandle:
    sym = _StubSymbol()
    return codemap_tools._ApiHandle(
        flavor="stub",
        search_code=lambda q, *, k=20, kind=None: [_StubHit(sym)],
        get_symbol=lambda qn: sym,
        who_calls=lambda qn: [sym],
        imports_of=lambda path: ["os", "sys"],
        neighbors=lambda qn, *, hops=1: [sym],
        repo_summary=lambda: _StubStats(),
    )


@pytest.fixture()
def stub_registry(monkeypatch: pytest.MonkeyPatch) -> ToolRegistry:
    """Return a ``ToolRegistry`` with the six codemap tools backed by stubs."""
    handle = _make_stub_api()
    monkeypatch.setattr(codemap_tools, "_resolve_api", lambda: handle)
    # Reset the module-level cache so register picks up our stub.
    monkeypatch.setattr(
        codemap_tools, "_API_CACHE", {"handle": handle, "resolved": True}
    )
    registry = ToolRegistry()
    n = codemap_tools.register_codemap_tools(registry)
    assert n == len(EXPECTED_TOOLS)
    return registry


# ---------------------------------------------------------------------------
# Registration shape.
# ---------------------------------------------------------------------------


def test_all_six_tools_registered(stub_registry: ToolRegistry) -> None:
    names = {t.name for t in stub_registry.list_tools()}
    assert names == set(EXPECTED_TOOLS)


def test_tool_names_constant_matches_implementation() -> None:
    assert set(codemap_tools.CODEMAP_TOOL_NAMES) == set(EXPECTED_TOOLS)


def test_openai_tool_schema_well_formed(stub_registry: ToolRegistry) -> None:
    schemas = stub_registry.get_tools_for_provider("openai")
    assert len(schemas) == len(EXPECTED_TOOLS)
    for s in schemas:
        assert s["type"] == "function"
        fn = s["function"]
        assert fn["name"] in EXPECTED_TOOLS
        assert isinstance(fn["description"], str) and fn["description"]
        params = fn["parameters"]
        assert params["type"] == "object"
        assert "properties" in params
        assert "required" in params


def test_anthropic_tool_schema_well_formed(stub_registry: ToolRegistry) -> None:
    schemas = stub_registry.get_tools_for_provider("anthropic")
    assert len(schemas) == len(EXPECTED_TOOLS)
    for s in schemas:
        assert s["name"] in EXPECTED_TOOLS
        assert "input_schema" in s
        assert s["input_schema"]["type"] == "object"


@pytest.mark.parametrize("name", EXPECTED_TOOLS)
def test_each_tool_has_required_query_parameter(
    stub_registry: ToolRegistry, name: str
) -> None:
    """Each tool's required-parameter set must exactly match its signature."""
    tool = stub_registry.get_tool(name)
    assert tool is not None
    required = {p.name for p in tool.parameters if p.required}
    expected_required = {
        "search_code": {"query"},
        "get_symbol": {"qualified_name"},
        "who_calls": {"qualified_name"},
        "imports_of": {"path"},
        "neighbors": {"qualified_name"},
        "repo_summary": set(),
    }
    assert required == expected_required[name]


# ---------------------------------------------------------------------------
# Execution contracts.
# ---------------------------------------------------------------------------


def test_search_code_returns_formatted_hits(stub_registry: ToolRegistry) -> None:
    r = stub_registry.execute("search_code", {"query": "ChatDockWidget"})
    assert r.success is True
    assert isinstance(r.result, dict)
    assert r.result["success"] is True
    assert "foo.do_thing" in r.result["result"]
    assert "src/foo.py:10-20" in r.result["result"]
    assert r.result["count"] == 1


def test_get_symbol_renders_signature_and_docstring(
    stub_registry: ToolRegistry,
) -> None:
    r = stub_registry.execute("get_symbol", {"qualified_name": "foo.do_thing"})
    assert r.success is True
    text = r.result["result"]
    assert "[function] foo.do_thing" in text
    assert "Do the thing." in text


def test_who_calls_returns_caller_lines(stub_registry: ToolRegistry) -> None:
    r = stub_registry.execute("who_calls", {"qualified_name": "foo.do_thing"})
    assert r.success is True
    assert r.result["count"] == 1


def test_imports_of_lists_strings(stub_registry: ToolRegistry) -> None:
    r = stub_registry.execute("imports_of", {"path": "src/foo.py"})
    assert r.success is True
    assert "os" in r.result["result"]
    assert "sys" in r.result["result"]


def test_neighbors_accepts_hops(stub_registry: ToolRegistry) -> None:
    r = stub_registry.execute(
        "neighbors", {"qualified_name": "foo.do_thing", "hops": 2}
    )
    assert r.success is True
    assert r.result["count"] == 1


def test_repo_summary_returns_formatted_stats(stub_registry: ToolRegistry) -> None:
    r = stub_registry.execute("repo_summary", {})
    assert r.success is True
    text = r.result["result"]
    assert "Files: 12" in text
    assert "Symbols: 99" in text
    assert "python=12" in text


# ---------------------------------------------------------------------------
# Graceful degradation.
# ---------------------------------------------------------------------------


def test_no_api_means_zero_tools_and_no_crash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If no codemap API is importable, registration is a no-op."""
    monkeypatch.setattr(codemap_tools, "_resolve_api", lambda: None)
    monkeypatch.setattr(codemap_tools, "_API_CACHE", {"handle": None, "resolved": True})
    registry = ToolRegistry()
    n = codemap_tools.register_codemap_tools(registry)
    assert n == 0
    assert len(registry) == 0


def test_index_missing_returns_friendly_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing .codemap/index.db should not raise; tools return an error msg."""

    def _boom(*a: Any, **kw: Any) -> Any:
        raise FileNotFoundError(".codemap/index.db")

    handle = codemap_tools._ApiHandle(
        flavor="stub-missing",
        search_code=_boom,
        get_symbol=_boom,
        who_calls=_boom,
        imports_of=_boom,
        neighbors=_boom,
        repo_summary=_boom,
    )
    monkeypatch.setattr(codemap_tools, "_resolve_api", lambda: handle)
    monkeypatch.setattr(
        codemap_tools, "_API_CACHE", {"handle": handle, "resolved": True}
    )
    registry = ToolRegistry()
    codemap_tools.register_codemap_tools(registry)
    r = registry.execute("search_code", {"query": "x"})
    assert r.success is True  # tool ran; payload reports the failure
    assert r.result["success"] is False
    assert "codemap rebuild" in r.result["error"]
