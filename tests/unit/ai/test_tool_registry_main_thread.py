"""Tests for ToolRegistry GUI-thread dispatch of main-thread tools.

The registry routes tools flagged ``requires_main_thread`` through an
installed dispatcher (the GUI layer marshals them onto the GUI thread);
plain tools run inline. These tests use a fake dispatcher and need no Qt.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from src.shared.python.ai.tool_registry import Tool, ToolRegistry, ToolResult

pytestmark = [pytest.mark.unit]


def _make_registry_with_tool(*, requires_main_thread: bool) -> ToolRegistry:
    registry = ToolRegistry()
    registry._tools["touch"] = Tool(
        name="touch",
        description="A tool that would touch widgets.",
        handler=lambda: "ok",
        requires_main_thread=requires_main_thread,
    )
    return registry


def test_flagged_tool_routes_through_dispatcher() -> None:
    registry = _make_registry_with_tool(requires_main_thread=True)
    calls: list[Callable[[], ToolResult]] = []

    def dispatcher(thunk: Callable[[], ToolResult]) -> ToolResult:
        calls.append(thunk)
        return thunk()

    registry.set_main_thread_dispatcher(dispatcher)
    result = registry.execute("touch", {})
    assert result.success
    assert result.result == "ok"
    assert len(calls) == 1  # the dispatcher saw exactly one thunk


def test_unflagged_tool_runs_inline_even_with_dispatcher() -> None:
    registry = _make_registry_with_tool(requires_main_thread=False)
    calls: list[Callable[[], ToolResult]] = []

    def dispatcher(thunk: Callable[[], ToolResult]) -> ToolResult:
        calls.append(thunk)
        return thunk()

    registry.set_main_thread_dispatcher(dispatcher)
    result = registry.execute("touch", {})
    assert result.success
    assert calls == []  # dispatcher must NOT be used for plain tools


def test_flagged_tool_runs_inline_without_dispatcher() -> None:
    # Headless / no-GUI: a flagged tool still runs (inline), not crashes.
    registry = _make_registry_with_tool(requires_main_thread=True)
    result = registry.execute("touch", {})
    assert result.success
    assert result.result == "ok"


def test_set_main_thread_dispatcher_rejects_non_callable() -> None:
    registry = ToolRegistry()
    with pytest.raises(TypeError, match="callable"):
        registry.set_main_thread_dispatcher(object())  # type: ignore[arg-type]


def test_clearing_dispatcher_restores_inline_execution() -> None:
    registry = _make_registry_with_tool(requires_main_thread=True)
    calls: list[Callable[[], ToolResult]] = []

    def dispatcher(thunk: Callable[[], ToolResult]) -> ToolResult:
        calls.append(thunk)
        return thunk()

    registry.set_main_thread_dispatcher(dispatcher)
    registry.set_main_thread_dispatcher(None)
    result = registry.execute("touch", {})
    assert result.success
    assert calls == []


def test_tool_default_does_not_require_main_thread() -> None:
    tool = Tool(name="x", description="d", handler=lambda: None)
    assert tool.requires_main_thread is False


def test_decorator_can_mark_tool_as_main_thread_only() -> None:
    registry = ToolRegistry()
    calls: list[Callable[[], ToolResult]] = []

    @registry.register(
        "touch",
        "A tool that would touch widgets.",
        requires_main_thread=True,
    )
    def touch() -> str:
        return "ok"

    def dispatcher(thunk: Callable[[], ToolResult]) -> ToolResult:
        calls.append(thunk)
        return thunk()

    registry.set_main_thread_dispatcher(dispatcher)
    result = registry.execute("touch", {})
    tool = registry.get_tool("touch")

    assert tool is not None
    assert tool.requires_main_thread is True
    assert result.success
    assert result.result == "ok"
    assert len(calls) == 1
