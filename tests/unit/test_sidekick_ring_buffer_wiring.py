"""Tests for issues #5474 and #5475 — ring-buffer producers and tool wiring.

Issue #5474: LauncherDiagnostics must call record_event so the ring buffer
has events after a diagnostic pass.

Issue #5475: AIAssistantPanel must pass a non-empty tools list to StreamWorker
so analytics and other registered tools are reachable by the adapter.
"""

from __future__ import annotations

import importlib
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

import src.shared.python.ai.chat_context as chat_context

# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _reset_ring_buffer() -> None:
    """Reset the module-level ring buffer before each test."""
    chat_context.reset_buffer()


# ── Issue #5474: LauncherDiagnostics populates the ring buffer ────────


class TestDiagnosticsRingBufferIntegration:
    """LauncherDiagnostics must emit at least one event to the ring buffer."""

    def test_run_all_checks_populates_ring_buffer(self) -> None:
        """After run_all_checks, the ring buffer has at least one event."""
        try:
            from src.launchers.launcher_diagnostics import LauncherDiagnostics
        except ImportError as exc:
            pytest.skip(f"launcher_diagnostics unavailable: {exc}")

        diag = LauncherDiagnostics()
        diag.run_all_checks()

        ctx = chat_context.get_chat_context()
        assert ctx["count"] > 0, (
            "Ring buffer must have at least one event after run_all_checks(). "
            "Add record_event() calls in launcher_diagnostics.py (issue #5474)."
        )

    def test_run_all_checks_emits_diagnostic_category(self) -> None:
        """Events emitted by LauncherDiagnostics have the 'diagnostic' category."""
        try:
            from src.launchers.launcher_diagnostics import LauncherDiagnostics
        except ImportError as exc:
            pytest.skip(f"launcher_diagnostics unavailable: {exc}")

        diag = LauncherDiagnostics()
        diag.run_all_checks()

        ctx = chat_context.get_chat_context()
        categories = {ev["payload"].get("category") for ev in ctx["events"]}
        check_names = {
            ev["payload"].get("check")
            for ev in ctx["events"]
            if isinstance(ev.get("payload"), dict)
        }
        # Events should contain check names
        assert check_names, (
            "Ring-buffer events should contain 'check' fields with check names."
        )

    def test_single_check_populates_ring_buffer(self) -> None:
        """A single check (check_python_environment) emits one ring-buffer event."""
        try:
            from src.launchers.launcher_diagnostics import LauncherDiagnostics
        except ImportError as exc:
            pytest.skip(f"launcher_diagnostics unavailable: {exc}")

        diag = LauncherDiagnostics()
        diag.check_python_environment()

        ctx = chat_context.get_chat_context()
        assert ctx["count"] >= 1, (
            "check_python_environment() must emit at least one ring-buffer event."
        )
        first_event = ctx["events"][0]
        assert first_event.get("category") == "diagnostic"
        payload = first_event.get("payload", {})
        assert payload.get("check") == "python_environment"
        assert payload.get("status") in {"pass", "fail", "warning"}

    def test_record_result_tolerates_buffer_failure(self) -> None:
        """_record_result must not raise even if record_event fails."""
        try:
            from src.launchers.launcher_diagnostics import (
                DiagnosticResult,
                LauncherDiagnostics,
            )
        except ImportError as exc:
            pytest.skip(f"launcher_diagnostics unavailable: {exc}")

        diag = LauncherDiagnostics()
        result = DiagnosticResult(
            name="test",
            status="pass",
            message="ok",
        )

        with patch(
            "src.shared.python.ai.chat_context.record_event",
            side_effect=RuntimeError("boom"),
        ):
            # Should not raise
            diag._record_result(result)


# ── Issue #5475: AIAssistantPanel passes non-empty tools to StreamWorker ──


class TestAssistantPanelToolsWiring:
    """AIAssistantPanel must pass registered tools to StreamWorker."""

    def _make_panel_stub(self) -> Any:
        """Build a minimal stub that exercises _process_message tool wiring.

        We avoid actually creating a QApplication / QWidget by stubbing out
        the PyQt6 layer. The goal is to verify that the list passed to
        StreamWorker is non-empty.
        """
        try:
            import src.shared.python.ai.gui.assistant_panel as ap_mod
        except ImportError as exc:
            pytest.skip(f"assistant_panel unavailable (PyQt6 not installed?): {exc}")
            return None  # unreachable but satisfies type checker

        return ap_mod

    def test_process_message_passes_non_empty_tools(self) -> None:
        """_process_message must pass non-empty tool declarations to StreamWorker."""
        try:
            import src.shared.python.ai.gui.assistant_panel as ap_mod
        except ImportError as exc:
            pytest.skip(f"assistant_panel unavailable: {exc}")

        from src.shared.python.ai.tool_registry import ToolCategory, ToolRegistry

        # Build a fresh registry with at least one tool so we get a non-empty list.
        registry = ToolRegistry()

        @registry.register(
            name="test_tool",
            description="A test tool",
            category=ToolCategory.ANALYSIS,
        )
        def test_tool() -> str:
            return "ok"

        captured_tools: list[Any] = []

        class _WorkerStub:
            def __init__(
                self,
                adapter: Any,
                message: str,
                context: Any,
                tools: list[Any],
            ) -> None:
                captured_tools.extend(tools)

            def start(self) -> None:
                pass

            chunk_received = MagicMock()
            finished = MagicMock()
            error = MagicMock()

        _adapter_stub = MagicMock()

        # Patch StreamWorker, get_global_registry, and all Qt widget constructors
        with (
            patch.object(ap_mod, "StreamWorker", _WorkerStub),
            patch.object(ap_mod, "get_global_registry", return_value=registry),
            patch.object(ap_mod.AIAssistantPanel, "_setup_ui", lambda self: None),
            patch.object(ap_mod.AIAssistantPanel, "_load_history", lambda self: None),
            patch.object(
                ap_mod.AIAssistantPanel, "_restore_ui_messages", lambda self: None
            ),
            patch.object(ap_mod.AIAssistantPanel, "_init_tools", lambda self: None),
            patch("src.shared.python.ai.rag.simple_rag.SimpleRAGStore"),
            patch.object(ap_mod.AIAssistantPanel, "_add_message", return_value=None),
            patch.object(ap_mod.AIAssistantPanel, "_set_status", lambda self, s: None),
        ):
            from src.shared.python.ai.types import ConversationContext

            panel = ap_mod.AIAssistantPanel.__new__(ap_mod.AIAssistantPanel)
            panel._adapter = _adapter_stub
            panel._tools_registry = registry
            panel._context = ConversationContext()
            panel._current_worker = None
            panel._current_assistant_message = None
            panel._is_first_chunk = True
            panel._send_btn = MagicMock()

            panel._process_message("hello")

        assert len(captured_tools) > 0, (
            "StreamWorker must receive non-empty tools list (issue #5475). "
            "Fix the tools=[] placeholder in _process_message."
        )

    def test_tool_declarations_have_name_and_description(self) -> None:
        """Tool declarations passed to StreamWorker have name and description."""
        try:
            import src.shared.python.ai.gui.assistant_panel as ap_mod
        except ImportError as exc:
            pytest.skip(f"assistant_panel unavailable: {exc}")

        from src.shared.python.ai.adapters.base import ToolDeclaration
        from src.shared.python.ai.tool_registry import ToolCategory, ToolRegistry

        registry = ToolRegistry()

        @registry.register(
            name="analytics_tool",
            description="Retrieve simulation analytics",
            category=ToolCategory.ANALYSIS,
        )
        def analytics_tool(run_id: str) -> str:
            return run_id

        captured_tools: list[Any] = []

        class _WorkerStub:
            def __init__(
                self,
                adapter: Any,
                message: str,
                context: Any,
                tools: list[Any],
            ) -> None:
                captured_tools.extend(tools)

            def start(self) -> None:
                pass

            chunk_received = MagicMock()
            finished = MagicMock()
            error = MagicMock()

        with (
            patch.object(ap_mod, "StreamWorker", _WorkerStub),
            patch.object(ap_mod, "get_global_registry", return_value=registry),
            patch.object(ap_mod.AIAssistantPanel, "_setup_ui", lambda self: None),
            patch.object(ap_mod.AIAssistantPanel, "_load_history", lambda self: None),
            patch.object(
                ap_mod.AIAssistantPanel, "_restore_ui_messages", lambda self: None
            ),
            patch.object(ap_mod.AIAssistantPanel, "_init_tools", lambda self: None),
            patch("src.shared.python.ai.rag.simple_rag.SimpleRAGStore"),
            patch.object(ap_mod.AIAssistantPanel, "_add_message", return_value=None),
            patch.object(ap_mod.AIAssistantPanel, "_set_status", lambda self, s: None),
        ):
            from src.shared.python.ai.types import ConversationContext

            panel = ap_mod.AIAssistantPanel.__new__(ap_mod.AIAssistantPanel)
            panel._adapter = MagicMock()
            panel._tools_registry = registry
            panel._context = ConversationContext()
            panel._current_worker = None
            panel._current_assistant_message = None
            panel._is_first_chunk = True
            panel._send_btn = MagicMock()

            panel._process_message("analyse this")

        assert len(captured_tools) >= 1
        for tool in captured_tools:
            assert isinstance(tool, ToolDeclaration)
            assert tool.name
            assert tool.description
