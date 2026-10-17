"""Tests for the Sidekick chat workspace bridge (Tools issue #2849).

Covers:

* The pure ``WorkspaceContextProtocol`` / ``WorkspaceVariableInfo`` types.
* ``ChatDockWidget`` construction with and without a workspace provider.
* ``_build_workspace_context_block`` system-prompt fragment shape.
* Slash command routing for ``/ws.read``, ``/ws.write`` and ``/plot``.
* Provider precondition violations (``KeyError`` / ``TypeError``).

PyQt6 widget tests guard their imports with :func:`pytest.importorskip`
so the module still collects on CI lanes without Qt installed.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

import pytest

# Workspace protocol tests are pure-Python and run everywhere.
from chat._workspace_protocol import (
    WorkspaceContextProtocol,
    WorkspaceVariableInfo,
)


class FakeWorkspaceProvider:
    """Minimal in-memory ``WorkspaceContextProtocol`` impl for tests."""

    # A long unique string we can grep for in the system-prompt fragment
    # to prove previews-only injection.
    FULL_VALUE_SENTINEL = "FULL_VALUE_DO_NOT_LEAK_TO_PROMPT_3f8a91c2"

    def __init__(self) -> None:
        # Use the sentinel as the *value* but a short preview, so any
        # accidental serialization of the full value into the prompt is
        # detectable.
        self._values: dict[str, Any] = {
            "pressure": [self.FULL_VALUE_SENTINEL] * 4,
            "time": self.FULL_VALUE_SENTINEL,
        }

    def describe(self) -> list[WorkspaceVariableInfo]:
        return [
            WorkspaceVariableInfo(
                name="pressure",
                dtype="float64",
                shape=(1024,),
                preview="[101.3, 101.4, ...]",
            ),
            WorkspaceVariableInfo(
                name="time",
                dtype="float64",
                shape=(1024,),
                preview="[0.0, 0.001, ...]",
            ),
        ]

    def read(self, name: str) -> Any:
        if name not in self._values:
            raise KeyError(name)
        return self._values[name]

    def write(self, name: str, value: Any) -> None:
        if not isinstance(name, str):
            raise TypeError("name must be a str")
        self._values[name] = value


# ──────────────────────────────────────────────────────────────────────────────
# Pure protocol/dataclass tests
# ──────────────────────────────────────────────────────────────────────────────


class TestWorkspaceVariableInfo:
    def test_dataclass_fields(self) -> None:
        info = WorkspaceVariableInfo(
            name="x",
            dtype="float64",
            shape=(4,),
            preview="[1.0, 2.0, ...]",
        )
        assert info.name == "x"
        assert info.dtype == "float64"
        assert info.shape == (4,)
        assert info.preview == "[1.0, 2.0, ...]"

    def test_scalar_shape_is_none(self) -> None:
        info = WorkspaceVariableInfo(
            name="alpha",
            dtype="float",
            shape=None,
            preview="3.14",
        )
        assert info.shape is None

    def test_immutable(self) -> None:
        info = WorkspaceVariableInfo(
            name="x",
            dtype="float64",
            shape=None,
            preview="0",
        )
        # frozen dataclasses raise FrozenInstanceError (a TypeError
        # subclass) on attribute assignment.
        with pytest.raises((TypeError, AttributeError)):
            info.name = "y"  # type: ignore[misc]


class TestWorkspaceContextProtocol:
    def test_fake_is_runtime_checkable(self) -> None:
        provider = FakeWorkspaceProvider()
        assert isinstance(provider, WorkspaceContextProtocol)

    def test_fake_describe_shape(self) -> None:
        provider = FakeWorkspaceProvider()
        items = provider.describe()
        assert len(items) == 2
        assert {item.name for item in items} == {"pressure", "time"}

    def test_fake_read_unknown_raises_key_error(self) -> None:
        provider = FakeWorkspaceProvider()
        with pytest.raises(KeyError):
            provider.read("does_not_exist")

    def test_fake_write_non_str_raises_type_error(self) -> None:
        provider = FakeWorkspaceProvider()
        with pytest.raises(TypeError):
            provider.write(123, 42)  # type: ignore[arg-type]


# ──────────────────────────────────────────────────────────────────────────────
# Qt-backed ChatDockWidget tests
# ──────────────────────────────────────────────────────────────────────────────

# Offscreen platform so Qt does not need a display on CI. Setting this
# before any PyQt6 import is safe even on systems without PyQt6 — we
# only attempt the import inside the qapp fixture, where importorskip
# kicks in and skips the Qt-dependent suite cleanly.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="module")
def qapp() -> Any:
    pytest.importorskip("PyQt6")
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication(sys.argv)
    return app


@pytest.fixture
def chat_module(qapp: Any) -> Any:
    from chat import _chat_dock_widget_qt

    return _chat_dock_widget_qt


def _make_dock(
    chat_module: Any,
    *,
    workspace_provider: Any = None,
    plot_request_sink: Any = None,
) -> Any:
    dock = chat_module.ChatDockWidget(
        connection=chat_module.ChatConnectionConfig(
            app_context="test",
            app_name="test_workspace_bridge",
        ),
        integrations=chat_module.ChatIntegrationHooks(
            workspace_provider=workspace_provider,
            plot_request_sink=plot_request_sink,
        ),
    )
    # The dock subclasses QDockWidget; connect-on-show defers networking,
    # so simply constructing it does not hit the network.
    return dock


class TestChatDockWidgetConstruction:
    def test_construct_with_provider_stores_provider(
        self,
        chat_module: Any,
    ) -> None:
        provider = FakeWorkspaceProvider()
        dock = _make_dock(chat_module, workspace_provider=provider)
        try:
            assert dock._workspace_provider is provider
            assert dock._plot_request_sink is None
        finally:
            dock.close()

    def test_construct_without_provider(self, chat_module: Any) -> None:
        dock = _make_dock(chat_module)
        try:
            assert dock._workspace_provider is None
            assert dock._plot_request_sink is None
        finally:
            dock.close()

    def test_construct_with_plot_sink(self, chat_module: Any) -> None:
        received: list[Any] = []
        dock = _make_dock(chat_module, plot_request_sink=received.append)
        try:
            assert dock._plot_request_sink is not None
        finally:
            dock.close()


class TestWorkspaceContextBlock:
    def test_no_provider_returns_empty(self, chat_module: Any) -> None:
        dock = _make_dock(chat_module)
        try:
            assert dock._build_workspace_context_block() == ""
        finally:
            dock.close()

    def test_lists_each_variable(self, chat_module: Any) -> None:
        provider = FakeWorkspaceProvider()
        dock = _make_dock(chat_module, workspace_provider=provider)
        try:
            block = dock._build_workspace_context_block()
        finally:
            dock.close()
        assert "Available workspace variables:" in block
        assert "pressure" in block
        assert "time" in block
        assert "float64" in block
        # Previews appear:
        assert "101.3" in block
        # Full values must never leak into the prompt:
        assert FakeWorkspaceProvider.FULL_VALUE_SENTINEL not in block

    def test_empty_describe_returns_empty(self, chat_module: Any) -> None:
        class EmptyProvider:
            def describe(self) -> list[WorkspaceVariableInfo]:
                return []

            def read(self, name: str) -> Any:  # pragma: no cover - unused
                raise KeyError(name)

            def write(self, name: str, value: Any) -> None:  # pragma: no cover
                if not isinstance(name, str):
                    raise TypeError("name must be a str")

        dock = _make_dock(chat_module, workspace_provider=EmptyProvider())
        try:
            assert dock._build_workspace_context_block() == ""
        finally:
            dock.close()


class TestSlashCommandRouting:
    def test_ws_read_routes_to_provider(self, chat_module: Any) -> None:
        provider = FakeWorkspaceProvider()
        dock = _make_dock(chat_module, workspace_provider=provider)
        try:
            dock._input_edit.setPlainText("/ws.read time")
            dock._handle_slash_command("/ws.read time")
            # The handler appends an assistant bubble containing the value.
            text = dock._get_thread_markdown()
        finally:
            dock.close()
        assert "time" in text
        assert FakeWorkspaceProvider.FULL_VALUE_SENTINEL in text  # value printed back

    def test_ws_read_unknown_reports_not_found(self, chat_module: Any) -> None:
        provider = FakeWorkspaceProvider()
        dock = _make_dock(chat_module, workspace_provider=provider)
        try:
            dock._handle_slash_command("/ws.read missing_var")
            text = dock._get_thread_markdown()
        finally:
            dock.close()
        assert "not found" in text.lower()

    def test_ws_read_without_provider(self, chat_module: Any) -> None:
        dock = _make_dock(chat_module)
        try:
            dock._handle_slash_command("/ws.read whatever")
            text = dock._get_thread_markdown()
        finally:
            dock.close()
        assert "not available" in text.lower()

    def test_ws_write_calls_provider(self, chat_module: Any) -> None:
        provider = FakeWorkspaceProvider()
        dock = _make_dock(chat_module, workspace_provider=provider)
        try:
            dock._handle_slash_command('/ws.write foo {"a": 1}')
        finally:
            dock.close()
        assert provider.read("foo") == {"a": 1}

    def test_ws_write_invalid_json(self, chat_module: Any) -> None:
        provider = FakeWorkspaceProvider()
        dock = _make_dock(chat_module, workspace_provider=provider)
        try:
            dock._handle_slash_command("/ws.write foo not-json-:(")
            text = dock._get_thread_markdown()
        finally:
            dock.close()
        assert "json" in text.lower()

    def test_plot_calls_sink_with_parsed_spec(self, chat_module: Any) -> None:
        received: list[Any] = []
        dock = _make_dock(chat_module, plot_request_sink=received.append)
        spec = {"source": "expression_range", "expression": "sin(x)"}
        try:
            dock._handle_slash_command(f"/plot {json.dumps(spec)}")
        finally:
            dock.close()
        assert received == [spec]

    def test_plot_without_sink(self, chat_module: Any) -> None:
        dock = _make_dock(chat_module)
        try:
            dock._handle_slash_command('/plot {"source": "x"}')
            text = dock._get_thread_markdown()
        finally:
            dock.close()
        assert "not available" in text.lower()

    def test_unknown_slash_command_still_dispatches_skill(
        self,
        chat_module: Any,
    ) -> None:
        """Unrelated slash commands keep their existing skill-invoke path."""
        sent: list[dict[str, Any]] = []
        dock = _make_dock(chat_module)
        try:
            dock._send_ws = sent.append  # type: ignore[method-assign]
            dock._handle_slash_command("/condense")
        finally:
            dock.close()
        # ``/condense`` is not a workspace command, so it must still hit the
        # WebSocket skill-invoke path. Workspace commands deliberately do
        # not call _send_ws.
        assert any(payload.get("action") == "skill_invoke" for payload in sent)


class TestProviderPreconditions:
    def test_write_non_str_raises_type_error(self) -> None:
        provider = FakeWorkspaceProvider()
        with pytest.raises(TypeError):
            provider.write(42, "value")  # type: ignore[arg-type]

    def test_read_missing_raises_key_error(self) -> None:
        provider = FakeWorkspaceProvider()
        with pytest.raises(KeyError):
            provider.read("nope")
