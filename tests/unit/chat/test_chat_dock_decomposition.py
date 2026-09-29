"""Regression tests for the chat-dock compatibility-shell decomposition."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest


pytestmark = pytest.mark.unit


def test_message_handler_delegates_to_runtime_helper() -> None:
    from chat import _chat_dock_widget_qt as qt_module

    widget = SimpleNamespace()

    with patch("chat._chat_dock_widget_qt._runtime.handle_message") as handler:
        qt_module.ChatDockWidget._on_message(widget, '{"type": "unknown"}')

    args, kwargs = handler.call_args
    assert args == (widget, '{"type": "unknown"}')
    assert callable(kwargs["set_shared_session_id"])
    assert callable(kwargs["write_shared_session_id"])


def test_terminal_setup_delegates_to_runtime_helper() -> None:
    from chat import _chat_dock_widget_qt as qt_module

    widget = SimpleNamespace()

    with patch("chat._chat_dock_widget_qt._runtime.populate_shell_combo") as populate:
        qt_module.ChatDockWidget._populate_shell_combo(widget)

    populate.assert_called_once_with(widget)


def test_collapsed_state_delegates_to_ui_builder() -> None:
    from chat import _chat_dock_widget_qt as qt_module

    widget = SimpleNamespace()

    with patch(
        "chat._chat_dock_widget_qt._runtime.set_chat_dock_collapsed"
    ) as set_collapsed:
        qt_module.ChatDockWidget.set_collapsed(widget, True)

    set_collapsed.assert_called_once_with(widget, True)


def test_terminal_start_preserves_runtime_payload_contract() -> None:
    from chat._qt import runtime

    class _Combo:
        def __init__(self, value: str) -> None:
            self._value = value

        def currentData(self) -> str:
            return self._value

    class _Output:
        def __init__(self) -> None:
            self.cleared = False

        def clear(self) -> None:
            self.cleared = True

    lines: list[str] = []
    payloads: list[dict[str, object]] = []
    sync_calls: list[None] = []
    output = _Output()
    dock = SimpleNamespace(
        _terminal_runtime_available=True,
        _terminal_session_id=None,
        _terminal_start_pending=False,
        _shell_combo=_Combo("powershell"),
        _provider_combo=_Combo("codex"),
        _terminal_output=output,
        _project_root=Path("project"),
        _app_context="test-app",
        _append_terminal_line=lines.append,
        _sync_terminal_controls=lambda: sync_calls.append(None),
        _send_ws=payloads.append,
    )

    runtime.on_terminal_start(dock)

    assert dock._terminal_start_pending is True
    assert output.cleared is True
    assert sync_calls == [None]
    assert lines == ["[terminal] starting..."]
    assert payloads == [
        {
            "action": "terminal_start",
            "project_root": "project",
            "shell_id": "powershell",
            "provider_id": "codex",
            "app_context": "test-app",
        }
    ]
