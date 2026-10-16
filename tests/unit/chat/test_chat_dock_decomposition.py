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


# ---------------------------------------------------------------------------
# Constructor contract (Tools #4896 / UD #9357): config objects, not 16 kwargs
# ---------------------------------------------------------------------------


def test_chat_dock_constructor_takes_config_objects_only() -> None:
    import inspect

    from chat import _chat_dock_widget_qt as qt_module

    params = inspect.signature(qt_module.ChatDockWidget.__init__).parameters
    assert list(params) == [
        "self",
        "connection",
        "presentation",
        "integrations",
        "parent",
    ]
    assert all(p.default is None for n, p in params.items() if n != "self")


def test_chat_config_defaults_preserve_legacy_defaults() -> None:
    from chat import ChatConnectionConfig, ChatIntegrationHooks
    from chat import ChatPresentationConfig

    connection = ChatConnectionConfig()
    assert connection.app_context == "unknown"
    assert connection.app_name == "shared_chat"
    assert connection.server_url is None
    assert connection.session_id is None
    assert connection.ws_path_template == "/api/ws/chat/{session_id}"
    assert connection.project_root is None

    presentation = ChatPresentationConfig()
    assert presentation.placeholder_text == "Ask a question..."
    assert presentation.accent_color == "#FF8800"
    assert presentation.theme_provider is None
    assert presentation.auto_index_on_open is False

    hooks = ChatIntegrationHooks()
    assert hooks.terminal_registry is None
    assert hooks.workspace_provider is None
    assert hooks.plot_request_sink is None
    assert hooks.session_manager is None
    assert hooks.memory_manager_factory is None


def test_chat_dock_rejects_missing_app_context() -> None:
    from chat import _chat_dock_widget_qt as qt_module

    with pytest.raises(ValueError, match="app_context"):
        qt_module.ChatDockWidget(
            connection=qt_module.ChatConnectionConfig(app_context=None)  # type: ignore[arg-type]
        )
