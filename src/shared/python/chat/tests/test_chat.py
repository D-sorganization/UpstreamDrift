"""Tests for the chat module.

Covers:
- Pydantic contract models (ChatMessageRequest, ChatChunkResponse,
  ChatSessionInfo, ChatHistoryResponse)
- Pure utility functions: _session_file_path, _read_shared_session_id,
  _write_shared_session_id
- ChatMessageBubble (mocked Qt) and ChatDockWidget construction (mocked Qt)

Note: PyQt6 widget tests use the qtbot fixture from pytest-qt where
available; pure function tests need no Qt at all.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

# ──────────────────────────────────────────────────────────────────────────────
# Pydantic model tests (no Qt required)
# ──────────────────────────────────────────────────────────────────────────────


class TestChatModels:
    def test_chat_message_request_defaults(self) -> None:
        from chat.models import ChatMessageRequest

        req = ChatMessageRequest(message="hello world")
        assert req.message == "hello world"
        assert req.expertise_level == "beginner"
        assert req.app_context is None

    def test_chat_message_request_with_context(self) -> None:
        from chat.models import ChatMessageRequest

        req = ChatMessageRequest(
            message="how does this work?",
            app_context="mujoco",
            expertise_level="advanced",
        )
        assert req.app_context == "mujoco"
        assert req.expertise_level == "advanced"

    def test_chat_message_request_min_length_validation(self) -> None:
        import pydantic
        from chat.models import ChatMessageRequest

        with pytest.raises(pydantic.ValidationError):
            ChatMessageRequest(message="")  # min_length=1

    def test_chat_message_request_max_length_validation(self) -> None:
        import pydantic
        from chat.models import ChatMessageRequest

        with pytest.raises(pydantic.ValidationError):
            ChatMessageRequest(message="x" * 10001)  # max_length=10000

    def test_chat_chunk_response_defaults(self) -> None:
        from chat.models import ChatChunkResponse

        chunk = ChatChunkResponse(content="hello")
        assert chunk.content == "hello"
        assert chunk.is_final is False
        assert chunk.index == 0

    def test_chat_chunk_response_final(self) -> None:
        from chat.models import ChatChunkResponse

        chunk = ChatChunkResponse(content="done", is_final=True, index=5)
        assert chunk.is_final is True
        assert chunk.index == 5

    def test_chat_session_info(self) -> None:
        from chat.models import ChatSessionInfo

        info = ChatSessionInfo(
            session_id="abc-123",
            message_count=10,
            created_at="2026-01-01T00:00:00",
            last_active="2026-01-01T01:00:00",
            app_contexts=["gasification", "mujoco"],
        )
        assert info.session_id == "abc-123"
        assert info.message_count == 10
        assert len(info.app_contexts) == 2

    def test_chat_session_info_defaults(self) -> None:
        from chat.models import ChatSessionInfo

        info = ChatSessionInfo(
            session_id="xyz",
            message_count=0,
            created_at="now",
            last_active="now",
        )
        assert info.app_contexts == []

    def test_chat_history_response(self) -> None:
        from chat.models import ChatHistoryResponse

        resp = ChatHistoryResponse(
            session_id="abc",
            messages=[
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi!"},
            ],
        )
        assert resp.session_id == "abc"
        assert len(resp.messages) == 2
        assert resp.messages[0]["role"] == "user"


# ──────────────────────────────────────────────────────────────────────────────
# Pure utility function tests (no Qt required)
# ──────────────────────────────────────────────────────────────────────────────


class TestSessionFileFunctions:
    def test_session_file_path_format(self) -> None:
        from chat.chat_dock_widget import _session_file_path

        path = _session_file_path("my_app")
        assert path.name == "active_chat_session.txt"
        assert ".my_app" in str(path)

    def test_read_shared_session_id_missing_file(self, tmp_path: Path) -> None:
        from chat.chat_dock_widget import _read_shared_session_id

        missing = tmp_path / "no_file.txt"
        result = _read_shared_session_id(missing)
        assert result is None

    def test_read_shared_session_id_empty_file(self, tmp_path: Path) -> None:
        from chat.chat_dock_widget import _read_shared_session_id

        f = tmp_path / "session.txt"
        f.write_text("   ", encoding="utf-8")
        result = _read_shared_session_id(f)
        assert result is None

    def test_read_shared_session_id_valid(self, tmp_path: Path) -> None:
        from chat.chat_dock_widget import _read_shared_session_id

        f = tmp_path / "session.txt"
        f.write_text("my-session-123\n", encoding="utf-8")
        result = _read_shared_session_id(f)
        assert result == "my-session-123"

    def test_write_shared_session_id(self, tmp_path: Path) -> None:
        from chat.chat_dock_widget import _write_shared_session_id

        f = tmp_path / "subdir" / "session.txt"
        _write_shared_session_id("abc-def", f)
        assert f.exists()
        assert f.read_text(encoding="utf-8") == "abc-def"

    def test_write_shared_session_id_creates_parents(self, tmp_path: Path) -> None:
        from chat.chat_dock_widget import _write_shared_session_id

        f = tmp_path / "a" / "b" / "c" / "session.txt"
        _write_shared_session_id("xyz", f)
        assert f.exists()

    def test_read_shared_session_id_permission_error(self, tmp_path: Path) -> None:
        """PermissionError is swallowed and returns None."""
        from chat.chat_dock_widget import _read_shared_session_id

        f = tmp_path / "session.txt"
        f.write_text("abc", encoding="utf-8")
        with (
            patch.object(Path, "exists", return_value=True),
            patch.object(Path, "read_text", side_effect=PermissionError("denied")),
        ):
            result = _read_shared_session_id(f)
        assert result is None

    def test_write_shared_session_id_permission_error(self, tmp_path: Path) -> None:
        """PermissionError is swallowed silently."""
        from chat.chat_dock_widget import _write_shared_session_id

        f = tmp_path / "session.txt"
        with patch.object(Path, "mkdir", side_effect=PermissionError("denied")):
            # Should not raise
            _write_shared_session_id("abc", f)

    def test_roundtrip(self, tmp_path: Path) -> None:
        """Write then read gives back the same session ID."""
        from chat.chat_dock_widget import (
            _read_shared_session_id,
            _write_shared_session_id,
        )

        f = tmp_path / "session.txt"
        _write_shared_session_id("roundtrip-id-999", f)
        result = _read_shared_session_id(f)
        assert result == "roundtrip-id-999"


class TestQtRuntimeDiagnostics:
    def test_qt_runtime_diagnostic_reports_probe_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from chat import qt_diagnostics

        def fake_run(*_args: Any, **_kwargs: Any) -> Any:
            return SimpleNamespace(
                returncode=1,
                stderr="ImportError: DLL load failed while importing QtCore",
                stdout="",
            )

        monkeypatch.setattr(qt_diagnostics.subprocess, "run", fake_run)

        diagnostic = qt_diagnostics.diagnose_chat_qt_runtime()

        assert diagnostic.available is False
        assert diagnostic.reason == "import_failed"
        assert "QtCore" in diagnostic.detail

    def test_lazy_chat_dock_loader_raises_structured_qt_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from chat.qt_diagnostics import ChatQtDiagnostic

        from chat import chat_dock_widget

        monkeypatch.setattr(
            chat_dock_widget,
            "diagnose_chat_qt_runtime",
            lambda: ChatQtDiagnostic(
                available=False,
                reason="import_failed",
                detail="broken QtCore",
            ),
        )

        with pytest.raises(chat_dock_widget.ChatQtUnavailableError) as exc_info:
            _ = chat_dock_widget.ChatDockWidget

        assert exc_info.value.diagnostic.reason == "import_failed"
        assert "broken QtCore" in str(exc_info.value)


class _FakeLabel:
    def __init__(self) -> None:
        self.text = ""
        self.stylesheet = ""

    def setText(self, text: str) -> None:
        self.text = text

    def setStyleSheet(self, stylesheet: str) -> None:
        self.stylesheet = stylesheet


class _FakeButton:
    def __init__(self) -> None:
        self.enabled: bool | None = None

    def setEnabled(self, enabled: bool) -> None:
        self.enabled = enabled


class _FakeTimer:
    def __init__(self) -> None:
        self.started_ms: int | None = None

    def start(self, ms: int) -> None:
        self.started_ms = ms


class TestChatDockDisconnectLifecycle:
    def test_disconnected_during_close_does_not_start_reconnect_timer(self) -> None:
        qt_module = pytest.importorskip("chat._chat_dock_widget_qt")
        timer = _FakeTimer()
        widget = SimpleNamespace(
            _is_closing=True,
            _is_streaming=True,
            _send_btn=_FakeButton(),
            _reconnect_timer=timer,
            _status_label=_FakeLabel(),
        )

        qt_module.ChatDockWidget._on_disconnected(widget)

        assert widget._is_streaming is False
        assert widget._send_btn.enabled is True
        assert timer.started_ms is None
        assert widget._status_label.text == ""

    def test_unexpected_disconnect_keeps_existing_reconnect_behavior(self) -> None:
        qt_module = pytest.importorskip("chat._chat_dock_widget_qt")
        timer = _FakeTimer()
        widget = SimpleNamespace(
            _is_closing=False,
            _is_streaming=True,
            _send_btn=_FakeButton(),
            _reconnect_timer=timer,
            _status_label=_FakeLabel(),
        )

        qt_module.ChatDockWidget._on_disconnected(widget)

        assert widget._is_streaming is False
        assert widget._send_btn.enabled is True
        assert timer.started_ms == 3000
        expected_status = (
            "Sidekick API unavailable — retrying in 3s. "
            "Set UD_CHAT_WS_URL if the local API is external."
        )
        assert widget._status_label.text == expected_status


class TestChatDockControls:
    def test_on_new_chat_clicked(self) -> None:
        qt_module = pytest.importorskip("chat._chat_dock_widget_qt")
        sent_messages: list[dict[str, Any]] = []
        widget = SimpleNamespace(_send_ws=lambda msg: sent_messages.append(msg))
        qt_module.ChatDockWidget._on_new_chat_clicked(widget)
        assert sent_messages == [{"action": "new_session"}]

    def test_on_toggle_history(self) -> None:
        qt_module = pytest.importorskip("chat._chat_dock_widget_qt")

        class FakeSidebar:
            def __init__(self) -> None:
                self._visible = False
                self.refreshed = False

            def isVisible(self) -> bool:
                return self._visible

            def setVisible(self, visible: bool) -> None:
                self._visible = visible

            def refresh_lists(self) -> None:
                self.refreshed = True

        sidebar = FakeSidebar()
        widget = SimpleNamespace(_history_sidebar=sidebar)
        # Toggle: False -> True
        qt_module.ChatDockWidget._on_toggle_history(widget)
        assert sidebar._visible is True
        assert sidebar.refreshed is True

        # Toggle: True -> False
        sidebar.refreshed = False
        qt_module.ChatDockWidget._on_toggle_history(widget)
        assert sidebar._visible is False  # type: ignore[unreachable,unused-ignore]
        assert sidebar.refreshed is False  # type: ignore[unreachable,unused-ignore]

    def test_session_created_message_handling(self) -> None:
        qt_module = pytest.importorskip("chat._chat_dock_widget_qt")

        class FakeLayoutItem:
            def __init__(self, widget: Any) -> None:
                self._widget = widget

            def widget(self) -> Any:
                return self._widget

        class FakeWidget:
            def __init__(self) -> None:
                self.deleted = False

            def deleteLater(self) -> None:
                self.deleted = True

        class FakeLayout:
            def __init__(self) -> None:
                # The first item (idx 0) and second item (idx 1) will be removed
                # because the layout count has to remain > 1.
                self.items = [
                    FakeLayoutItem(FakeWidget()),
                    FakeLayoutItem(FakeWidget()),
                    FakeLayoutItem(FakeWidget()),
                ]

            def count(self) -> int:
                return len(self.items)

            def takeAt(self, idx: int) -> Any:
                return self.items.pop(idx)

        class FakeSidebar:
            def __init__(self) -> None:
                self.refreshed = False

            def refresh_lists(self) -> None:
                self.refreshed = True

        layout = FakeLayout()
        sidebar = FakeSidebar()
        bubbles: list[tuple[str, str]] = []

        widget = SimpleNamespace(
            _session_file=Path("dummy_session.txt"),
            _message_layout=layout,
            _message_history=["some", "old", "messages"],
            _add_bubble=lambda role, text: bubbles.append((role, text)),
            _history_sidebar=sidebar,
        )

        with (
            patch.object(
                qt_module.ChatDockWidget, "_set_shared_session_id"
            ) as mock_set_id,
            patch(
                "chat._chat_dock_widget_qt._write_shared_session_id"
            ) as mock_write_id,
        ):
            raw_msg = '{"type": "session_created", "session_id": "new-session-xyz"}'
            qt_module.ChatDockWidget._on_message(widget, raw_msg)

            mock_set_id.assert_called_once_with("new-session-xyz")
            mock_write_id.assert_called_once_with(
                "new-session-xyz", widget._session_file
            )

        assert layout.count() == 1
        assert widget._message_history == []
        assert bubbles == [("assistant", "Hello! How can I help you today?")]
        assert sidebar.refreshed is True

    def test_load_session(self) -> None:
        qt_module = pytest.importorskip("chat._chat_dock_widget_qt")

        class FakeLayoutItem:
            def __init__(self, widget: Any) -> None:
                self._widget = widget

            def widget(self) -> Any:
                return self._widget

        class FakeWidget:
            def __init__(self) -> None:
                self.deleted = False

            def deleteLater(self) -> None:
                self.deleted = True

        class FakeLayout:
            def __init__(self) -> None:
                self.items = [
                    FakeLayoutItem(FakeWidget()),
                    FakeLayoutItem(FakeWidget()),
                ]

            def count(self) -> int:
                return len(self.items)

            def takeAt(self, idx: int) -> Any:
                return self.items.pop(idx)

        class FakeSidebar:
            def __init__(self) -> None:
                self.refreshed = False

            def refresh_lists(self) -> None:
                self.refreshed = True

        layout = FakeLayout()
        sidebar = FakeSidebar()
        connected = False

        def mock_connect() -> None:
            nonlocal connected
            connected = True

        widget = SimpleNamespace(
            _session_file=Path("dummy_session.txt"),
            _message_layout=layout,
            _message_history=["some", "old", "messages"],
            _connect=mock_connect,
            _history_sidebar=sidebar,
        )

        with (
            patch.object(
                qt_module.ChatDockWidget, "_set_shared_session_id"
            ) as mock_set_id,
            patch(
                "chat._chat_dock_widget_qt._write_shared_session_id"
            ) as mock_write_id,
        ):
            qt_module.ChatDockWidget.load_session(widget, "loaded-session-123")

            mock_set_id.assert_called_once_with("loaded-session-123")
            mock_write_id.assert_called_once_with(
                "loaded-session-123", widget._session_file
            )

        assert layout.count() == 1
        assert widget._message_history == []
        assert connected is True
        assert sidebar.refreshed is True
