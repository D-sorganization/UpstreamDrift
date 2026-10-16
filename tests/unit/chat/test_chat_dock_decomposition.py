"""Regression tests for the chat-dock compatibility-shell decomposition."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.unit


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
