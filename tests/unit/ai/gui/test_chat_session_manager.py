"""Regression tests for persisted chat session metadata."""

from __future__ import annotations

import json

import pytest

from src.shared.python.ai.gui.session_manager import ChatSessionManager

pytestmark = pytest.mark.unit


def test_list_sessions_sorts_mixed_timezone_timestamps(tmp_path) -> None:
    old_session = {
        "session_id": "old",
        "metadata": {},
        "messages": [
            {
                "role": "user",
                "content": "old",
                "timestamp": "2024-01-01T00:00:00",
            }
        ],
    }
    new_session = {
        "session_id": "new",
        "metadata": {},
        "messages": [
            {
                "role": "user",
                "content": "new",
                "timestamp": "2025-01-01T00:00:00+00:00",
            }
        ],
    }
    (tmp_path / "old.json").write_text(json.dumps(old_session), encoding="utf-8")
    (tmp_path / "new.json").write_text(json.dumps(new_session), encoding="utf-8")

    sessions = ChatSessionManager(tmp_path).list_sessions()

    assert [session["id"] for session in sessions] == ["new", "old"]
