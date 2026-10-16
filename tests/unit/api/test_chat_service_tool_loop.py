"""Characterization tests for the ChatService stream worker tool loop (#12846).

Pins the observable behaviour of ``ChatService.stream_response`` for the
multi-turn tool-call loop, error sentinel handling and stop-event handling so
the worker can be lifted out of its nested closure without behaviour change.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


def _make_service(tmp_path) -> Any:
    with patch("src.api.services.chat_service.ChatService._load_adapter"):
        from src.api.services.chat_service import ChatService

        svc = ChatService()
        svc.PERSIST_DIR = tmp_path / "chat_sessions"
        svc._adapter = MagicMock()
        # Real persistence runs (spied): the stop_event no longer leaks into
        # the persisted metadata (#12872).
        svc._persist_session = MagicMock(wraps=svc._persist_session)
        return svc


def _chunk(content: str = "", tool_call_delta: dict | None = None) -> Any:
    return SimpleNamespace(content=content, tool_call_delta=tool_call_delta)


def _collect(svc: Any, session_id: str) -> list[Any]:
    async def _run() -> list[Any]:
        return [item async for item in svc.stream_response(session_id)]

    return asyncio.run(_run())


def _session(svc: Any) -> Any:
    ctx = svc.get_or_create_session(None)
    svc.add_user_message(ctx.session_id, "hi")
    return ctx


@pytest.mark.unit
def test_plain_response_streams_and_persists(tmp_path) -> None:
    svc = _make_service(tmp_path)
    ctx = _session(svc)
    svc._adapter.stream_response.side_effect = lambda *_a: iter(
        [_chunk("Hel"), _chunk("lo")]
    )
    svc._persist_session.reset_mock()
    items = _collect(svc, ctx.session_id)
    assert items == ["Hel", "lo"]
    assert ctx.messages[-1].role == "assistant"
    assert ctx.messages[-1].content == "Hello"
    assert svc._persist_session.call_count == 1


@pytest.mark.unit
def test_tool_call_loop_executes_tool_then_continues(tmp_path) -> None:
    svc = _make_service(tmp_path)
    ctx = _session(svc)
    svc._tool_registry = MagicMock()
    svc._tool_registry.execute.return_value = SimpleNamespace(
        success=True, result="42", error=None
    )
    delta = {
        "tool_calls": [
            {
                "index": 0,
                "id": "c1",
                "function": {"name": "t", "arguments": '{"a":'},
            }
        ]
    }
    delta2 = {"tool_calls": [{"index": 0, "function": {"arguments": "1}"}}]}
    turns = iter(
        [
            iter([_chunk("", delta), _chunk("", delta2)]),
            iter([_chunk("done")]),
        ]
    )
    svc._adapter.stream_response.side_effect = lambda *_a: next(turns)
    items = _collect(svc, ctx.session_id)
    assert items == [
        {"type": "tool_call_started", "tool": "t"},
        {"type": "tool_call_result", "tool": "t", "detail": "42"},
        "done",
    ]
    svc._tool_registry.execute.assert_called_once_with("t", {"a": 1})
    roles = [m.role for m in ctx.messages]
    assert roles[-3:] == ["assistant", "tool", "assistant"]


@pytest.mark.unit
def test_tool_failure_and_exception_emit_tool_error(tmp_path) -> None:
    svc = _make_service(tmp_path)
    ctx = _session(svc)
    svc._tool_registry = MagicMock()
    svc._tool_registry.execute.side_effect = RuntimeError("boom")
    delta = {
        "tool_calls": [
            {"index": 0, "id": "c1", "function": {"name": "t", "arguments": "not-json"}}
        ]
    }
    turns = iter([iter([_chunk("", delta)]), iter([])])
    svc._adapter.stream_response.side_effect = lambda *_a: next(turns)
    items = _collect(svc, ctx.session_id)
    assert items[0] == {"type": "tool_call_started", "tool": "t"}
    assert items[1] == {"type": "tool_error", "tool": "t", "detail": "boom"}
    svc._tool_registry.execute.assert_called_once_with("t", {"raw": "not-json"})


@pytest.mark.unit
def test_adapter_exception_yields_error_marker(tmp_path) -> None:
    svc = _make_service(tmp_path)
    ctx = _session(svc)

    def _boom(*_a: Any) -> Any:
        raise ValueError("bad adapter")

    svc._adapter.stream_response.side_effect = _boom
    items = _collect(svc, ctx.session_id)
    assert items == ["\n[Error: bad adapter]"]


@pytest.mark.unit
def test_missing_adapter_and_session_messages(tmp_path) -> None:
    svc = _make_service(tmp_path)
    assert _collect(svc, "nope") == ["Session not found."]
    svc._adapter = None
    svc._backend_error = "oops"
    assert _collect(svc, "x") == ["oops"]
