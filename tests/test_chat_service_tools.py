"""Top-level tests for ChatService tool calling integration (issue #3162).

Asserts that:
- Tool-call chunks from the adapter are executed via the registry.
- A tool_call_result event is emitted after a successful tool execution.
- The tool result is appended to the conversation context so a restart
  pass could narrate it.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.shared.python.ai.types import AgentChunk

pytestmark = pytest.mark.unit


@pytest.fixture()
def chat_service(tmp_path: Any) -> Any:
    """ChatService with mock adapter and ephemeral session dir."""
    with patch("src.api.services.chat_service.ChatService._load_adapter"):
        from src.api.services.chat_service import ChatService

        svc = ChatService()
        svc.PERSIST_DIR = tmp_path / "chat_sessions"
        svc._adapter = MagicMock()
        return svc


def _drain(svc: Any, sid: str) -> list[str]:
    async def _run() -> list[str]:
        out: list[str] = []
        async for chunk in svc.stream_response(sid):
            out.append(chunk)
        return out

    return asyncio.get_event_loop().run_until_complete(_run())


def test_explain_concept_tool_call_emits_result(chat_service: Any) -> None:
    """A tool_call chunk for explain_concept runs the tool and emits result."""
    svc = chat_service
    ctx = svc.get_or_create_session(None)
    svc.add_user_message(ctx.session_id, "What is inverse dynamics?")

    tool_chunk = AgentChunk(
        content="",
        tool_call_delta={
            "id": "tc_1",
            "name": "explain_concept",
            "arguments": json.dumps({"term": "inverse_dynamics", "expertise_level": 1}),
        },
        is_final=True,
    )
    svc._adapter.stream_response.return_value = iter([tool_chunk])

    out = _drain(svc, ctx.session_id)
    assert any("tool_call_result" in s for s in out)
    # A tool role message was appended to the real session.
    tool_msgs = [m for m in ctx.messages if m.role == "tool"]
    assert len(tool_msgs) >= 1


def test_tool_call_followed_by_text_chunk(chat_service: Any) -> None:
    """After the tool result is appended, streaming restarts for narration."""
    svc = chat_service
    ctx = svc.get_or_create_session(None)
    svc.add_user_message(ctx.session_id, "Explain backspin")

    first_pass = [
        AgentChunk(
            content="",
            tool_call_delta={
                "id": "tc_2",
                "name": "explain_concept",
                "arguments": json.dumps({"term": "backspin"}),
            },
            is_final=True,
        )
    ]
    second_pass = [AgentChunk(content="Here is the answer.", is_final=True)]
    svc._adapter.stream_response.side_effect = [iter(first_pass), iter(second_pass)]

    out = _drain(svc, ctx.session_id)
    # The narration text from the second pass must appear.
    combined = "".join(s for s in out if not s.startswith("\x00EVENT\x00"))
    assert "Here is the answer." in combined
    # Adapter was invoked twice (restart-once semantics).
    assert svc._adapter.stream_response.call_count == 2


def test_max_tool_calls_guard(chat_service: Any) -> None:
    """The loop-guard stops after MAX_TOOL_CALLS_PER_TURN tool calls."""
    svc = chat_service
    ctx = svc.get_or_create_session(None)
    svc.add_user_message(ctx.session_id, "Loop forever")

    def _loop(*_args: Any, **_kwargs: Any) -> Any:
        return iter(
            [
                AgentChunk(
                    content="",
                    tool_call_delta={
                        "id": f"tc_{len(ctx.messages)}",
                        "name": "list_physics_engines",
                        "arguments": "{}",
                    },
                    is_final=True,
                )
            ]
        )

    svc._adapter.stream_response.side_effect = _loop

    out = _drain(svc, ctx.session_id)
    assert any("max tool calls exceeded" in s for s in out)
