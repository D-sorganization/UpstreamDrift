"""Timeout handling for tool calls in ChatService (issue #3162)."""

from __future__ import annotations

import asyncio
import time
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.shared.python.ai.tool_registry import ToolCategory
from src.shared.python.ai.types import AgentChunk

pytestmark = pytest.mark.unit


@pytest.fixture()
def chat_service(tmp_path: Any) -> Any:
    with patch("src.api.services.chat_service.ChatService._load_adapter"):
        from src.api.services.chat_service import ChatService

        svc = ChatService()
        svc.PERSIST_DIR = tmp_path / "chat_sessions"
        svc._adapter = MagicMock()
        return svc


def test_slow_tool_emits_tool_error(chat_service: Any, monkeypatch: Any) -> None:
    """A tool that exceeds TOOL_CALL_TIMEOUT_S surfaces tool_error."""
    import src.api.services.chat_service as chat_service_mod

    # Shrink the timeout to keep the test quick.
    monkeypatch.setattr(chat_service_mod, "TOOL_CALL_TIMEOUT_S", 0.2)

    svc = chat_service

    @svc._tool_registry.register(
        name="slow_tool",
        description="Sleeps longer than the configured tool timeout.",
        category=ToolCategory.ANALYSIS,
    )
    def slow_tool() -> dict[str, Any]:
        time.sleep(1.0)
        return {"ok": True}

    ctx = svc.get_or_create_session(None)
    svc.add_user_message(ctx.session_id, "Run slow tool")

    svc._adapter.stream_response.return_value = iter(
        [
            AgentChunk(
                content="",
                tool_call_delta={
                    "id": "tc_slow",
                    "name": "slow_tool",
                    "arguments": "{}",
                },
                is_final=True,
            )
        ]
    )

    async def _run() -> list[str]:
        out: list[str] = []
        async for chunk in svc.stream_response(ctx.session_id):
            out.append(chunk)
        return out

    out = asyncio.get_event_loop().run_until_complete(_run())
    assert any("tool_error" in s and "slow_tool" in s for s in out)
    assert any("timed out" in s for s in out)
