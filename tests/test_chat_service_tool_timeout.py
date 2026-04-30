import time
from collections.abc import Iterator

import pytest

from src.api.services.chat_service import ChatService
from src.shared.python.ai.adapters.base import BaseAgentAdapter, ToolDeclaration
from src.shared.python.ai.tool_registry import ToolCategory
from src.shared.python.ai.types import (
    AgentChunk,
    AgentResponse,
    ConversationContext,
    ProviderCapabilities,
    ProviderCapability,
)


class MockToolAdapter(BaseAgentAdapter):
    def __init__(self):
        self.call_count = 0

    def send_message(
        self, message: str, context: ConversationContext, tools: list[ToolDeclaration]
    ) -> AgentResponse:
        return AgentResponse(content="mock")

    def stream_response(
        self, message: str, context: ConversationContext, tools: list[ToolDeclaration]
    ) -> Iterator[AgentChunk]:
        self.call_count += 1
        if self.call_count == 1:
            yield AgentChunk(
                tool_call_delta={
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "tc_123",
                            "function": {"name": "slow_tool", "arguments": "{}"},
                        }
                    ]
                },
                is_final=True,
            )
        else:
            yield AgentChunk(content="The slow tool finished.", is_final=True)

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            supported=frozenset([ProviderCapability.FUNCTION_CALLING]),
            max_tokens=1000,
            model_name="mock",
        )

    def validate_connection(self) -> tuple[bool, str]:
        return True, "Mock connected"


@pytest.mark.asyncio
async def test_chat_service_tool_timeout():
    service = ChatService()

    # Register a slow tool
    @service._tool_registry.register(
        name="slow_tool",
        description="A tool that takes too long",
        category=ToolCategory.ANALYSIS,
    )
    def slow_tool() -> dict:
        time.sleep(2.0)
        return {"status": "done"}

    # Patch get_tool_timeout to a small value
    import src.shared.python.ai.config

    original_get_tool_timeout = src.shared.python.ai.config.get_tool_timeout
    src.shared.python.ai.config.get_tool_timeout = lambda: 0.1

    try:
        mock_adapter = MockToolAdapter()
        service._adapter = mock_adapter

        session = service.get_or_create_session(None)
        service.add_user_message(session.session_id, "Use the slow tool")

        results = []
        async for item in service.stream_response(session.session_id):
            results.append(item)

        assert any(
            isinstance(r, dict)
            and r.get("type") == "tool_call_started"
            and r.get("tool") == "slow_tool"
            for r in results
        )
        assert any(
            isinstance(r, dict)
            and r.get("type") == "tool_error"
            and r.get("tool") == "slow_tool"
            for r in results
        )

        error_chunk = next(
            r for r in results if isinstance(r, dict) and r.get("type") == "tool_error"
        )
        assert "timed out" in error_chunk.get("detail", "")
    finally:
        src.shared.python.ai.config.get_tool_timeout = original_get_tool_timeout
