from collections.abc import Iterator

import pytest

from src.api.services.chat_service import ChatService
from src.shared.python.ai.adapters.base import BaseAgentAdapter, ToolDeclaration
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
                            "function": {
                                "name": "explain_concept",
                                "arguments": '{"term": "inverse dynamics"}',
                            },
                        }
                    ]
                },
                is_final=True,
            )
        else:
            yield AgentChunk(content="The concept has been explained.", is_final=True)

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
async def test_chat_service_tools():
    service = ChatService()

    mock_adapter = MockToolAdapter()
    service._adapter = mock_adapter

    session = service.get_or_create_session(None)
    service.add_user_message(session.session_id, "Explain inverse dynamics")

    results = []
    async for item in service.stream_response(session.session_id):
        results.append(item)

    assert mock_adapter.call_count == 2

    assert any(
        isinstance(r, dict)
        and r.get("type") == "tool_call_started"
        and r.get("tool") == "explain_concept"
        for r in results
    )
    assert any(
        isinstance(r, dict)
        and r.get("type") == "tool_call_result"
        and r.get("tool") == "explain_concept"
        for r in results
    )
    assert "The concept has been explained." in results
