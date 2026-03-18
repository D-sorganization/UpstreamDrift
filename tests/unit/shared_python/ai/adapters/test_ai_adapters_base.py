"""Tests for the base AI adapter module."""

from src.shared.python.ai.adapters.base import BaseAgentAdapter, ToolDeclaration
from src.shared.python.ai.types import (
    AgentChunk,
    AgentResponse,
    ConversationContext,
    Message,
    ProviderCapabilities,
)


class DummyAdapter(BaseAgentAdapter):
    """A dummy adapter for testing the concrete methods of BaseAgentAdapter."""

    def send_message(
        self, message: str, context: ConversationContext, tools: list[ToolDeclaration]
    ) -> AgentResponse:
        return AgentResponse(content="test")

    def stream_response(
        self, message: str, context: ConversationContext, tools: list[ToolDeclaration]
    ):
        yield AgentChunk(content="test")

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities()

    def validate_connection(self) -> tuple[bool, str]:
        return True, "OK"


def test_tool_declaration_init():
    """Test ToolDeclaration initialization."""
    tool = ToolDeclaration(
        name="test_tool",
        description="A test tool",
        parameters={"prop1": "val1"},
        required=["prop1"],
    )
    assert tool.name == "test_tool"
    assert tool.description == "A test tool"
    assert tool.parameters == {"prop1": "val1"}
    assert tool.required == ["prop1"]

    # Test defaults
    tool2 = ToolDeclaration(name="test2", description="test2")
    assert tool2.parameters == {}
    assert tool2.required == []


def test_tool_declaration_openai_format():
    """Test converting ToolDeclaration to OpenAI format."""
    tool = ToolDeclaration(
        name="weather_tool",
        description="Get weather",
        parameters={"location": {"type": "string"}},
        required=["location"],
    )
    formatted = tool.to_openai_format()
    assert formatted == {
        "type": "function",
        "function": {
            "name": "weather_tool",
            "description": "Get weather",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"],
            },
        },
    }


def test_tool_declaration_anthropic_format():
    """Test converting ToolDeclaration to Anthropic format."""
    tool = ToolDeclaration(
        name="calc_tool",
        description="Calculator",
        parameters={"expr": {"type": "string"}},
        required=["expr"],
    )
    formatted = tool.to_anthropic_format()
    assert formatted == {
        "name": "calc_tool",
        "description": "Calculator",
        "input_schema": {
            "type": "object",
            "properties": {"expr": {"type": "string"}},
            "required": ["expr"],
        },
    }


def test_format_messages_for_provider():
    """Test formatting conversation history."""
    adapter = DummyAdapter()

    context = ConversationContext()
    context.messages = [
        Message(role="user", content="hello"),
        Message(role="assistant", content="hi there", tool_call_id="call_123"),
    ]

    formatted = adapter.format_messages_for_provider(context, "next message")

    assert len(formatted) == 3
    assert formatted[0] == {"role": "user", "content": "hello"}
    assert formatted[1] == {
        "role": "assistant",
        "content": "hi there",
        "tool_call_id": "call_123",
    }
    assert formatted[2] == {"role": "user", "content": "next message"}


def test_build_system_prompt():
    """Test building a basic system prompt."""
    adapter = DummyAdapter()

    tools = [
        ToolDeclaration(name="tool1", description="desc1"),
        ToolDeclaration(name="tool2", description="desc2"),
    ]

    prompt = adapter.build_system_prompt(tools, expertise_level="expert")

    assert "You are an AI assistant" in prompt
    assert "expertise level: expert" in prompt
    assert "- tool1: desc1" in prompt
    assert "- tool2: desc2" in prompt
    assert "Guidelines:" in prompt
