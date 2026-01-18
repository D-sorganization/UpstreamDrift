"""Unit tests for AI module types."""

from __future__ import annotations

import sys
from datetime import timezone

# Python 3.10 compatibility: UTC was added in 3.11
from datetime import UTC

from shared.python.ai.types import (
    AgentChunk,
    AgentResponse,
    ConversationContext,
    ExpertiseLevel,
    Message,
    ProviderCapabilities,
    ProviderCapability,
    ToolCall,
    ToolResult,
)


class TestExpertiseLevel:
    """Tests for ExpertiseLevel enum."""

    def test_expertise_level_ordering(self) -> None:
        """Test that expertise levels are correctly ordered."""
        assert ExpertiseLevel.BEGINNER < ExpertiseLevel.INTERMEDIATE
        assert ExpertiseLevel.INTERMEDIATE < ExpertiseLevel.ADVANCED
        assert ExpertiseLevel.ADVANCED < ExpertiseLevel.EXPERT

    def test_expertise_level_values(self) -> None:
        """Test expertise level numeric values."""
        assert ExpertiseLevel.BEGINNER.value == 1
        assert ExpertiseLevel.INTERMEDIATE.value == 2
        assert ExpertiseLevel.ADVANCED.value == 3
        assert ExpertiseLevel.EXPERT.value == 4

    def test_expertise_level_comparison_with_invalid_type(self) -> None:
        """Test comparison with non-ExpertiseLevel type."""
        assert ExpertiseLevel.BEGINNER.__lt__("invalid") == NotImplemented
        assert ExpertiseLevel.BEGINNER.__le__("invalid") == NotImplemented


class TestProviderCapabilities:
    """Tests for ProviderCapabilities."""

    def test_has_capability_true(self) -> None:
        """Test has_capability returns True for supported capability."""
        caps = ProviderCapabilities(
            supported=frozenset({ProviderCapability.STREAMING}),
            max_tokens=8192,
            model_name="test-model",
        )
        assert caps.has_capability(ProviderCapability.STREAMING) is True

    def test_has_capability_false(self) -> None:
        """Test has_capability returns False for unsupported capability."""
        caps = ProviderCapabilities(
            supported=frozenset({ProviderCapability.STREAMING}),
            max_tokens=8192,
            model_name="test-model",
        )
        assert caps.has_capability(ProviderCapability.VISION) is False


class TestMessage:
    """Tests for Message dataclass."""

    def test_message_creation(self) -> None:
        """Test basic message creation."""
        msg = Message(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"
        assert msg.timestamp.tzinfo == UTC

    def test_message_to_dict(self) -> None:
        """Test message serialization."""
        msg = Message(role="user", content="Hello")
        result = msg.to_dict()
        assert result["role"] == "user"
        assert result["content"] == "Hello"
        assert "timestamp" in result

    def test_message_with_tool_calls(self) -> None:
        """Test message with tool calls."""
        tool_call = ToolCall.create("test_tool", {"arg": "value"})
        msg = Message(role="assistant", content="Calling tool", tool_calls=[tool_call])
        result = msg.to_dict()
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["name"] == "test_tool"


class TestToolCall:
    """Tests for ToolCall dataclass."""

    def test_tool_call_create(self) -> None:
        """Test ToolCall factory method."""
        tc = ToolCall.create("my_tool", {"param": 123})
        assert tc.name == "my_tool"
        assert tc.arguments == {"param": 123}
        assert tc.id.startswith("tc_")

    def test_tool_call_to_dict(self) -> None:
        """Test ToolCall serialization."""
        tc = ToolCall(id="tc_123", name="tool", arguments={"x": 1})
        result = tc.to_dict()
        assert result == {"id": "tc_123", "name": "tool", "arguments": {"x": 1}}


class TestToolResult:
    """Tests for ToolResult dataclass."""

    def test_tool_result_success(self) -> None:
        """Test successful tool result."""
        result = ToolResult(
            tool_call_id="tc_123",
            success=True,
            result={"data": "value"},
            execution_time=0.5,
        )
        assert result.success is True
        assert result.result == {"data": "value"}
        assert result.error is None

    def test_tool_result_failure(self) -> None:
        """Test failed tool result."""
        result = ToolResult(
            tool_call_id="tc_123",
            success=False,
            error="Something went wrong",
        )
        assert result.success is False
        assert result.error == "Something went wrong"


class TestConversationContext:
    """Tests for ConversationContext."""

    def test_context_creation(self) -> None:
        """Test basic context creation."""
        ctx = ConversationContext()
        assert ctx.session_id.startswith("session_")
        assert ctx.messages == []
        assert ctx.user_expertise == ExpertiseLevel.BEGINNER

    def test_add_user_message(self) -> None:
        """Test adding user message."""
        ctx = ConversationContext()
        msg = ctx.add_user_message("Hello")
        assert len(ctx.messages) == 1
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_add_assistant_message(self) -> None:
        """Test adding assistant message."""
        ctx = ConversationContext()
        msg = ctx.add_assistant_message("Hi there")
        assert len(ctx.messages) == 1
        assert msg.role == "assistant"

    def test_add_assistant_message_with_tool_calls(self) -> None:
        """Test adding assistant message with tool calls."""
        ctx = ConversationContext()
        tool_call = ToolCall.create("analyze", {"file": "test.c3d"})
        msg = ctx.add_assistant_message("Analyzing...", tool_calls=[tool_call])
        assert len(msg.tool_calls) == 1

    def test_add_tool_result(self) -> None:
        """Test adding tool result."""
        ctx = ConversationContext()
        msg = ctx.add_tool_result("tc_123", "Result: 42")
        assert msg.role == "tool"
        assert msg.tool_call_id == "tc_123"

    def test_get_recent_messages(self) -> None:
        """Test getting recent messages."""
        ctx = ConversationContext()
        for i in range(15):
            ctx.add_user_message(f"Message {i}")
        recent = ctx.get_recent_messages(5)
        assert len(recent) == 5
        assert recent[0].content == "Message 10"

    def test_clear_history(self) -> None:
        """Test clearing conversation history."""
        ctx = ConversationContext()
        ctx.add_user_message("Test")
        assert len(ctx.messages) == 1
        ctx.clear_history()
        assert len(ctx.messages) == 0

    def test_to_dict(self) -> None:
        """Test context serialization."""
        ctx = ConversationContext()
        ctx.add_user_message("Hello")
        result = ctx.to_dict()
        assert "session_id" in result
        assert len(result["messages"]) == 1
        assert result["user_expertise"] == "BEGINNER"


class TestAgentResponse:
    """Tests for AgentResponse."""

    def test_response_creation(self) -> None:
        """Test basic response creation."""
        response = AgentResponse(content="Hello")
        assert response.content == "Hello"
        assert response.tool_calls == []
        assert response.finish_reason == "stop"

    def test_has_tool_calls_true(self) -> None:
        """Test has_tool_calls with tool calls."""
        tc = ToolCall.create("tool", {})
        response = AgentResponse(content="", tool_calls=[tc])
        assert response.has_tool_calls is True

    def test_has_tool_calls_false(self) -> None:
        """Test has_tool_calls without tool calls."""
        response = AgentResponse(content="Hello")
        assert response.has_tool_calls is False

    def test_response_to_dict(self) -> None:
        """Test response serialization."""
        response = AgentResponse(
            content="Test",
            usage={"prompt_tokens": 10, "completion_tokens": 5},
        )
        result = response.to_dict()
        assert result["content"] == "Test"
        assert result["usage"]["prompt_tokens"] == 10


class TestAgentChunk:
    """Tests for AgentChunk."""

    def test_chunk_creation(self) -> None:
        """Test basic chunk creation."""
        chunk = AgentChunk(content="Hello", index=0)
        assert chunk.content == "Hello"
        assert chunk.is_final is False
        assert chunk.index == 0

    def test_final_chunk(self) -> None:
        """Test final chunk marker."""
        chunk = AgentChunk(content="", is_final=True, index=5)
        assert chunk.is_final is True
