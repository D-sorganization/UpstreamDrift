"""Core types for the AI Assistant integration layer.

This module defines the fundamental data structures used throughout
the AI integration layer, including messages, contexts, and capabilities.

All types use dataclasses for immutability and clear structure.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

UTC = timezone.utc
from enum import Enum, auto
from typing import Any


class ExpertiseLevel(Enum):
    """User expertise levels for progressive disclosure.

    The AI assistant adjusts its explanations and feature exposure
    based on the user's current expertise level.

    Levels:
        BEGINNER: No prior knowledge assumed. Full explanations.
        INTERMEDIATE: Basic physics/biomechanics understanding.
        ADVANCED: Graduate-level comprehension. Minimal explanation.
        EXPERT: Research-publication ready. Technical details only.
    """

    BEGINNER = 1
    INTERMEDIATE = 2
    ADVANCED = 3
    EXPERT = 4

    def __lt__(self, other: object) -> bool:
        """Support comparison for level filtering."""
        if isinstance(other, ExpertiseLevel):
            return self.value < other.value
        return NotImplemented

    def __le__(self, other: object) -> bool:
        """Support comparison for level filtering."""
        if isinstance(other, ExpertiseLevel):
            return self.value <= other.value
        return NotImplemented


class ProviderCapability(Enum):
    """Capabilities that may vary by AI provider.

    Used to negotiate features between the AIP and provider adapters.
    """

    FUNCTION_CALLING = auto()  # Tool/function invocation support
    STREAMING = auto()  # Incremental response streaming
    VISION = auto()  # Image understanding
    CODE_EXECUTION = auto()  # Code execution sandbox
    LONG_CONTEXT = auto()  # >32k token context window
    JSON_MODE = auto()  # Structured JSON output
    SYSTEM_MESSAGE = auto()  # System message support


@dataclass(frozen=True)
class ProviderCapabilities:
    """Describes an AI provider's capabilities.

    Used by the AIP to determine which features are available
    for the current provider.

    Attributes:
        supported: Set of supported capabilities.
        max_tokens: Maximum context window [tokens].
        model_name: Specific model identifier.
        provider_name: Name of the provider (e.g., 'openai').
    """

    supported: frozenset[ProviderCapability]
    max_tokens: int
    model_name: str
    provider_name: str = "unknown"

    def has_capability(self, capability: ProviderCapability) -> bool:
        """Check if provider has a specific capability.

        Args:
            capability: Capability to check.

        Returns:
            True if capability is supported.
        """
        return capability in self.supported


@dataclass
class Message:
    """A single message in a conversation.

    Attributes:
        role: Message role ('user', 'assistant', 'system', 'tool').
        content: The message text content.
        timestamp: When the message was created [UTC].
        tool_calls: Tool calls made by the assistant (if any).
        tool_call_id: ID of the tool call this message responds to.
        metadata: Additional message metadata.
    """

    role: str
    content: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation of the message.
        """
        result: dict[str, Any] = {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
        }
        if self.tool_calls:
            result["tool_calls"] = [tc.to_dict() for tc in self.tool_calls]
        if self.tool_call_id:
            result["tool_call_id"] = self.tool_call_id
        if self.metadata:
            result["metadata"] = self.metadata
        return result


@dataclass
class ToolCall:
    """A request to execute a tool.

    Attributes:
        id: Unique identifier for this tool call.
        name: Name of the tool to execute.
        arguments: Arguments to pass to the tool.
    """

    id: str
    name: str
    arguments: dict[str, Any]

    @classmethod
    def create(cls, name: str, arguments: dict[str, Any]) -> ToolCall:
        """Create a new tool call with generated ID.

        Args:
            name: Name of the tool.
            arguments: Tool arguments.

        Returns:
            New ToolCall instance.
        """
        return cls(
            id=f"tc_{uuid.uuid4().hex[:12]}",
            name=name,
            arguments=arguments,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation of the tool call.
        """
        return {
            "id": self.id,
            "name": self.name,
            "arguments": self.arguments,
        }


@dataclass
class ToolResult:
    """Result of a tool execution.

    Attributes:
        tool_call_id: ID of the tool call this result responds to.
        success: Whether the tool executed successfully.
        result: The result value (if successful).
        error: Error message (if failed).
        execution_time: How long the tool took to execute [s].
    """

    tool_call_id: str
    success: bool
    result: Any = None
    error: str | None = None
    execution_time: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation of the result.
        """
        return {
            "tool_call_id": self.tool_call_id,
            "success": self.success,
            "result": self.result,
            "error": self.error,
            "execution_time": self.execution_time,
        }


@dataclass
class ConversationContext:
    """Manages conversation state for multi-turn interactions.

    Attributes:
        session_id: Unique identifier for this session.
        messages: List of previous messages in conversation.
        user_expertise: User's current expertise level.
        active_workflow_id: Currently executing workflow (if any).
        metadata: Additional session metadata.
    """

    session_id: str = field(default_factory=lambda: f"session_{uuid.uuid4().hex[:12]}")
    messages: list[Message] = field(default_factory=list)
    user_expertise: ExpertiseLevel = ExpertiseLevel.BEGINNER
    active_workflow_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    # Token tracking for context management
    _estimated_tokens: int = field(default=0, repr=False)
    _max_tokens: int = field(default=8192, repr=False)

    def add_message(self, role: str, content: str, **kwargs: Any) -> Message:
        """Add a message to the conversation history.

        Args:
            role: Message role ('user', 'assistant', 'system', 'tool').
            content: The message content.
            **kwargs: Additional message attributes.

        Returns:
            The created Message instance.
        """
        message = Message(role=role, content=content, **kwargs)
        self.messages.append(message)
        self._update_token_estimate()
        return message

    def add_user_message(self, content: str) -> Message:
        """Add a user message to the conversation.

        Args:
            content: The user's message.

        Returns:
            The created Message instance.
        """
        return self.add_message("user", content)

    def add_assistant_message(
        self,
        content: str,
        tool_calls: list[ToolCall] | None = None,
    ) -> Message:
        """Add an assistant message to the conversation.

        Args:
            content: The assistant's response.
            tool_calls: Any tool calls made.

        Returns:
            The created Message instance.
        """
        return self.add_message(
            "assistant",
            content,
            tool_calls=tool_calls or [],
        )

    def add_tool_result(
        self,
        tool_call_id: str,
        content: str,
    ) -> Message:
        """Add a tool result to the conversation.

        Args:
            tool_call_id: ID of the tool call this responds to.
            content: The tool result content.

        Returns:
            The created Message instance.
        """
        return self.add_message(
            "tool",
            content,
            tool_call_id=tool_call_id,
        )

    def get_recent_messages(self, count: int = 10) -> list[Message]:
        """Get the most recent messages.

        Args:
            count: Maximum number of messages to return.

        Returns:
            List of recent messages.
        """
        return self.messages[-count:]

    def clear_history(self) -> None:
        """Clear conversation history."""
        self.messages.clear()
        self._estimated_tokens = 0

    def _update_token_estimate(self) -> None:
        """Update estimated token count and truncate if needed."""
        # Rough estimate: 4 characters per token
        chars_per_token = 4
        total_chars = sum(len(m.content) for m in self.messages)
        self._estimated_tokens = total_chars // chars_per_token

        # Truncate old messages if over limit
        while self._estimated_tokens > self._max_tokens and len(self.messages) > 2:
            removed = self.messages.pop(0)
            self._estimated_tokens -= len(removed.content) // chars_per_token

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation of the context.
        """
        return {
            "session_id": self.session_id,
            "messages": [m.to_dict() for m in self.messages],
            "user_expertise": self.user_expertise.name,
            "active_workflow_id": self.active_workflow_id,
            "metadata": self.metadata,
        }


@dataclass
class AgentResponse:
    """Response from an AI provider.

    Attributes:
        content: The text response content.
        tool_calls: Any tool calls requested.
        finish_reason: Why the response ended ('stop', 'tool_calls', etc.).
        usage: Token usage statistics.
        metadata: Additional response metadata.
    """

    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = "stop"
    usage: dict[str, int] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def has_tool_calls(self) -> bool:
        """Check if response includes tool calls.

        Returns:
            True if tool calls are present.
        """
        return len(self.tool_calls) > 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation of the response.
        """
        return {
            "content": self.content,
            "tool_calls": [tc.to_dict() for tc in self.tool_calls],
            "finish_reason": self.finish_reason,
            "usage": self.usage,
            "metadata": self.metadata,
        }


@dataclass
class AgentChunk:
    """A streaming chunk from an AI provider.

    Used for incremental response delivery during streaming.

    Attributes:
        content: The text content in this chunk.
        tool_call_delta: Partial tool call data (if any).
        is_final: Whether this is the final chunk.
        index: Chunk index in the stream.
    """

    content: str = ""
    tool_call_delta: dict[str, Any] | None = None
    is_final: bool = False
    index: int = 0
