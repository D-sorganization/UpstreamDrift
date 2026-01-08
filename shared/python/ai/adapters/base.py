"""Base adapter protocol for AI providers.

This module defines the abstract interface that all AI provider adapters
must implement, ensuring consistent behavior across providers.

The protocol pattern allows for easy addition of new providers without
modifying existing code.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator

from shared.python.ai.types import (
    AgentChunk,
    AgentResponse,
    ConversationContext,
    ProviderCapabilities,
)

logger = logging.getLogger(__name__)


@dataclass
class ToolDeclaration:
    """Declaration of a tool available to the AI.

    This is a simplified version for adapter communication.
    The full ToolDeclaration with validation is in tool_registry.py.

    Attributes:
        name: Unique tool identifier.
        description: What the tool does (AI-consumable).
        parameters: JSON Schema for tool parameters.
        required: List of required parameter names.
    """

    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)
    required: list[str] = field(default_factory=list)

    def to_openai_format(self) -> dict[str, Any]:
        """Convert to OpenAI function calling format.

        Returns:
            OpenAI-compatible function definition.
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": self.parameters,
                    "required": self.required,
                },
            },
        }

    def to_anthropic_format(self) -> dict[str, Any]:
        """Convert to Anthropic tool format.

        Returns:
            Anthropic-compatible tool definition.
        """
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": self.parameters,
                "required": self.required,
            },
        }


class BaseAgentAdapter(ABC):
    """Abstract base class for AI provider adapters.

    All provider-specific adapters inherit from this class and implement
    the required methods for communication with their respective APIs.

    The adapter is responsible for:
    1. Translating AIP messages to provider format
    2. Managing authentication and connections
    3. Handling provider-specific errors
    4. Implementing streaming where supported

    Example:
        >>> class MyAdapter(BaseAgentAdapter):
        ...     def send_message(self, message, context, tools):
        ...         # Translate and send to provider
        ...         ...
    """

    @abstractmethod
    def send_message(
        self,
        message: str,
        context: ConversationContext,
        tools: list[ToolDeclaration],
    ) -> AgentResponse:
        """Send a message to the AI provider.

        This is the primary method for synchronous communication with
        the AI provider. It handles the full request/response cycle.

        Args:
            message: User message to process.
            context: Current conversation context with history.
            tools: List of tools available for this request.

        Returns:
            Provider response translated to standard AgentResponse format.

        Raises:
            AIProviderError: If provider communication fails.
            AIConnectionError: If network connection fails.
            AIRateLimitError: If rate limit is exceeded.
            AITimeoutError: If request times out.
        """
        ...

    @abstractmethod
    def stream_response(
        self,
        message: str,
        context: ConversationContext,
        tools: list[ToolDeclaration],
    ) -> Iterator[AgentChunk]:
        """Stream response chunks from the AI provider.

        For providers that support streaming, this method yields
        response chunks as they arrive, enabling real-time UI updates.

        Args:
            message: User message to process.
            context: Current conversation context with history.
            tools: List of tools available for this request.

        Yields:
            Response chunks as they arrive from the provider.

        Raises:
            AIProviderError: If provider communication fails.
        """
        ...

    @property
    @abstractmethod
    def capabilities(self) -> ProviderCapabilities:
        """Return the provider's capabilities.

        This allows the AIP to adjust behavior based on what features
        the current provider supports.

        Returns:
            ProviderCapabilities describing supported features.
        """
        ...

    @abstractmethod
    def validate_connection(self) -> tuple[bool, str]:
        """Test connection to the AI provider.

        This method should perform a lightweight check to verify:
        1. Network connectivity
        2. Authentication validity
        3. Model availability

        Returns:
            Tuple of (success: bool, diagnostic_message: str).
        """
        ...

    def format_messages_for_provider(
        self,
        context: ConversationContext,
        current_message: str,
    ) -> list[dict[str, Any]]:
        """Format conversation history for the provider.

        This default implementation provides a basic format that works
        for most providers. Override for provider-specific formatting.

        Args:
            context: Conversation context with history.
            current_message: The current user message.

        Returns:
            List of message dictionaries for the provider.
        """
        messages: list[dict[str, Any]] = []

        # Add conversation history
        for msg in context.messages:
            formatted: dict[str, Any] = {
                "role": msg.role,
                "content": msg.content,
            }
            if msg.tool_call_id:
                formatted["tool_call_id"] = msg.tool_call_id
            messages.append(formatted)

        # Add current message
        messages.append(
            {
                "role": "user",
                "content": current_message,
            }
        )

        return messages

    def build_system_prompt(
        self,
        tools: list[ToolDeclaration],
        expertise_level: str = "beginner",
    ) -> str:
        """Build a system prompt including tool context.

        This default implementation provides a basic system prompt.
        Override for provider-specific or use-case-specific prompts.

        Args:
            tools: Available tools to describe.
            expertise_level: User's expertise level.

        Returns:
            System prompt string.
        """
        tool_descriptions = "\n".join(
            f"- {tool.name}: {tool.description}" for tool in tools
        )

        return (
            f"You are an AI assistant for the Golf Modeling Suite, a research-grade "
            f"biomechanics simulation platform.\n\n"
            f"Your role is to help users analyze golf swings using advanced physics "
            f"simulations across multiple engines (MuJoCo, Drake, Pinocchio).\n\n"
            f"User expertise level: {expertise_level}\n\n"
            f"Available tools:\n{tool_descriptions}\n\n"
            f"Guidelines:\n"
            f"1. Always validate scientific claims before presenting them\n"
            f"2. Explain concepts at the user's expertise level\n"
            f"3. Use tools to perform analyses rather than making up results\n"
            f"4. Cite sources and acknowledge uncertainty\n"
            f"5. Guide users through workflows step by step"
        )
