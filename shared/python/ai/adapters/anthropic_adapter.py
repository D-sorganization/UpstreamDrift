"""Anthropic adapter for Claude 3.x models.

This adapter provides integration with Anthropic's Claude API,
including tool use and streaming support.

Requirements:
    - Anthropic API key (user-provided)
    - anthropic package: pip install anthropic

Cost Model:
    - Claude 3 Opus: ~$15/million input, ~$75/million output
    - Claude 3 Sonnet: ~$3/million input, ~$15/million output
    - Typical workflow: ~$0.30-0.70

Example:
    >>> from shared.python.ai.adapters.anthropic_adapter import AnthropicAdapter
    >>> adapter = AnthropicAdapter(api_key="sk-ant-...")
    >>> response = adapter.send_message("Analyze this swing", context, tools)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator

from shared.python.ai.adapters.base import BaseAgentAdapter, ToolDeclaration
from shared.python.ai.exceptions import (
    AIConnectionError,
    AIProviderError,
    AIRateLimitError,
    AITimeoutError,
)
from shared.python.ai.types import (
    AgentChunk,
    AgentResponse,
    ConversationContext,
    ProviderCapabilities,
    ProviderCapability,
    ToolCall,
)

logger = logging.getLogger(__name__)

# Anthropic configuration defaults
ANTHROPIC_DEFAULT_MODEL = "claude-3-sonnet-20240229"
ANTHROPIC_DEFAULT_TIMEOUT = 60.0  # [s]
ANTHROPIC_MAX_TOKENS = 200000  # Claude 3 context window


class AnthropicAdapter(BaseAgentAdapter):
    """Adapter for Anthropic Claude models.

    Provides integration with Anthropic's Claude API:
    - Tool use
    - Streaming responses
    - Long context (200K tokens)

    Attributes:
        api_key: Anthropic API key.
        model: Model name to use.
        timeout: Request timeout [s].

    Example:
        >>> adapter = AnthropicAdapter(api_key="sk-ant-...")
        >>> success, message = adapter.validate_connection()
        >>> if success:
        ...     response = adapter.send_message(
        ...         "Analyze joint torques",
        ...         context,
        ...         tools
        ...     )
    """

    def __init__(
        self,
        api_key: str,
        model: str = ANTHROPIC_DEFAULT_MODEL,
        timeout: float = ANTHROPIC_DEFAULT_TIMEOUT,
    ) -> None:
        """Initialize Anthropic adapter.

        Args:
            api_key: Anthropic API key.
            model: Model name. Default: claude-3-sonnet-20240229
            timeout: Request timeout [s]. Default: 60.0
        """
        self._api_key = api_key
        self._model = model
        self._timeout = timeout
        self._client: Any = None  # Lazy-loaded Anthropic client

        logger.info("Initialized AnthropicAdapter: model=%s", self._model)

    def _get_client(self) -> Any:
        """Get or create Anthropic client.

        Returns:
            Anthropic client instance.

        Raises:
            AIProviderError: If anthropic package not installed.
        """
        if self._client is None:
            try:
                from anthropic import Anthropic

                self._client = Anthropic(
                    api_key=self._api_key,
                    timeout=self._timeout,
                )
            except ImportError as e:
                raise AIProviderError(
                    "anthropic package required for AnthropicAdapter. "
                    "Install with: pip install anthropic",
                    provider="anthropic",
                ) from e
        return self._client

    def send_message(
        self,
        message: str,
        context: ConversationContext,
        tools: list[ToolDeclaration],
    ) -> AgentResponse:
        """Send a message to Anthropic Claude.

        Args:
            message: User message to process.
            context: Current conversation context.
            tools: Available tools for this request.

        Returns:
            AgentResponse with model's reply.
        """
        client = self._get_client()

        # Format messages
        messages = self._format_messages(context, message)
        system = self._build_system_message(context)

        # Format tools
        anthropic_tools = [t.to_anthropic_format() for t in tools] if tools else None

        try:
            kwargs: dict[str, Any] = {
                "model": self._model,
                "max_tokens": 4096,
                "system": system,
                "messages": messages,
            }
            if anthropic_tools:
                kwargs["tools"] = anthropic_tools

            response = client.messages.create(**kwargs)
            return self._parse_response(response)

        except Exception as e:
            return self._handle_error(e)

    def stream_response(
        self,
        message: str,
        context: ConversationContext,
        tools: list[ToolDeclaration],
    ) -> Iterator[AgentChunk]:
        """Stream response from Anthropic.

        Args:
            message: User message to process.
            context: Current conversation context.
            tools: Available tools.

        Yields:
            AgentChunk instances as they arrive.
        """
        client = self._get_client()
        messages = self._format_messages(context, message)
        system = self._build_system_message(context)
        anthropic_tools = [t.to_anthropic_format() for t in tools] if tools else None

        try:
            kwargs: dict[str, Any] = {
                "model": self._model,
                "max_tokens": 4096,
                "system": system,
                "messages": messages,
            }
            if anthropic_tools:
                kwargs["tools"] = anthropic_tools

            with client.messages.stream(**kwargs) as stream:
                index = 0
                for event in stream:
                    if hasattr(event, "type"):
                        if event.type == "content_block_delta":
                            delta = event.delta
                            if hasattr(delta, "text"):
                                yield AgentChunk(
                                    content=delta.text,
                                    is_final=False,
                                    index=index,
                                )
                                index += 1
                        elif event.type == "message_stop":
                            yield AgentChunk(
                                content="",
                                is_final=True,
                                index=index,
                            )

        except Exception as e:
            logger.error("Anthropic streaming error: %s", e)
            raise AIProviderError(
                f"Anthropic streaming error: {e}",
                provider="anthropic",
            ) from e

    @property
    def capabilities(self) -> ProviderCapabilities:
        """Return Anthropic capabilities.

        Returns:
            ProviderCapabilities for Anthropic.
        """
        return ProviderCapabilities(
            supported=frozenset(
                {
                    ProviderCapability.FUNCTION_CALLING,
                    ProviderCapability.STREAMING,
                    ProviderCapability.VISION,
                    ProviderCapability.LONG_CONTEXT,
                    ProviderCapability.SYSTEM_MESSAGE,
                }
            ),
            max_tokens=ANTHROPIC_MAX_TOKENS,
            model_name=self._model,
            provider_name="anthropic",
        )

    def validate_connection(self) -> tuple[bool, str]:
        """Test connection to Anthropic.

        Returns:
            Tuple of (success, diagnostic_message).
        """
        try:
            client = self._get_client()

            # Simple test message
            response = client.messages.create(
                model=self._model,
                max_tokens=10,
                messages=[{"role": "user", "content": "Hi"}],
            )

            if response.content:
                return True, f"Connected to Anthropic with {self._model}"

            return True, "Connected to Anthropic"

        except AIProviderError:
            return False, (
                "anthropic package not installed. "
                "Install with: pip install anthropic"
            )
        except Exception as e:
            error_str = str(e).lower()
            if "authentication" in error_str or "api key" in error_str:
                return False, "Invalid API key. Check your Anthropic API key."
            if "rate limit" in error_str:
                return False, "Rate limited. Try again later."
            return False, f"Connection error: {e}"

    def _format_messages(
        self,
        context: ConversationContext,
        current_message: str,
    ) -> list[dict[str, Any]]:
        """Format messages for Anthropic API.

        Anthropic requires alternating user/assistant messages.

        Args:
            context: Conversation context.
            current_message: Current user message.

        Returns:
            List of message dicts for Anthropic.
        """
        messages: list[dict[str, Any]] = []

        # Process conversation history
        for msg in context.messages:
            content: Any = msg.content

            # Handle tool results
            if msg.role == "tool":
                # In Anthropic, tool results are part of user messages
                content = [
                    {
                        "type": "tool_result",
                        "tool_use_id": msg.tool_call_id,
                        "content": msg.content,
                    }
                ]
                role = "user"
            else:
                role = msg.role

            # Handle assistant tool calls
            if msg.tool_calls:
                content = []
                if msg.content:
                    content.append({"type": "text", "text": msg.content})
                for tc in msg.tool_calls:
                    content.append(
                        {
                            "type": "tool_use",
                            "id": tc.id,
                            "name": tc.name,
                            "input": tc.arguments,
                        }
                    )

            messages.append(
                {
                    "role": role,
                    "content": content,
                }
            )

        # Add current message
        messages.append(
            {
                "role": "user",
                "content": current_message,
            }
        )

        # Ensure alternating roles (Anthropic requirement)
        messages = self._ensure_alternating_roles(messages)

        return messages

    def _ensure_alternating_roles(
        self,
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Ensure messages alternate between user and assistant.

        Anthropic requires strictly alternating roles.

        Args:
            messages: List of messages.

        Returns:
            Messages with alternating roles.
        """
        if not messages:
            return messages

        result: list[dict[str, Any]] = []

        for msg in messages:
            if not result:
                result.append(msg)
                continue

            last_role = result[-1]["role"]
            current_role = msg["role"]

            # If same role, merge content
            if last_role == current_role:
                last_content = result[-1]["content"]
                current_content = msg["content"]

                # Handle string content
                if isinstance(last_content, str) and isinstance(current_content, str):
                    result[-1]["content"] = f"{last_content}\n\n{current_content}"
                else:
                    # Handle list content
                    if isinstance(last_content, str):
                        last_content = [{"type": "text", "text": last_content}]
                    if isinstance(current_content, str):
                        current_content = [{"type": "text", "text": current_content}]
                    result[-1]["content"] = last_content + current_content
            else:
                result.append(msg)

        return result

    def _build_system_message(self, context: ConversationContext) -> str:
        """Build Anthropic-optimized system message.

        Args:
            context: Current conversation context.

        Returns:
            System message string.
        """
        expertise = context.user_expertise.name.lower()

        return (
            f"You are Claude, an AI assistant for the Golf Modeling Suite, a "
            f"research-grade biomechanics simulation platform for analyzing golf swings.\n\n"
            f"Current user expertise level: {expertise}\n\n"
            f"Your capabilities include:\n"
            f"- Analyzing C3D motion capture data\n"
            f"- Running physics simulations (MuJoCo, Drake, Pinocchio)\n"
            f"- Computing inverse dynamics and joint torques\n"
            f"- Performing drift-control decomposition\n"
            f"- Generating visualizations and reports\n\n"
            f"Guidelines:\n"
            f"1. Use tools to perform analyses - never fabricate numerical results\n"
            f"2. Explain concepts at the {expertise} level\n"
            f"3. Validate scientific claims before presenting them\n"
            f"4. Guide users through workflows step by step\n"
            f"5. Acknowledge uncertainty and cite limitations\n"
            f"6. Be precise about physical units (SI: m, kg, s, rad, N, N·m)"
        )

    def _parse_response(self, response: Any) -> AgentResponse:
        """Parse Anthropic response into AgentResponse.

        Args:
            response: Raw Anthropic response.

        Returns:
            Parsed AgentResponse.
        """
        # Extract content blocks
        content_parts: list[str] = []
        tool_calls: list[ToolCall] = []

        for block in response.content:
            if block.type == "text":
                content_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=block.id,
                        name=block.name,
                        arguments=block.input,
                    )
                )

        content = "\n".join(content_parts)

        # Extract usage
        usage: dict[str, int] = {}
        if hasattr(response, "usage"):
            usage = {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            }

        return AgentResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason=response.stop_reason or "end_turn",
            usage=usage,
            metadata={
                "model": response.model,
                "id": response.id,
            },
        )

    def _handle_error(self, error: Exception) -> AgentResponse:
        """Handle Anthropic API errors.

        Args:
            error: The exception that occurred.

        Raises:
            Appropriate AIError subclass.
        """
        error_str = str(error).lower()

        # Rate limit
        if "rate limit" in error_str or "429" in error_str:
            raise AIRateLimitError(
                "Anthropic rate limit exceeded. Please wait and retry.",
                provider="anthropic",
            ) from error

        # Timeout
        if "timeout" in error_str:
            raise AITimeoutError(
                f"Anthropic request timed out after {self._timeout}s",
                provider="anthropic",
                timeout=self._timeout,
            ) from error

        # Connection
        if "connection" in error_str or "network" in error_str:
            raise AIConnectionError(
                "Cannot connect to Anthropic. Check your network.",
                provider="anthropic",
            ) from error

        # Generic
        raise AIProviderError(
            f"Anthropic error: {error}",
            provider="anthropic",
        ) from error
