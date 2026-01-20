"""OpenAI adapter for GPT-4 and GPT-4 Turbo models.

This adapter provides integration with OpenAI's chat completion API,
including full support for function calling and streaming.

Requirements:
    - OpenAI API key (user-provided)
    - openai package: pip install openai

Cost Model:
    - GPT-4 Turbo: ~$0.01/1K input tokens, ~$0.03/1K output tokens
    - Typical workflow: ~$0.50-1.00

Example:
    >>> from shared.python.ai.adapters.openai_adapter import OpenAIAdapter
    >>> adapter = OpenAIAdapter(api_key="sk-...")
    >>> response = adapter.send_message("Analyze this swing", context, tools)
"""

from __future__ import annotations

import json
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

# OpenAI configuration defaults
OPENAI_DEFAULT_MODEL = "gpt-4-turbo-preview"
OPENAI_DEFAULT_TIMEOUT = 60.0  # [s]
OPENAI_MAX_TOKENS = 128000  # GPT-4 Turbo context window


class OpenAIAdapter(BaseAgentAdapter):
    """Adapter for OpenAI GPT-4 models.

    Provides full integration with OpenAI's chat completion API:
    - Function/tool calling
    - Streaming responses
    - JSON mode
    - Long context (128K tokens)

    Attributes:
        api_key: OpenAI API key.
        model: Model name to use.
        timeout: Request timeout [s].
        organization: Optional organization ID.

    Example:
        >>> adapter = OpenAIAdapter(api_key="sk-...")
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
        model: str = OPENAI_DEFAULT_MODEL,
        timeout: float = OPENAI_DEFAULT_TIMEOUT,
        organization: str | None = None,
    ) -> None:
        """Initialize OpenAI adapter.

        Args:
            api_key: OpenAI API key.
            model: Model name. Default: gpt-4-turbo-preview
            timeout: Request timeout [s]. Default: 60.0
            organization: Optional OpenAI organization ID.
        """
        self._api_key = api_key
        self._model = model
        self._timeout = timeout
        self._organization = organization
        self._client: Any = None  # Lazy-loaded OpenAI client

        logger.info("Initialized OpenAIAdapter: model=%s", self._model)

    def _get_client(self) -> Any:
        """Get or create OpenAI client.

        Lazy-loads the openai package to avoid import errors.

        Returns:
            OpenAI client instance.

        Raises:
            AIProviderError: If openai package not installed.
        """
        if self._client is None:
            try:
                from openai import OpenAI

                self._client = OpenAI(
                    api_key=self._api_key,
                    organization=self._organization,
                    timeout=self._timeout,
                )
            except ImportError as e:
                raise AIProviderError(
                    "openai package required for OpenAIAdapter. "
                    "Install with: pip install openai",
                    provider="openai",
                ) from e
        return self._client

    def send_message(
        self,
        message: str,
        context: ConversationContext,
        tools: list[ToolDeclaration],
    ) -> AgentResponse:
        """Send a message to OpenAI.

        Args:
            message: User message to process.
            context: Current conversation context.
            tools: Available tools for this request.

        Returns:
            AgentResponse with model's reply.

        Raises:
            AIProviderError: For OpenAI API errors.
            AIRateLimitError: If rate limit exceeded.
            AITimeoutError: If request times out.
        """
        client = self._get_client()

        # Format messages
        messages = self._format_messages(context, message)

        # Format tools
        openai_tools = [t.to_openai_format() for t in tools] if tools else None

        try:
            response = client.chat.completions.create(
                model=self._model,
                messages=messages,
                tools=openai_tools,
                temperature=0.7,
                max_tokens=4096,
            )

            return self._parse_response(response)

        except Exception as e:
            return self._handle_error(e)

    def stream_response(
        self,
        message: str,
        context: ConversationContext,
        tools: list[ToolDeclaration],
    ) -> Iterator[AgentChunk]:
        """Stream response from OpenAI.

        Args:
            message: User message to process.
            context: Current conversation context.
            tools: Available tools.

        Yields:
            AgentChunk instances as they arrive.
        """
        client = self._get_client()
        messages = self._format_messages(context, message)
        openai_tools = [t.to_openai_format() for t in tools] if tools else None

        try:
            stream = client.chat.completions.create(
                model=self._model,
                messages=messages,
                tools=openai_tools,
                temperature=0.7,
                stream=True,
            )

            index = 0
            for chunk in stream:
                delta = chunk.choices[0].delta if chunk.choices else None

                if delta:
                    content = delta.content or ""
                    is_final = chunk.choices[0].finish_reason is not None

                    # Handle tool call deltas
                    tool_delta = None
                    if delta.tool_calls:
                        tool_delta = {
                            "tool_calls": [
                                {
                                    "index": tc.index,
                                    "id": tc.id,
                                    "function": {
                                        "name": (
                                            tc.function.name if tc.function else None
                                        ),
                                        "arguments": (
                                            tc.function.arguments
                                            if tc.function
                                            else None
                                        ),
                                    },
                                }
                                for tc in delta.tool_calls
                            ]
                        }

                    yield AgentChunk(
                        content=content,
                        tool_call_delta=tool_delta,
                        is_final=is_final,
                        index=index,
                    )
                    index += 1

        except Exception as e:
            logger.error("OpenAI streaming error: %s", e)
            raise AIProviderError(
                f"OpenAI streaming error: {e}",
                provider="openai",
            ) from e

    @property
    def capabilities(self) -> ProviderCapabilities:
        """Return OpenAI capabilities.

        Returns:
            ProviderCapabilities for OpenAI.
        """
        return ProviderCapabilities(
            supported=frozenset(
                {
                    ProviderCapability.FUNCTION_CALLING,
                    ProviderCapability.STREAMING,
                    ProviderCapability.VISION,
                    ProviderCapability.JSON_MODE,
                    ProviderCapability.LONG_CONTEXT,
                    ProviderCapability.SYSTEM_MESSAGE,
                }
            ),
            max_tokens=OPENAI_MAX_TOKENS,
            model_name=self._model,
            provider_name="openai",
        )

    def validate_connection(self) -> tuple[bool, str]:
        """Test connection to OpenAI.

        Returns:
            Tuple of (success, diagnostic_message).
        """
        try:
            client = self._get_client()

            # Simple model list call to verify API key
            models = client.models.list()

            # Check if our model is available
            model_ids = [m.id for m in models.data]

            if self._model in model_ids or any(self._model in m for m in model_ids):
                return True, f"Connected to OpenAI with {self._model}"

            return True, (
                f"Connected to OpenAI. Note: {self._model} not in "
                f"visible models, but may still work."
            )

        except AIProviderError:
            return False, (
                "openai package not installed. Install with: pip install openai"
            )
        except Exception as e:
            error_str = str(e).lower()
            if "authentication" in error_str or "api key" in error_str:
                return False, "Invalid API key. Check your OpenAI API key."
            if "rate limit" in error_str:
                return False, "Rate limited. Try again later."
            return False, f"Connection error: {e}"

    def _format_messages(
        self,
        context: ConversationContext,
        current_message: str,
    ) -> list[dict[str, Any]]:
        """Format messages for OpenAI API.

        Args:
            context: Conversation context.
            current_message: Current user message.

        Returns:
            List of message dicts for OpenAI.
        """
        messages: list[dict[str, Any]] = []

        # Add system message
        messages.append(
            {
                "role": "system",
                "content": self._build_system_message(context),
            }
        )

        # Add conversation history
        for msg in context.messages:
            formatted: dict[str, Any] = {
                "role": msg.role,
                "content": msg.content,
            }

            # Handle tool results
            if msg.role == "tool" and msg.tool_call_id:
                formatted["tool_call_id"] = msg.tool_call_id

            # Handle assistant tool calls
            if msg.tool_calls:
                formatted["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),
                        },
                    }
                    for tc in msg.tool_calls
                ]

            messages.append(formatted)

        # Add current message
        messages.append(
            {
                "role": "user",
                "content": current_message,
            }
        )

        return messages

    def _build_system_message(self, context: ConversationContext) -> str:
        """Build OpenAI-optimized system message.

        Args:
            context: Current conversation context.

        Returns:
            System message string.
        """
        expertise = context.user_expertise.name.lower()

        return (
            f"You are an AI assistant for the Golf Modeling Suite, a research-grade "
            f"biomechanics simulation platform for analyzing golf swings.\n\n"
            f"Current user expertise level: {expertise}\n\n"
            f"Your capabilities include:\n"
            f"- Analyzing C3D motion capture data\n"
            f"- Running physics simulations (MuJoCo, Drake, Pinocchio)\n"
            f"- Computing inverse dynamics and joint torques\n"
            f"- Performing drift-control decomposition\n"
            f"- Generating visualizations and reports\n\n"
            f"Guidelines:\n"
            f"1. Use tools to perform analyses - never make up numerical results\n"
            f"2. Explain concepts at the {expertise} level\n"
            f"3. Validate scientific claims before presenting them\n"
            f"4. Guide users through workflows step by step\n"
            f"5. Acknowledge uncertainty and cite limitations\n\n"
            f"When the user asks about analysis:\n"
            f"1. First, understand what data they have\n"
            f"2. Suggest appropriate analyses for their goals\n"
            f"3. Execute using available tools\n"
            f"4. Interpret results with scientific rigor"
        )

    def _parse_response(self, response: Any) -> AgentResponse:
        """Parse OpenAI response into AgentResponse.

        Args:
            response: Raw OpenAI response.

        Returns:
            Parsed AgentResponse.
        """
        choice = response.choices[0]
        message = choice.message

        # Extract content
        content = message.content or ""

        # Parse tool calls
        tool_calls: list[ToolCall] = []
        if message.tool_calls:
            for tc in message.tool_calls:
                try:
                    arguments = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    arguments = {"raw": tc.function.arguments}

                tool_calls.append(
                    ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=arguments,
                    )
                )

        # Extract usage
        usage: dict[str, int] = {}
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        return AgentResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason or "stop",
            usage=usage,
            metadata={
                "model": response.model,
                "id": response.id,
            },
        )

    def _handle_error(self, error: Exception) -> AgentResponse:
        """Handle OpenAI API errors.

        Args:
            error: The exception that occurred.

        Raises:
            Appropriate AIError subclass.
        """
        error_str = str(error).lower()

        # Rate limit
        if "rate limit" in error_str or "429" in error_str:
            raise AIRateLimitError(
                "OpenAI rate limit exceeded. Please wait and retry.",
                provider="openai",
            ) from error

        # Timeout
        if "timeout" in error_str:
            raise AITimeoutError(
                f"OpenAI request timed out after {self._timeout}s",
                provider="openai",
                timeout=self._timeout,
            ) from error

        # Connection
        if "connection" in error_str or "network" in error_str:
            raise AIConnectionError(
                "Cannot connect to OpenAI. Check your network.",
                provider="openai",
            ) from error

        # Generic
        raise AIProviderError(
            f"OpenAI error: {error}",
            provider="openai",
        ) from error
