"""Base adapter protocol for AI providers.

This module defines the abstract interface that all AI provider adapters
must implement, ensuring consistent behavior across providers.

The protocol pattern allows for easy addition of new providers without
modifying existing code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from src.shared.python.logging_pkg.logging_config import get_logger

if TYPE_CHECKING:
    from collections.abc import Iterator

from src.shared.python.ai.exceptions import (
    AIConnectionError,
    AIProviderError,
    AIRateLimitError,
    AITimeoutError,
)
from src.shared.python.ai.memory_manager import (
    build_memory_prompt_section,
    load_agents_md,
)
from src.shared.python.ai.types import (
    AgentChunk,
    AgentResponse,
    ChatModelInfo,
    ConversationContext,
    ProviderCapabilities,
)

logger = get_logger(__name__)


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

    # ------------------------------------------------------------------ #
    # Provider catalogue surface for the shared chat dock dropdowns
    # (Tools issue #2871).
    #
    # Concrete adapters MUST override these and SHOULD do so without
    # performing a blocking network call when the local catalogue is
    # sufficient. The default implementations raise NotImplementedError
    # so a forgotten override is loud.
    # ------------------------------------------------------------------ #
    @abstractmethod
    def list_models(self) -> list[ChatModelInfo]:
        """Return a non-empty list of ChatModelInfo objects known to this adapter.

        Implementations may probe a live API but MUST fall back to a
        static catalogue when the API is unreachable so unit tests and
        offline environments can always populate the model dropdown.

        Returns:
            List of model identifiers; always non-empty.
        """
        ...

    @abstractmethod
    def thinking_capabilities(self) -> Any:
        """Return the :class:`ThinkingCapabilities` for the current model.

        Returns:
            ``ThinkingCapabilities`` describing supported reasoning
            levels. Adapters whose models do not support reasoning return
            a capability bundle with only the ``"none"`` level so the
            shared chat dropdown always has at least one entry.
        """
        ...

    # ------------------------------------------------------------------ #
    # Token-count normalization (issue #2763)                           #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _normalize_token_counts(raw_usage: dict[str, int]) -> dict[str, int]:
        """Normalize provider-specific token-count keys to a canonical set.

        Each provider uses different key names for the same concepts:
        - Anthropic: ``input_tokens``, ``output_tokens``
        - OpenAI / Ollama: ``prompt_tokens``, ``completion_tokens``, ``total_tokens``
        - Cline: already uses ``input_tokens`` / ``output_tokens``
        - BitNet / Rust: return ``{}``

        This method maps all variants to the canonical keys so callers never
        need to know which provider produced a response (issue #2763).

        Args:
            raw_usage: Raw usage dict from the provider.

        Returns:
            Dict with keys ``input_tokens``, ``output_tokens``,
            ``total_tokens`` (all ``int``).  Missing source keys default to 0.
            ``total_tokens`` is computed as ``input + output`` when not
            present in the raw dict.
        """
        if not raw_usage:
            return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

        # Resolve input tokens (Anthropic / Cline style vs. OpenAI / Ollama style)
        input_tokens: int = raw_usage.get(
            "input_tokens",
            raw_usage.get("prompt_tokens", 0),
        )
        # Resolve output tokens
        output_tokens: int = raw_usage.get(
            "output_tokens",
            raw_usage.get("completion_tokens", 0),
        )
        # Prefer an explicit total; fall back to sum
        total_tokens: int = raw_usage.get("total_tokens")
        if total_tokens is None:
            total_tokens = input_tokens + output_tokens
        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
        }

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
        if context is None:
            raise ValueError("context must be provided")
        if context is None:
            raise ValueError("context must be provided")
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
        context: ConversationContext | None = None,
        app_context: str = "assistant",
    ) -> str:
        """Build a system prompt including tool context.

        This default implementation delegates the preamble to
        :func:`src.shared.python.ai.system_prompts.build_system_prompt` so
        that domain-specific branding is injected by the consuming
        application rather than hardcoded here.  Callers that previously
        relied on the default Golf-Modeling-Suite preamble should pass
        ``app_context="upstream_drift"``.

        Args:
            tools: Available tools to describe.
            expertise_level: User's expertise level.
            context: Optional conversation context containing project and
                prompt-memory metadata.
            app_context: Registry key for the consuming application
                (e.g. ``"upstream_drift"``, ``"gasification"``).  Defaults
                to ``"assistant"`` which produces a brand-neutral preamble.

        Returns:
            System prompt string.
        """
        if tools is None:
            raise ValueError("tools must be provided")
        tool_descriptions = "\n".join(
            f"- {tool.name}: {tool.description}" for tool in tools
        )
        memory_section = self.build_context_instruction_section(context)

        # Response style instructions (Tools #2750)
        style = (context.response_style if context else "standard").lower()
        style_instructions = ""
        if style == "concise":
            style_instructions = (
                "Reply concisely. Prefer code, tables, and short bullet lists over "
                "prose. Skip preamble and recap."
            )
        elif style == "detailed":
            style_instructions = (
                "Reply in detail. Walk through reasoning, name relevant trade-offs, "
                "and include worked examples when they clarify the answer."
            )
        else:  # standard
            style_instructions = (
                "Reply at a standard level of detail. Briefly explain reasoning "
                "where it helps the user act on the answer."
            )

        from src.shared.python.ai.system_prompts import (
            build_system_prompt as _build_preamble,
        )

        preamble = _build_preamble(
            app_context=app_context,
            expertise_level=expertise_level,
        )

        parts = [preamble]
        if style_instructions:
            parts.append(f"Response style: {style_instructions}")
        if memory_section:
            parts.append(memory_section)
        if tool_descriptions:
            parts.append(f"Available tools:\n{tool_descriptions}")

        return "\n\n".join(parts)

    def build_context_instruction_section(
        self,
        context: ConversationContext | None,
    ) -> str:
        """Build repository and persisted-memory prompt context."""
        if context is None:
            return ""

        project_root_value = context.metadata.get("project_root")
        project_root = None
        if isinstance(project_root_value, str) and project_root_value:
            from pathlib import Path

            project_root = Path(project_root_value)

        prompt_memory = context.metadata.get("prompt_memory")
        if not isinstance(prompt_memory, dict):
            prompt_memory = None

        return build_memory_prompt_section(
            prompt_memory=prompt_memory,
            agents_md=load_agents_md(project_root),
        )

    def _classify_error(
        self,
        error: Exception,
        provider: str,
        timeout: float | None = None,
    ) -> AIProviderError:
        """Classify an exception into the canonical AI error hierarchy.

        Performs a string-scan on the exception message to determine the
        most specific ``AIProviderError`` subclass.  Adapters call this
        instead of replicating the classification ladder themselves.

        Pre-check typed provider exceptions *before* calling this helper
        when the provider SDK exposes them (e.g. ``anthropic.RateLimitError``).

        Args:
            error: The original exception to classify.
            provider: Provider name string embedded in the raised error
                (e.g. ``"anthropic"``, ``"openai"``, ``"cline"``).
            timeout: Timeout value [s] to embed in :class:`AITimeoutError`
                when the error is classified as a timeout.

        Returns:
            An :class:`AIProviderError` (or subclass) instance.  Callers
            should raise this with ``raise ... from error``.
        """
        err_str = str(error).lower()

        if any(s in err_str for s in ("rate limit", "429", "too many requests")):
            return AIRateLimitError(
                f"{provider} rate limit exceeded. Please wait and retry.",
                provider=provider,
            )

        if any(s in err_str for s in ("timeout", "timed out")):
            return AITimeoutError(
                f"{provider} request timed out"
                + (f" after {timeout}s" if timeout is not None else ""),
                provider=provider,
                timeout=timeout,
            )

        _conn_keywords = ("connection", "network", "refused", "unreachable")
        if any(s in err_str for s in _conn_keywords):
            return AIConnectionError(
                f"Cannot connect to {provider}. Check your network.",
                provider=provider,
            )

        return AIProviderError(
            f"{provider} error: {error}",
            provider=provider,
        )
