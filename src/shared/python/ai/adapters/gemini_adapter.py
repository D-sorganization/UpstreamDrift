"""Google Gemini API Adapter.

This module provides the adapter interface for Google's Gemini models
via the google-generativeai library.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from src.shared.python.ai.adapters.base import BaseAgentAdapter, ToolDeclaration
from src.shared.python.ai.types import (
    AgentChunk,
    AgentResponse,
    ChatModelInfo,
    ConversationContext,
    ProviderCapabilities,
    ThinkingCapabilities,
    ThinkingLevel,
)
from src.shared.python.contracts import precondition
from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)

# Try to import google-generativeai
try:
    import google.generativeai as genai
    from google.generativeai import GenerativeModel
    from google.generativeai.types import GenerateContentResponse

    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False


class GeminiAdapter(BaseAgentAdapter):
    """Adapter for Google Gemini API."""

    def __init__(self, api_key: str, model: str = "gemini-pro") -> None:
        """Initialize Gemini adapter.

        Args:
            api_key: Google Cloud/AI Studio API Key.
            model: Model identifier (e.g., 'gemini-pro').
        """
        if not HAS_GEMINI:
            raise ImportError(
                "google-generativeai package is not installed. "
                "Run `pip install google-generativeai`."
            )

        self._api_key = api_key
        self._model_name = model

        # Configure global API key (Gemini SDK uses global config usually, but can be instance based)
        # Ideally, we should use a client object if supported, to avoid thread safety issues.
        # But `genai.configure` is global.
        genai.configure(api_key=self._api_key)
        self._model = GenerativeModel(self._model_name)

    @precondition(
        lambda message, context: bool(message.strip()) or (context is not None and bool(context.messages)),
        "message must not be empty unless context has messages",
    )
    def send_message(
        self,
        message: str,
        context: ConversationContext,
        tools: list[ToolDeclaration],
    ) -> AgentResponse:
        """Send a message to Gemini."""
        try:
            chat, effective_message = self._build_chat_session(context, message)
            response = chat.send_message(effective_message)
            return AgentResponse(content=response.text)
        except (RuntimeError, ValueError, OSError) as e:
            logger.error(f"Gemini API error: {e}")
            return AgentResponse(content=f"Error: {e}")

    def stream_response(
        self,
        message: str,
        context: ConversationContext,
        tools: list[ToolDeclaration],
    ) -> Iterator[AgentChunk]:
        """Stream response from Gemini."""
        try:
            chat, effective_message = self._build_chat_session(context, message)
            response: Iterator[GenerateContentResponse] = chat.send_message(
                effective_message, stream=True
            )

            for chunk in response:
                if chunk.text:
                    yield AgentChunk(content=chunk.text)

        except (RuntimeError, ValueError, OSError) as e:
            logger.error(f"Gemini streaming error: {e}")
            yield AgentChunk(content=f"\n[Error: {e}]")

    @property
    def capabilities(self) -> ProviderCapabilities:
        """Return the set of capabilities supported by the Gemini provider."""
        from src.shared.python.ai.types import ProviderCapability

        return ProviderCapabilities(
            supported=frozenset(
                {
                    ProviderCapability.STREAMING,
                    ProviderCapability.VISION,
                }
            ),
            max_tokens=30720,  # Gemini 1.0 Pro context
            model_name=self._model_name,
            provider_name="google",
        )

    def validate_connection(self) -> tuple[bool, str]:
        """Validate Gemini connection."""
        try:
            if not HAS_GEMINI:
                return False, "google-generativeai package missing"

            # Simple prompt to test
            self._model.generate_content("Hello")
            return True, "Connected successfully"
        except (RuntimeError, ValueError, OSError) as e:
            logger.error(f"Gemini validation error: {e}")
            return False, f"Connection failed: {e}"

    def list_models(self) -> list[ChatModelInfo]:
        """Return available Gemini models (static curated list).

        Returns:
            List of ChatModelInfo entries for Google Gemini.
        """
        return [
            ChatModelInfo(
                model_id="gemini-2.0-flash",
                display_name="Gemini 2.0 Flash",
                context_window=1_048_576,
                supports_thinking=True,
            ),
            ChatModelInfo(
                model_id="gemini-1.5-pro",
                display_name="Gemini 1.5 Pro",
                context_window=2_000_000,
                supports_thinking=True,
            ),
            ChatModelInfo(
                model_id="gemini-1.5-flash",
                display_name="Gemini 1.5 Flash",
                context_window=1_048_576,
                supports_thinking=False,
            ),
        ]

    def thinking_capabilities(self) -> ThinkingCapabilities:
        """Return thinking capabilities for Gemini models.

        Gemini 1.5 Pro and 2.0 Flash support thinking_budget (auto/off/manual).

        Returns:
            ThinkingCapabilities for the current Gemini model.
        """
        _thinking_models = {
            "gemini-2.0-flash",
            "gemini-2.0-flash-thinking",
            "gemini-1.5-pro",
        }
        supports = any(m in self._model_name for m in _thinking_models)
        if supports:
            return ThinkingCapabilities(
                supports_levels=True,
                available_levels=[
                    ThinkingLevel.OFF,
                    ThinkingLevel.AUTO,
                    ThinkingLevel.HIGH,
                ],
            )
        return ThinkingCapabilities(
            supports_levels=False,
            available_levels=[ThinkingLevel.OFF],
        )

    def _build_chat_session(
        self, context: ConversationContext, current_message: str
    ) -> tuple[Any, str]:
        """Build a chat session with history."""
        if not (context is not None):
            raise ValueError("context must be provided")
        history = []
        msg_list = list(context.messages)

        effective_message = current_message
        if not effective_message.strip() and msg_list:
            for i in range(len(msg_list) - 1, -1, -1):
                if msg_list[i].role == "user":
                    effective_message = msg_list[i].content
                    msg_list.pop(i)
                    break

        for msg in msg_list:
            role = "user" if msg.role == "user" else "model"
            history.append({"role": role, "parts": [msg.content]})

        chat = self._model.start_chat(history=history)  # type: ignore[arg-type]
        return chat, effective_message
