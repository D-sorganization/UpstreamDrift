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
    ConversationContext,
    ProviderCapabilities,
)
from src.shared.python.logging_config import get_logger

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

    def __init__(self, api_key: str, model: str = "gemini-pro"):
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

    def send_message(
        self,
        message: str,
        context: ConversationContext,
        tools: list[ToolDeclaration],
    ) -> AgentResponse:
        """Send a message to Gemini."""
        try:
            chat = self._build_chat_session(context)
            response = chat.send_message(message)
            return AgentResponse(content=response.text)
        except Exception as e:
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
            chat = self._build_chat_session(context)
            response: Iterator[GenerateContentResponse] = chat.send_message(
                message, stream=True
            )

            for chunk in response:
                if chunk.text:
                    yield AgentChunk(content=chunk.text)

        except Exception as e:
            logger.error(f"Gemini streaming error: {e}")
            yield AgentChunk(content=f"\n[Error: {e}]")

    @property
    def capabilities(self) -> ProviderCapabilities:
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
        except Exception as e:
            logger.error(f"Gemini validation error: {e}")
            return False, f"Connection failed: {e}"

    def _build_chat_session(self, context: ConversationContext) -> Any:
        """Build a chat session with history."""
        history = []
        for msg in context.messages:
            role = "user" if msg.role == "user" else "model"
            history.append({"role": role, "parts": [msg.content]})

        return self._model.start_chat(history=history)
