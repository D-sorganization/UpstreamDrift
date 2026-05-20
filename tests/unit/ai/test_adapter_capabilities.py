"""Tests for adapter list_models() and thinking_capabilities() contract.

Every concrete adapter must expose:
- list_models() -> list[ChatModelInfo]
- thinking_capabilities() -> ThinkingCapabilities

HTTP calls are stubbed via unittest.mock so no real provider is contacted.
"""

from __future__ import annotations

import pytest

from src.shared.python.chat.models import (
    ChatModelInfo,
    ThinkingCapabilities,
    ThinkingLevel,
)


class TestOllamaAdapterCapabilities:
    """OllamaAdapter must expose model list and thinking capabilities."""

    def test_list_models_returns_nonempty_list(self) -> None:
        """list_models() returns at least one ChatModelInfo entry."""
        from unittest.mock import MagicMock, patch

        from src.shared.python.ai.adapters.ollama_adapter import OllamaAdapter

        adapter = OllamaAdapter()
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "models": [{"name": "llama3.1:8b"}, {"name": "mistral"}]
        }
        mock_response.status_code = 200

        with patch("requests.get", return_value=mock_response):
            models = adapter.list_models()

        assert isinstance(models, list)
        assert len(models) >= 1
        assert all(isinstance(m, ChatModelInfo) for m in models)
        assert all(m.model_id for m in models)

    def test_thinking_capabilities_returns_dataclass(self) -> None:
        """thinking_capabilities() returns ThinkingCapabilities."""
        from src.shared.python.ai.adapters.ollama_adapter import OllamaAdapter

        adapter = OllamaAdapter()
        caps = adapter.thinking_capabilities()

        assert isinstance(caps, ThinkingCapabilities)
        assert isinstance(caps.supports_levels, bool)
        assert isinstance(caps.available_levels, list)


class TestOpenAIAdapterCapabilities:
    """OpenAIAdapter must expose model list and thinking capabilities."""

    def test_list_models_returns_nonempty_list(self) -> None:
        """list_models() returns at least one ChatModelInfo entry."""
        from src.shared.python.ai.adapters.openai_adapter import OpenAIAdapter

        adapter = OpenAIAdapter(api_key="test-key-openai")
        models = adapter.list_models()

        assert isinstance(models, list)
        assert len(models) >= 1
        assert all(isinstance(m, ChatModelInfo) for m in models)
        assert all(m.model_id for m in models)

    def test_thinking_capabilities_returns_dataclass(self) -> None:
        """thinking_capabilities() returns ThinkingCapabilities."""
        from src.shared.python.ai.adapters.openai_adapter import OpenAIAdapter

        adapter = OpenAIAdapter(api_key="test-key-openai")
        caps = adapter.thinking_capabilities()

        assert isinstance(caps, ThinkingCapabilities)
        assert isinstance(caps.supports_levels, bool)
        assert isinstance(caps.available_levels, list)


class TestAnthropicAdapterCapabilities:
    """AnthropicAdapter must expose model list and thinking capabilities."""

    def test_list_models_returns_nonempty_list(self) -> None:
        """list_models() returns at least one ChatModelInfo entry."""
        from src.shared.python.ai.adapters.anthropic_adapter import AnthropicAdapter

        adapter = AnthropicAdapter(api_key="test-key-anthropic")
        models = adapter.list_models()

        assert isinstance(models, list)
        assert len(models) >= 1
        assert all(isinstance(m, ChatModelInfo) for m in models)
        assert all(m.model_id for m in models)

    def test_thinking_capabilities_returns_dataclass(self) -> None:
        """thinking_capabilities() returns ThinkingCapabilities with levels."""
        from src.shared.python.ai.adapters.anthropic_adapter import AnthropicAdapter

        adapter = AnthropicAdapter(api_key="test-key-anthropic")
        caps = adapter.thinking_capabilities()

        assert isinstance(caps, ThinkingCapabilities)
        # Anthropic supports extended thinking
        assert caps.supports_levels is True
        assert len(caps.available_levels) > 0

    def test_thinking_levels_include_known_values(self) -> None:
        """Anthropic thinking levels include at least 'off' and a budget level."""
        from src.shared.python.ai.adapters.anthropic_adapter import AnthropicAdapter

        adapter = AnthropicAdapter(api_key="test-key-anthropic")
        caps = adapter.thinking_capabilities()

        level_names = [lvl.value for lvl in caps.available_levels]
        assert ThinkingLevel.OFF.value in level_names


class TestGeminiAdapterCapabilities:
    """GeminiAdapter must expose model list and thinking capabilities."""

    def test_list_models_returns_nonempty_list(self) -> None:
        """list_models() returns at least one ChatModelInfo entry."""
        from unittest.mock import MagicMock, patch

        with patch.dict(
            "sys.modules",
            {
                "google": MagicMock(),
                "google.generativeai": MagicMock(),
                "google.generativeai.types": MagicMock(),
            },
        ):
            import importlib

            import src.shared.python.ai.adapters.gemini_adapter as gem_mod

            importlib.reload(gem_mod)
            gem_mod.HAS_GEMINI = True

            mock_genai = MagicMock()
            gem_mod.genai = mock_genai

            adapter = gem_mod.GeminiAdapter.__new__(gem_mod.GeminiAdapter)
            adapter._api_key = "test-key-gemini"
            adapter._model_name = "gemini-1.5-pro"
            adapter._model = MagicMock()

            models = adapter.list_models()

        assert isinstance(models, list)
        assert len(models) >= 1
        assert all(isinstance(m, ChatModelInfo) for m in models)

    def test_thinking_capabilities_returns_dataclass(self) -> None:
        """thinking_capabilities() returns ThinkingCapabilities."""
        from unittest.mock import MagicMock, patch

        with patch.dict(
            "sys.modules",
            {
                "google": MagicMock(),
                "google.generativeai": MagicMock(),
                "google.generativeai.types": MagicMock(),
            },
        ):
            import importlib

            import src.shared.python.ai.adapters.gemini_adapter as gem_mod

            importlib.reload(gem_mod)
            gem_mod.HAS_GEMINI = True
            gem_mod.genai = MagicMock()

            adapter = gem_mod.GeminiAdapter.__new__(gem_mod.GeminiAdapter)
            adapter._api_key = "test-key-gemini"
            adapter._model_name = "gemini-1.5-pro"
            adapter._model = MagicMock()

            caps = adapter.thinking_capabilities()

        assert isinstance(caps, ThinkingCapabilities)
        assert isinstance(caps.supports_levels, bool)
        assert isinstance(caps.available_levels, list)
