"""Tests for adapter ``list_models()`` and ``thinking_capabilities()``.

Every concrete adapter must expose:

* ``list_models() -> list[str]`` -- a non-empty catalogue of model ids, with a
  static fallback so offline environments can still populate a dropdown. The
  contract is declared on :meth:`AIProviderAdapter.list_models` in
  ``src/shared/python/ai/adapters/base.py``.
* ``thinking_capabilities() -> ThinkingCapabilities`` -- the provider/levels
  dataclass from ``src.shared.python.chat_contracts.models``, re-exported by
  ``src.shared.python.chat.models``.

History: this file previously asserted a ``list[ChatModelInfo]`` return and a
``ThinkingCapabilities(supports_levels=..., available_levels=[...])`` shape.
Neither has existed since the chat contract moved to
``chat_contracts.models`` -- ``ThinkingCapabilities`` now carries ``provider``,
``levels`` and ``default_level_name``, and adapters return plain model-id
strings. The assertions below follow the current contract; the checks are
stronger, not weaker, because ``default_level_name`` membership is now pinned.

Provider calls are stubbed via ``unittest.mock`` so no real provider is
contacted.
"""

from __future__ import annotations

from typing import Any

import pytest

from src.shared.python.chat.models import ThinkingCapabilities


pytestmark = [pytest.mark.unit]


def _assert_model_catalogue(models: Any) -> None:
    """Assert ``models`` satisfies the ``list_models()`` contract."""
    assert isinstance(models, list)
    assert models, "list_models() must never return an empty catalogue"
    assert all(isinstance(name, str) for name in models)
    assert all(name.strip() for name in models)


def _assert_thinking_capabilities(caps: Any) -> None:
    """Assert ``caps`` satisfies the ``thinking_capabilities()`` contract."""
    assert isinstance(caps, ThinkingCapabilities)
    assert isinstance(caps.provider, str) and caps.provider.strip()
    assert caps.levels, "capabilities must expose at least one level"
    assert caps.default_level_name in caps.level_names()


class TestOllamaAdapterCapabilities:
    """OllamaAdapter must expose model list and thinking capabilities."""

    def test_list_models_returns_nonempty_list(self) -> None:
        """A live probe response is turned into a list of model ids."""
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

        _assert_model_catalogue(models)

    def test_thinking_capabilities_returns_dataclass(self) -> None:
        """thinking_capabilities() returns a valid ThinkingCapabilities."""
        from src.shared.python.ai.adapters.ollama_adapter import OllamaAdapter

        adapter = OllamaAdapter()

        _assert_thinking_capabilities(adapter.thinking_capabilities())


class TestOpenAIAdapterCapabilities:
    """OpenAIAdapter must expose model list and thinking capabilities."""

    def test_list_models_returns_nonempty_list(self) -> None:
        """The static catalogue is returned when no live probe succeeds."""
        from src.shared.python.ai.adapters.openai_adapter import OpenAIAdapter

        adapter = OpenAIAdapter(api_key="test-key-openai")

        _assert_model_catalogue(adapter.list_models())

    def test_thinking_capabilities_returns_dataclass(self) -> None:
        """thinking_capabilities() returns a valid ThinkingCapabilities."""
        from src.shared.python.ai.adapters.openai_adapter import OpenAIAdapter

        adapter = OpenAIAdapter(api_key="test-key-openai")

        _assert_thinking_capabilities(adapter.thinking_capabilities())


class TestAnthropicAdapterCapabilities:
    """AnthropicAdapter must expose model list and thinking capabilities."""

    def test_list_models_returns_nonempty_list(self) -> None:
        """The static catalogue is returned when no live probe succeeds."""
        from src.shared.python.ai.adapters.anthropic_adapter import AnthropicAdapter

        adapter = AnthropicAdapter(api_key="test-key-anthropic")

        _assert_model_catalogue(adapter.list_models())

    def test_thinking_capabilities_returns_dataclass(self) -> None:
        """thinking_capabilities() returns a valid ThinkingCapabilities."""
        from src.shared.python.ai.adapters.anthropic_adapter import AnthropicAdapter

        adapter = AnthropicAdapter(api_key="test-key-anthropic")

        _assert_thinking_capabilities(adapter.thinking_capabilities())

    @pytest.mark.parametrize("model", ["claude-3-5-sonnet-20241022", "claude-3-opus"])
    def test_sonnet_and_opus_expose_extended_thinking(self, model: str) -> None:
        """Sonnet/Opus models expose the four-level extended-thinking budget."""
        from src.shared.python.ai.adapters.anthropic_adapter import AnthropicAdapter

        adapter = AnthropicAdapter(api_key="test-key-anthropic", model=model)
        caps = adapter.thinking_capabilities()

        _assert_thinking_capabilities(caps)
        assert caps.provider == "anthropic"
        assert caps.level_names() == ("none", "low", "medium", "high")

    def test_non_reasoning_model_exposes_only_the_none_level(self) -> None:
        """A model without extended thinking advertises exactly one level."""
        from src.shared.python.ai.adapters.anthropic_adapter import AnthropicAdapter

        adapter = AnthropicAdapter(
            api_key="test-key-anthropic",
            model="claude-3-haiku-20240307",
        )
        caps = adapter.thinking_capabilities()

        _assert_thinking_capabilities(caps)
        assert caps.level_names() == ("none",)


class TestGeminiAdapterCapabilities:
    """GeminiAdapter must expose model list and thinking capabilities."""

    @staticmethod
    def _make_adapter() -> Any:
        """Build a GeminiAdapter without importing the real SDK."""
        import importlib
        from unittest.mock import MagicMock

        import src.shared.python.ai.adapters.gemini_adapter as gem_mod

        importlib.reload(gem_mod)
        gem_mod.HAS_GEMINI = True
        gem_mod.genai = MagicMock()

        adapter = gem_mod.GeminiAdapter.__new__(gem_mod.GeminiAdapter)
        adapter._api_key = "test-key-gemini"
        adapter._model_name = "gemini-1.5-pro"
        adapter._model = MagicMock()
        return adapter

    def test_list_models_returns_nonempty_list(self) -> None:
        """The static catalogue is returned when no live probe succeeds."""
        from unittest.mock import MagicMock, patch

        with patch.dict(
            "sys.modules",
            {
                "google": MagicMock(),
                "google.generativeai": MagicMock(),
                "google.generativeai.types": MagicMock(),
            },
        ):
            models = self._make_adapter().list_models()

        _assert_model_catalogue(models)

    def test_thinking_capabilities_returns_dataclass(self) -> None:
        """thinking_capabilities() returns a valid ThinkingCapabilities."""
        from unittest.mock import MagicMock, patch

        with patch.dict(
            "sys.modules",
            {
                "google": MagicMock(),
                "google.generativeai": MagicMock(),
                "google.generativeai.types": MagicMock(),
            },
        ):
            caps = self._make_adapter().thinking_capabilities()

        _assert_thinking_capabilities(caps)
