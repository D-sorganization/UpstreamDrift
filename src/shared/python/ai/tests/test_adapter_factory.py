"""Tests for the AdapterFactory and system prompts.

Covers:
- Factory creation for each provider type
- Best-available provider resolution
- API key resolution (CredentialManager + env vars)
- System prompt building and registration
- Unknown provider handling
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

# ── AdapterFactory tests ─────────────────────────────────────────────


class TestAdapterFactory:
    """Tests for the unified adapter factory."""

    def test_create_ollama(self) -> None:
        from src.shared.python.ai.adapters.factory import AdapterFactory

        adapter = AdapterFactory.create("ollama")
        assert adapter is not None
        assert adapter.capabilities.provider_name == "ollama"

    def test_create_ollama_case_insensitive(self) -> None:
        from src.shared.python.ai.adapters.factory import AdapterFactory

        adapter = AdapterFactory.create("OLLAMA")
        assert adapter is not None

    def test_create_unknown_raises(self) -> None:
        from src.shared.python.ai.adapters.factory import AdapterFactory

        with pytest.raises(ValueError, match="Unknown provider"):
            AdapterFactory.create("nonexistent")

    def test_create_empty_provider_raises(self) -> None:
        """DbC: empty provider string is a precondition violation."""
        from src.shared.python.ai.adapters.factory import AdapterFactory

        with pytest.raises(ValueError, match="non-empty"):
            AdapterFactory.create("")

    def test_create_whitespace_provider_raises(self) -> None:
        """DbC: whitespace-only provider string is a precondition violation."""
        from src.shared.python.ai.adapters.factory import AdapterFactory

        with pytest.raises(ValueError, match="non-empty"):
            AdapterFactory.create("   ")

    def test_create_openai_without_key_raises(self) -> None:
        from src.shared.python.ai.adapters.factory import AdapterFactory

        with (
            patch.dict("os.environ", {}, clear=True),
            patch(
                "src.shared.python.ai.adapters.factory.AdapterFactory._resolve_api_key",
                return_value=None,
            ),
            pytest.raises(ValueError, match="API key required"),
        ):
            AdapterFactory.create("openai")

    def test_create_anthropic_without_key_raises(self) -> None:
        from src.shared.python.ai.adapters.factory import AdapterFactory

        with (
            patch.dict("os.environ", {}, clear=True),
            patch(
                "src.shared.python.ai.adapters.factory.AdapterFactory._resolve_api_key",
                return_value=None,
            ),
            pytest.raises(ValueError, match="API key required"),
        ):
            AdapterFactory.create("anthropic")

    def test_create_cline(self) -> None:
        from src.shared.python.ai.adapters.factory import AdapterFactory

        adapter = AdapterFactory.create("cline")
        assert adapter is not None
        assert adapter.capabilities.provider_name == "cline"

    def test_codex_alias_creates_openai(self) -> None:
        """'codex' is an alias for OpenAI adapter."""
        from src.shared.python.ai.adapters.factory import AdapterFactory

        adapter = AdapterFactory.create("codex", api_key="sk-test-key")
        assert adapter is not None
        assert adapter.capabilities.provider_name == "openai"

    def test_get_best_available_returns_none_when_no_providers(self) -> None:
        from src.shared.python.ai.adapters.factory import AdapterFactory

        with patch.object(
            AdapterFactory,
            "_try_create",
            return_value=None,
        ):
            result = AdapterFactory.get_best_available()
            assert result is None

    def test_get_best_available_returns_first_valid(self) -> None:
        from src.shared.python.ai.adapters.factory import AdapterFactory

        mock_adapter = MagicMock()
        mock_adapter.validate_connection.return_value = (True, "OK")

        with patch.object(
            AdapterFactory,
            "_try_create",
            side_effect=lambda p: mock_adapter if p == "ollama" else None,
        ):
            result = AdapterFactory.get_best_available(prefer_local=True)
            assert result is mock_adapter

    def test_resolve_api_key_from_credential_manager(self) -> None:
        from src.shared.python.ai.adapters.factory import AdapterFactory

        mock_mgr = MagicMock()
        mock_mgr.get_api_key.return_value = "sk-from-keyring"

        # Patch the import inside _resolve_api_key by intercepting builtins
        original_import = (
            __builtins__.__import__
            if hasattr(__builtins__, "__import__")
            else __import__
        )

        def mock_import(name: str, *args: object, **kwargs: object) -> object:
            if name == "chat.credentials":
                mod = MagicMock()
                mod.CredentialManager.return_value = mock_mgr
                return mod
            return original_import(name, *args, **kwargs)  # type: ignore

        with patch("builtins.__import__", side_effect=mock_import):
            key = AdapterFactory._resolve_api_key("openai")
            assert key == "sk-from-keyring"

    def test_resolve_api_key_fallback_to_env(self) -> None:
        from src.shared.python.ai.adapters.factory import AdapterFactory

        # Make the chat.credentials import fail by patching builtins
        original_import = (
            __builtins__.__import__
            if hasattr(__builtins__, "__import__")
            else __import__
        )

        def mock_import(name: str, *args: object, **kwargs: object) -> object:
            if name == "chat.credentials":
                raise ImportError("no chat.credentials")
            return original_import(name, *args, **kwargs)  # type: ignore

        with (
            patch("builtins.__import__", side_effect=mock_import),
            patch(
                "src.shared.python.ai.config.get_openai_api_key",
                return_value="sk-from-env",
            ),
        ):
            key = AdapterFactory._resolve_api_key("openai")
            assert key == "sk-from-env"

    def test_clear_cache(self) -> None:
        from src.shared.python.ai.adapters.factory import AdapterFactory

        AdapterFactory._cache["test"] = MagicMock()  # type: ignore[index]
        AdapterFactory.clear_cache()
        assert len(AdapterFactory._cache) == 0


# ── System prompts tests ─────────────────────────────────────────────


class TestSystemPrompts:
    def test_build_default_prompt(self) -> None:
        from src.shared.python.ai.system_prompts import build_system_prompt

        prompt = build_system_prompt()
        assert "AI Assistant" in prompt
        assert "beginner" in prompt

    def test_build_gasification_prompt(self) -> None:
        from src.shared.python.ai.system_prompts import build_system_prompt

        prompt = build_system_prompt(app_context="gasification")
        assert "Integrated Process Simulator" in prompt
        assert "thermodynamic" in prompt

    def test_build_upstream_drift_prompt(self) -> None:
        from src.shared.python.ai.system_prompts import build_system_prompt

        prompt = build_system_prompt(app_context="upstream_drift")
        assert "UpstreamDrift" in prompt
        assert "biomechanics" in prompt

    def test_build_with_expertise(self) -> None:
        from src.shared.python.ai.system_prompts import build_system_prompt

        prompt = build_system_prompt(expertise_level="expert")
        assert "expert" in prompt

    def test_build_with_extra_instructions(self) -> None:
        from src.shared.python.ai.system_prompts import build_system_prompt

        prompt = build_system_prompt(
            extra_instructions="Focus on heat transfer analysis."
        )
        assert "heat transfer" in prompt

    def test_register_custom_context(self) -> None:
        from src.shared.python.ai.system_prompts import (
            build_system_prompt,
            register_app_context,
        )

        register_app_context(
            "my_custom_app",
            name="Custom App",
            description="a custom engineering tool",
            capabilities=["Custom analysis"],
        )
        prompt = build_system_prompt(app_context="my_custom_app")
        assert "Custom App" in prompt
        assert "Custom analysis" in prompt

    def test_register_empty_key_raises(self) -> None:
        from src.shared.python.ai.system_prompts import register_app_context

        with pytest.raises(ValueError, match="non-empty"):
            register_app_context("", "name", "desc", [])

    def test_get_registered_contexts(self) -> None:
        from src.shared.python.ai.system_prompts import get_registered_contexts

        contexts = get_registered_contexts()
        assert "gasification" in contexts
        assert "upstream_drift" in contexts

    def test_unknown_context_uses_default(self) -> None:
        from src.shared.python.ai.system_prompts import build_system_prompt

        prompt = build_system_prompt(app_context="totally_unknown")
        assert "AI Assistant" in prompt


# ── ClineAdapter tests ───────────────────────────────────────────────


class TestClineAdapter:
    def test_construction(self) -> None:
        from src.shared.python.ai.adapters.cline_adapter import ClineAdapter

        adapter = ClineAdapter()
        assert adapter._host == "http://localhost:3000"
        assert adapter._timeout == 120.0

    def test_custom_host(self) -> None:
        from src.shared.python.ai.adapters.cline_adapter import ClineAdapter

        adapter = ClineAdapter(host="http://custom:9000/")
        assert adapter._host == "http://custom:9000"  # trailing slash stripped

    def test_capabilities(self) -> None:
        from src.shared.python.ai.adapters.cline_adapter import ClineAdapter
        from src.shared.python.ai.types import ProviderCapability

        adapter = ClineAdapter()
        caps = adapter.capabilities
        assert caps.provider_name == "cline"
        assert caps.has_capability(ProviderCapability.FUNCTION_CALLING)
        assert caps.has_capability(ProviderCapability.STREAMING)

    def test_validate_connection_failure(self) -> None:
        from src.shared.python.ai.adapters.cline_adapter import ClineAdapter

        adapter = ClineAdapter(host="http://localhost:1")

        mock_client = MagicMock()
        mock_client.get.side_effect = ConnectionError("refused")
        adapter._client = mock_client

        success, msg = adapter.validate_connection()
        assert success is False
        assert "Cannot connect" in msg
