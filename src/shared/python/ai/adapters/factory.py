"""Unified adapter factory for AI provider resolution.

Provides a single entry point to discover, instantiate, and health-check
AI provider adapters. Supports automatic best-available resolution with
configurable local-first preference.

Supported providers:
    - ollama  (local, free)
    - openai  (GPT-4 / Codex)
    - anthropic (Claude)
    - gemini  (Google)
    - cline   (local IDE agent)

Usage::

    from src.shared.python.ai.adapters.factory import AdapterFactory

    adapter = AdapterFactory.get_best_available(
        prefer_local=True, app_context="gasification"
    )
    if adapter:
        response = adapter.send_message("Hello", context, tools)
"""

from __future__ import annotations

import logging
from typing import Any

from src.shared.python.ai.adapters.base import BaseAgentAdapter
from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)

# Provider resolution order (local-first)
_LOCAL_FIRST_ORDER = ("ollama", "cline", "openai", "anthropic", "gemini")
_CLOUD_FIRST_ORDER = ("openai", "anthropic", "gemini", "ollama", "cline")


class AdapterFactory:
    """Factory for creating and managing AI provider adapters.

    Centralizes adapter creation so consuming applications never need
    to import individual adapter modules. Supports automatic provider
    discovery and health-checking.

    All adapters are created lazily and cached per-configuration.
    """

    _cache: dict[str, BaseAgentAdapter] = {}

    # Provider → (module_path, class_name, env_var_hint) for cloud providers
    _CLOUD_PROVIDERS: dict[str, tuple[str, str, str]] = {
        "openai": (
            "src.shared.python.ai.adapters.openai_adapter",
            "OpenAIAdapter",
            "OPENAI_API_KEY",
        ),
        "anthropic": (
            "src.shared.python.ai.adapters.anthropic_adapter",
            "AnthropicAdapter",
            "ANTHROPIC_API_KEY",
        ),
        "gemini": (
            "src.shared.python.ai.adapters.gemini_adapter",
            "GeminiAdapter",
            "GEMINI_API_KEY",
        ),
    }

    _SUPPORTED_PROVIDERS = frozenset(
        {"ollama", "openai", "codex", "anthropic", "gemini", "cline"}
    )

    @classmethod
    def create(
        cls,
        provider: str,
        *,
        api_key: str | None = None,
        model: str | None = None,
        host: str | None = None,
        timeout: float | None = None,
    ) -> BaseAgentAdapter:
        """Create an adapter for a specific provider.

        Args:
            provider: Provider name (ollama, openai, anthropic, gemini, cline).
            api_key: API key (for cloud providers).
            model: Model override.
            host: Host URL override (for ollama/cline).
            timeout: Request timeout override.

        Returns:
            Configured adapter instance.

        Raises:
            ValueError: If provider is unknown or empty.

        Contract:
            Pre: provider is a non-empty string.
            Post: returned adapter is a BaseAgentAdapter instance.
        """
        if not provider or not provider.strip():
            raise ValueError("provider must be a non-empty string")

        provider = provider.lower().strip()

        # Local adapters — no API key required
        if provider == "ollama":
            from src.shared.python.ai.adapters.ollama_adapter import OllamaAdapter

            return OllamaAdapter(host=host, model=model, timeout=timeout)

        if provider == "cline":
            from src.shared.python.ai.adapters.cline_adapter import ClineAdapter

            return ClineAdapter(host=host, timeout=timeout)

        # "codex" is an alias for OpenAI
        lookup_key = "openai" if provider == "codex" else provider

        # Cloud adapters — DRY key resolution
        if lookup_key in cls._CLOUD_PROVIDERS:
            return cls._create_cloud_adapter(
                lookup_key, api_key=api_key, model=model, timeout=timeout
            )

        raise ValueError(
            f"Unknown provider: {provider}. "
            f"Supported: {', '.join(sorted(cls._SUPPORTED_PROVIDERS))}"
        )

    @classmethod
    def _create_cloud_adapter(
        cls,
        provider: str,
        *,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
    ) -> BaseAgentAdapter:
        """Create a cloud provider adapter with key resolution (DRY helper).

        Args:
            provider: Canonical provider name (openai, anthropic, gemini).
            api_key: Explicit API key override.
            model: Model override.
            timeout: Timeout override.

        Returns:
            Configured cloud adapter.

        Raises:
            ValueError: If no API key available.
        """
        import importlib

        module_path, class_name, env_hint = cls._CLOUD_PROVIDERS[provider]

        key = api_key or cls._resolve_api_key(provider)
        if not key:
            raise ValueError(
                f"{provider.title()} API key required. Set {env_hint} or use "
                f"CredentialManager.store_api_key('{provider}', key)"
            )

        module = importlib.import_module(module_path)
        adapter_cls = getattr(module, class_name)

        # Gemini adapter doesn't accept timeout
        if provider == "gemini":
            return adapter_cls(api_key=key, model=model)
        return adapter_cls(api_key=key, model=model, timeout=timeout)

    @classmethod
    def get_best_available(
        cls,
        *,
        prefer_local: bool = True,
        app_context: str = "assistant",
    ) -> BaseAgentAdapter | None:
        """Find the best available provider and return its adapter.

        Tests providers in priority order and returns the first one
        that passes connection validation.

        Args:
            prefer_local: If True, try local providers first.
            app_context: Application context for system prompts.

        Returns:
            A connected adapter, or None if no providers available.
        """
        order = _LOCAL_FIRST_ORDER if prefer_local else _CLOUD_FIRST_ORDER

        for provider in order:
            try:
                adapter = cls._try_create(provider)
                if adapter is None:
                    continue

                success, msg = adapter.validate_connection()
                if success:
                    logger.info(
                        "Using %s provider for %s: %s",
                        provider,
                        app_context,
                        msg,
                    )
                    return adapter
                logger.debug("Provider %s not available: %s", provider, msg)
            except (ValueError, ImportError, OSError):
                logger.debug("Provider %s not available", provider)
                continue

        logger.warning("No AI providers available")
        return None

    @classmethod
    def get_available_providers(cls) -> list[str]:
        """List providers that pass connection validation.

        Returns:
            List of available provider names.
        """
        available: list[str] = []
        for provider in _LOCAL_FIRST_ORDER:
            try:
                adapter = cls._try_create(provider)
                if adapter is None:
                    continue
                success, _ = adapter.validate_connection()
                if success:
                    available.append(provider)
            except (ValueError, ImportError, OSError):
                continue
        return available

    @classmethod
    def _try_create(cls, provider: str) -> BaseAgentAdapter | None:
        """Try to create an adapter, returning None on failure."""
        try:
            return cls.create(provider)
        except (ValueError, ImportError):
            return None

    @classmethod
    def _resolve_api_key(cls, provider: str) -> str | None:
        """Resolve API key from CredentialManager then env vars."""
        # Try CredentialManager first
        try:
            from chat.credentials import CredentialManager

            mgr = CredentialManager()
            key = mgr.get_api_key(provider)
            if key:
                return key
        except (ImportError, ValueError):
            pass

        # Fall back to config module
        if provider == "openai":
            from src.shared.python.ai.config import get_openai_api_key

            return get_openai_api_key()
        if provider == "anthropic":
            from src.shared.python.ai.config import get_anthropic_api_key

            return get_anthropic_api_key()
        if provider == "gemini":
            from src.shared.python.ai.config import get_gemini_api_key

            return get_gemini_api_key()

        return None

    @classmethod
    def clear_cache(cls) -> None:
        """Clear the adapter cache."""
        cls._cache.clear()
