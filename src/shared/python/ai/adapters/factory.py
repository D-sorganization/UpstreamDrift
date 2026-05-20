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
    - bitnet  (local 1.58b models via direct subprocess)

Usage::

    from src.shared.python.ai.adapters.factory import AdapterFactory

    adapter = AdapterFactory.get_best_available(
        prefer_local=True, app_context="gasification"
    )
    if adapter:
        response = adapter.send_message("Hello", context, tools)
"""

from __future__ import annotations

from src.shared.python.ai.adapters.base import BaseAgentAdapter
from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)

# Provider resolution order (local-first)
# `claude_code`, `codex_cli`, and `gemini_cli` are LOCAL-shaped: they invoke a
# CLI on the host machine. They appear before cloud providers in local-first
# ordering, but after the truly-local (no network) options like Ollama and
# BitNet — the CLIs still hit the cloud underneath, just without exposing
# API keys to the application.
_LOCAL_FIRST_ORDER = (
    "ollama",
    "bitnet",
    "claude_code",
    "codex_cli",
    "gemini_cli",
    "cline",
    "openai",
    "anthropic",
    "gemini",
)
_CLOUD_FIRST_ORDER = (
    "openai",
    "anthropic",
    "gemini",
    "claude_code",
    "codex_cli",
    "gemini_cli",
    "ollama",
    "bitnet",
    "cline",
)

# Type alias for the cache key tuple: (provider, api_key, model, host, timeout)
_CacheKey = tuple[str, str | None, str | None, str | None, float | None]


class AdapterFactory:
    """Factory for creating and managing AI provider adapters.

    Centralizes adapter creation so consuming applications never need
    to import individual adapter modules. Supports automatic provider
    discovery and health-checking.

    Adapters are cached per-configuration: calling ``create()`` twice with
    identical arguments returns the **same** instance, avoiding redundant
    HTTP client setup, auth handshakes, and capability probes.  Call
    ``clear_cache()`` to force fresh construction (e.g. after rotating
    credentials).

    Cache key: ``(provider, api_key, model, host, timeout)`` — all
    keyword arguments are included so that any configuration difference
    produces a distinct cached entry.
    """

    _cache: dict[_CacheKey, BaseAgentAdapter] = {}

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
        {
            "ollama",
            "bitnet",
            "openai",
            "codex",  # historical alias for OpenAI; kept for back-compat
            "codex_cli",  # the @openai/codex CLI agent (distinct from OpenAI API)
            "anthropic",
            "claude_code",  # the Claude Code CLI agent (distinct from Anthropic API)
            "gemini",
            "gemini_cli",  # the @google/gemini-cli CLI agent (distinct from Gemini API)
            "cline",
        }
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

        Returns a cached adapter when the same ``(provider, api_key, model,
        host, timeout)`` combination has been requested before, avoiding
        redundant adapter construction.  Use ``clear_cache()`` to invalidate
        the cache (e.g. after rotating credentials).

        Args:
            provider: Provider name (ollama, openai, anthropic, gemini, cline).
            api_key: API key (for cloud providers).
            model: Model override.
            host: Host URL override (for ollama/cline).
            timeout: Request timeout override.

        Returns:
            Configured adapter instance (may be a previously cached object).

        Raises:
            ValueError: If provider is unknown or empty.

        Contract:
            Pre: provider is a non-empty string.
            Post: returned adapter is a BaseAgentAdapter instance.
        """
        if not provider or not provider.strip():
            raise ValueError("provider must be a non-empty string")

        provider = provider.lower().strip()

        # Check cache before constructing a new adapter
        cache_key: _CacheKey = (provider, api_key, model, host, timeout)
        if cache_key in cls._cache:
            logger.debug("AdapterFactory cache hit for provider=%s", provider)
            return cls._cache[cache_key]

        # Construct adapter, then store in cache before returning
        adapter: BaseAgentAdapter

        # Local adapters — no API key required
        if provider == "ollama":
            from src.shared.python.ai.adapters.ollama_adapter import OllamaAdapter

            adapter = OllamaAdapter(host=host, model=model, timeout=timeout)

        elif provider == "cline":
            from src.shared.python.ai.adapters.cline_adapter import ClineAdapter

            adapter = ClineAdapter(host=host, timeout=timeout)

        elif provider == "bitnet":
            from src.shared.python.ai.adapters.bitnet_adapter import BitnetAdapter

            # Bitnet uses 'host' param as bitnet_root in this context if provided
            adapter = BitnetAdapter(model=model, bitnet_root=host)

        elif provider == "claude_code":
            from src.shared.python.ai.adapters.claude_code_adapter import (
                ClaudeCodeAdapter,
            )

            # `host` is reused here as an explicit binary path override —
            # the CLI's "host" is its own filesystem path, not a URL.
            adapter = ClaudeCodeAdapter(binary=host, model=model, timeout=timeout)

        elif provider == "codex_cli":
            from src.shared.python.ai.adapters.codex_cli_adapter import (
                CodexCliAdapter,
            )

            # `host` reused as explicit binary path override (see claude_code above).
            adapter = CodexCliAdapter(binary=host, model=model, timeout=timeout)

        elif provider == "gemini_cli":
            from src.shared.python.ai.adapters.gemini_cli_adapter import (
                GeminiCliAdapter,
            )

            # `host` reused as explicit binary path override (see claude_code above).
            adapter = GeminiCliAdapter(binary=host, model=model, timeout=timeout)

        else:
            # Historical "codex" alias resolves to OpenAI API.
            # The new `@openai/codex` CLI agent uses provider="codex_cli" instead.
            lookup_key = "openai" if provider == "codex" else provider

            # Cloud adapters — DRY key resolution
            if lookup_key in cls._CLOUD_PROVIDERS:
                adapter = cls._create_cloud_adapter(
                    lookup_key, api_key=api_key, model=model, timeout=timeout
                )
            else:
                raise ValueError(
                    f"Unknown provider: {provider}. "
                    f"Supported: {', '.join(sorted(cls._SUPPORTED_PROVIDERS))}"
                )

        cls._cache[cache_key] = adapter
        logger.debug("AdapterFactory cached new adapter for provider=%s", provider)
        return adapter

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

        # `adapter_cls` is loaded dynamically via getattr() so mypy sees it
        # as Any. The annotated local rebinds the return to the contract this
        # method declares, which is more honest than a bare `# type: ignore`.
        # Gemini adapter doesn't accept timeout.
        if provider == "gemini":
            adapter: BaseAgentAdapter = adapter_cls(api_key=key, model=model)
            return adapter
        adapter = adapter_cls(api_key=key, model=model, timeout=timeout)
        return adapter

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
        """Resolve API key from CredentialManager then env vars.

        Returns the API key as a string, or None when no credential is
        available. Cross-module helpers below (``get_*_api_key``,
        ``CredentialManager.get_api_key``) all return ``str | None``, but
        CI runs mypy with ``--follow-imports=skip`` which strips that
        signature to ``Any``. Local annotations restore the contract.
        """
        # Try CredentialManager first
        try:
            from chat.credentials import CredentialManager

            mgr = CredentialManager()
            key: str | None = mgr.get_api_key(provider)
            if key:
                return key
        except (ImportError, ValueError):
            pass

        # Fall back to config module
        env_key: str | None
        if provider == "openai":
            from src.shared.python.ai.config import get_openai_api_key

            env_key = get_openai_api_key()
            return env_key
        if provider == "anthropic":
            from src.shared.python.ai.config import get_anthropic_api_key

            env_key = get_anthropic_api_key()
            return env_key
        if provider == "gemini":
            from src.shared.python.ai.config import get_gemini_api_key

            env_key = get_gemini_api_key()
            return env_key

        return None

    @classmethod
    def clear_cache(cls) -> None:
        """Clear the adapter cache.

        Forces the next ``create()`` call to construct a fresh adapter
        instance even for previously seen configurations.  Use after
        rotating API keys or changing connection settings.
        """
        cls._cache.clear()
