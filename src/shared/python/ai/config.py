"""AI configuration with environment variable support.

This module centralizes all AI-related configuration, allowing values to be
configured via environment variables. This addresses Pragmatic Programmer
Reversibility concerns by externalizing hardcoded configuration.

Environment Variables:
    OLLAMA_HOST: Ollama server URL (default: http://localhost:11434)
    OLLAMA_MODEL: Default Ollama model (default: llama3.1:8b)
    OLLAMA_TIMEOUT: Ollama request timeout in seconds (default: 120.0)

    OPENAI_API_KEY: OpenAI API key (no default - must be provided)
    OPENAI_MODEL: Default OpenAI model (default: gpt-4-turbo-preview)
    OPENAI_TIMEOUT: OpenAI request timeout in seconds (default: 60.0)
    OPENAI_ORGANIZATION: OpenAI organization ID (optional)

    ANTHROPIC_API_KEY: Anthropic API key (no default - must be provided)
    ANTHROPIC_MODEL: Default Anthropic model (default: claude-3-sonnet-20240229)
    ANTHROPIC_TIMEOUT: Anthropic request timeout in seconds (default: 60.0)

    GEMINI_API_KEY: Google Gemini API key (no default - must be provided)
    GEMINI_MODEL: Default Gemini model (default: gemini-pro)

Usage:
    from src.shared.python.ai.config import (
        get_ollama_host,
        get_openai_api_key,
        get_anthropic_model,
    )

    # Get Ollama host (uses env var or default)
    host = get_ollama_host()

    # Get API key (raises if not set and required=True)
    api_key = get_openai_api_key(required=True)
"""

from __future__ import annotations

from src.shared.python.config.environment import get_env, get_env_float
from src.shared.python.logging_config import get_logger

logger = get_logger(__name__)

# ============================================================================
# Environment Variable Names
# ============================================================================

# Ollama (local inference)
ENV_OLLAMA_HOST = "OLLAMA_HOST"
ENV_OLLAMA_MODEL = "OLLAMA_MODEL"
ENV_OLLAMA_TIMEOUT = "OLLAMA_TIMEOUT"

# OpenAI
ENV_OPENAI_API_KEY = "OPENAI_API_KEY"
ENV_OPENAI_MODEL = "OPENAI_MODEL"
ENV_OPENAI_TIMEOUT = "OPENAI_TIMEOUT"
ENV_OPENAI_ORGANIZATION = "OPENAI_ORGANIZATION"

# Anthropic
ENV_ANTHROPIC_API_KEY = "ANTHROPIC_API_KEY"
ENV_ANTHROPIC_MODEL = "ANTHROPIC_MODEL"
ENV_ANTHROPIC_TIMEOUT = "ANTHROPIC_TIMEOUT"

# Google Gemini
ENV_GEMINI_API_KEY = "GEMINI_API_KEY"
ENV_GEMINI_MODEL = "GEMINI_MODEL"

# ============================================================================
# Default Values
# ============================================================================

# Ollama defaults
DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "llama3.1:8b"
DEFAULT_OLLAMA_TIMEOUT = 120.0  # [s] - Local models can be slow

# OpenAI defaults
DEFAULT_OPENAI_MODEL = "gpt-4-turbo-preview"
DEFAULT_OPENAI_TIMEOUT = 60.0  # [s]
DEFAULT_OPENAI_MAX_TOKENS = 128000  # GPT-4 Turbo context window

# Anthropic defaults
DEFAULT_ANTHROPIC_MODEL = "claude-3-sonnet-20240229"
DEFAULT_ANTHROPIC_TIMEOUT = 60.0  # [s]
DEFAULT_ANTHROPIC_MAX_TOKENS = 200000  # Claude 3 context window

# Gemini defaults
DEFAULT_GEMINI_MODEL = "gemini-pro"


# ============================================================================
# Ollama Configuration
# ============================================================================


def get_ollama_host() -> str:
    """Get Ollama server host URL.

    Returns:
        Ollama server URL from OLLAMA_HOST or default.
    """
    return get_env(ENV_OLLAMA_HOST, default=DEFAULT_OLLAMA_HOST) or DEFAULT_OLLAMA_HOST


def get_ollama_model() -> str:
    """Get default Ollama model.

    Returns:
        Model name from OLLAMA_MODEL or default.
    """
    return (
        get_env(ENV_OLLAMA_MODEL, default=DEFAULT_OLLAMA_MODEL) or DEFAULT_OLLAMA_MODEL
    )


def get_ollama_timeout() -> float:
    """Get Ollama request timeout.

    Returns:
        Timeout in seconds from OLLAMA_TIMEOUT or default.
    """
    return (
        get_env_float(ENV_OLLAMA_TIMEOUT, default=DEFAULT_OLLAMA_TIMEOUT)
        or DEFAULT_OLLAMA_TIMEOUT
    )


# ============================================================================
# OpenAI Configuration
# ============================================================================


def get_openai_api_key(*, required: bool = False) -> str | None:
    """Get OpenAI API key.

    Args:
        required: If True, raise error if key not set.

    Returns:
        API key from OPENAI_API_KEY or None.

    Raises:
        EnvironmentError: If required and key not set.
    """
    return get_env(ENV_OPENAI_API_KEY, required=required)


def get_openai_model() -> str:
    """Get default OpenAI model.

    Returns:
        Model name from OPENAI_MODEL or default.
    """
    return (
        get_env(ENV_OPENAI_MODEL, default=DEFAULT_OPENAI_MODEL) or DEFAULT_OPENAI_MODEL
    )


def get_openai_timeout() -> float:
    """Get OpenAI request timeout.

    Returns:
        Timeout in seconds from OPENAI_TIMEOUT or default.
    """
    return (
        get_env_float(ENV_OPENAI_TIMEOUT, default=DEFAULT_OPENAI_TIMEOUT)
        or DEFAULT_OPENAI_TIMEOUT
    )


def get_openai_organization() -> str | None:
    """Get OpenAI organization ID.

    Returns:
        Organization ID from OPENAI_ORGANIZATION or None.
    """
    return get_env(ENV_OPENAI_ORGANIZATION)


# ============================================================================
# Anthropic Configuration
# ============================================================================


def get_anthropic_api_key(*, required: bool = False) -> str | None:
    """Get Anthropic API key.

    Args:
        required: If True, raise error if key not set.

    Returns:
        API key from ANTHROPIC_API_KEY or None.

    Raises:
        EnvironmentError: If required and key not set.
    """
    return get_env(ENV_ANTHROPIC_API_KEY, required=required)


def get_anthropic_model() -> str:
    """Get default Anthropic model.

    Returns:
        Model name from ANTHROPIC_MODEL or default.
    """
    return (
        get_env(ENV_ANTHROPIC_MODEL, default=DEFAULT_ANTHROPIC_MODEL)
        or DEFAULT_ANTHROPIC_MODEL
    )


def get_anthropic_timeout() -> float:
    """Get Anthropic request timeout.

    Returns:
        Timeout in seconds from ANTHROPIC_TIMEOUT or default.
    """
    return (
        get_env_float(ENV_ANTHROPIC_TIMEOUT, default=DEFAULT_ANTHROPIC_TIMEOUT)
        or DEFAULT_ANTHROPIC_TIMEOUT
    )


# ============================================================================
# Gemini Configuration
# ============================================================================


def get_gemini_api_key(*, required: bool = False) -> str | None:
    """Get Google Gemini API key.

    Args:
        required: If True, raise error if key not set.

    Returns:
        API key from GEMINI_API_KEY or None.

    Raises:
        EnvironmentError: If required and key not set.
    """
    return get_env(ENV_GEMINI_API_KEY, required=required)


def get_gemini_model() -> str:
    """Get default Gemini model.

    Returns:
        Model name from GEMINI_MODEL or default.
    """
    return (
        get_env(ENV_GEMINI_MODEL, default=DEFAULT_GEMINI_MODEL) or DEFAULT_GEMINI_MODEL
    )


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    # Environment variable names
    "ENV_OLLAMA_HOST",
    "ENV_OLLAMA_MODEL",
    "ENV_OLLAMA_TIMEOUT",
    "ENV_OPENAI_API_KEY",
    "ENV_OPENAI_MODEL",
    "ENV_OPENAI_TIMEOUT",
    "ENV_OPENAI_ORGANIZATION",
    "ENV_ANTHROPIC_API_KEY",
    "ENV_ANTHROPIC_MODEL",
    "ENV_ANTHROPIC_TIMEOUT",
    "ENV_GEMINI_API_KEY",
    "ENV_GEMINI_MODEL",
    # Default values
    "DEFAULT_OLLAMA_HOST",
    "DEFAULT_OLLAMA_MODEL",
    "DEFAULT_OLLAMA_TIMEOUT",
    "DEFAULT_OPENAI_MODEL",
    "DEFAULT_OPENAI_TIMEOUT",
    "DEFAULT_OPENAI_MAX_TOKENS",
    "DEFAULT_ANTHROPIC_MODEL",
    "DEFAULT_ANTHROPIC_TIMEOUT",
    "DEFAULT_ANTHROPIC_MAX_TOKENS",
    "DEFAULT_GEMINI_MODEL",
    # Getter functions
    "get_ollama_host",
    "get_ollama_model",
    "get_ollama_timeout",
    "get_openai_api_key",
    "get_openai_model",
    "get_openai_timeout",
    "get_openai_organization",
    "get_anthropic_api_key",
    "get_anthropic_model",
    "get_anthropic_timeout",
    "get_gemini_api_key",
    "get_gemini_model",
]
