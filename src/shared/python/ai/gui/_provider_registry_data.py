"""Provider metadata and helper functions extracted from settings_dialog.

Pure data + tiny helpers, no Qt widget instantiation. Importable from
non-GUI code paths (though it does import QComboBox for the populate
helpers, which Qt allows headlessly).
"""

from __future__ import annotations

from enum import Enum, auto

from PyQt6.QtWidgets import QComboBox

from src.shared.python.ai.config import DEFAULT_OLLAMA_MODEL  # noqa: F401  re-exported


class AIProvider(Enum):
    """Available AI providers."""

    OLLAMA = auto()
    OPENAI = auto()
    ANTHROPIC = auto()
    GEMINI = auto()
    CLAUDE_CLI = auto()
    CLINE_CLI = auto()
    CODEX_CLI = auto()
    BITNET = auto()


# Provider display info - explicitly typed for mypy
PROVIDER_INFO: dict[AIProvider, dict[str, str | bool | list[str]]] = {
    AIProvider.OLLAMA: {
        "name": "Ollama",
        "description": "Run AI locally on your computer. No API key needed.",
        "requires_key": False,
        "default_model": "llama3.1:8b",
        "models": [
            "llama3.1:8b",
            "llama3.1:70b",
            "mistral",
            "codellama",
            "opencodeinterpreter",
            "deepseek-coder",
            "phi3",
        ],
    },
    AIProvider.OPENAI: {
        "name": "OpenAI (GPT-4o)",
        "description": "Cloud-based GPT-4o. Requires OpenAI API key.",
        "requires_key": True,
        "key_service": "upstream_drift_openai_key",
        "default_model": "gpt-4o",
        "models": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"],
    },
    AIProvider.ANTHROPIC: {
        "name": "Anthropic (Claude 3.5)",
        "description": "Cloud-based Claude 3.5 Sonnet. Requires Anthropic API key.",
        "requires_key": True,
        "key_service": "upstream_drift_anthropic_key",
        "default_model": "claude-3-5-sonnet-20240620",
        "models": [
            "claude-3-5-sonnet-20241022",
            "claude-3-5-sonnet-20240620",
            "claude-3-5-haiku-20241022",
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
        ],
    },
    AIProvider.GEMINI: {
        "name": "Google Gemini (1.5)",
        "description": "Cloud-based Gemini 1.5. Requires Google API key.",
        "requires_key": True,
        "key_service": "upstream_drift_gemini_key",
        "default_model": "gemini-1.5-pro",
        "models": ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-1.0-pro"],
    },
    AIProvider.CLAUDE_CLI: {
        "name": "Claude CLI (Agent)",
        "description": "Run commands via Anthropic's Claude Code CLI tool.",
        "requires_key": False,
        "default_model": "claude-3-5-sonnet-20241022",
        "models": [
            "claude-3-5-sonnet-20241022",
            "claude-3-5-haiku-20241022",
            "claude-3-opus-20240229",
        ],
    },
    AIProvider.CLINE_CLI: {
        "name": "Cline CLI (Agent)",
        "description": "Run tasks locally with the Cline CLI agent.",
        "requires_key": False,
        "default_model": "claude-3-5-sonnet-20241022",
        "models": ["claude-3-5-sonnet-20241022", "gpt-4o", "o1-preview"],
    },
    AIProvider.CODEX_CLI: {
        "name": "Codex CLI (Agent)",
        "description": "Use OpenAI's Codex CLI for local development.",
        "requires_key": False,
        "default_model": "gpt-4o",
        "models": ["gpt-4o", "o1-preview", "o1-mini", "gpt-4-turbo"],
    },
    AIProvider.BITNET: {
        "name": "BitNet (1.58b)",
        "description": (
            "Run 1.58b quantized models natively via direct subprocess. "
            "No API key needed."
        ),
        "requires_key": False,
        "default_model": "bitnet-1.58b-q4_0.gguf",
        "models": [
            "bitnet-1.58b-q4_0.gguf",
            "bitnet-3b-q4_0.gguf",
        ],
    },
}

DEFAULT_CLINE_HOST = "http://localhost:3000"
BITNET_ROOT_ENV = "BITNET_ROOT"


def provider_display_name(provider: AIProvider) -> str:
    """Return the user-facing provider name from the shared registry."""
    info = PROVIDER_INFO[provider]
    return str(info.get("name", provider.name))


def provider_default_model(provider: AIProvider) -> str:
    """Return the provider default model from the shared registry."""
    info = PROVIDER_INFO[provider]
    default_model = info.get("default_model", "")
    return str(default_model) if isinstance(default_model, str) else ""


def provider_model_names(provider: AIProvider) -> list[str]:
    """Return model names for a provider from the shared registry."""
    info = PROVIDER_INFO[provider]
    models = info.get("models", [])
    if not isinstance(models, list):
        return []
    return [str(model) for model in models]


def populate_provider_combo(combo: QComboBox) -> None:
    """Populate a provider combo from ``AIProvider`` and ``PROVIDER_INFO``."""
    combo.clear()
    for provider in AIProvider:
        combo.addItem(provider_display_name(provider), provider)


def populate_model_combo(
    combo: QComboBox,
    provider: AIProvider,
    selected_model: str | None = None,
) -> None:
    """Populate a model combo and select a valid model for ``provider``."""
    combo.clear()
    models = provider_model_names(provider)
    for model in models:
        combo.addItem(model)

    target = (
        selected_model if selected_model in models else provider_default_model(provider)
    )
    idx = combo.findText(target)
    if idx < 0 and combo.count():
        idx = 0
    if idx >= 0:
        combo.setCurrentIndex(idx)


__all__ = [
    "AIProvider",
    "BITNET_ROOT_ENV",
    "DEFAULT_CLINE_HOST",
    "PROVIDER_INFO",
    "populate_model_combo",
    "populate_provider_combo",
    "provider_default_model",
    "provider_display_name",
    "provider_model_names",
]
