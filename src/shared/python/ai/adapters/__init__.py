"""AI Provider Adapters for the shared AI infrastructure.

This package provides adapters for various AI providers, translating
between the Agent Interface Protocol (AIP) format and provider-specific APIs.

Supported Providers:
    - OpenAI (GPT-4, GPT-4 Turbo, Codex)
    - Anthropic (Claude 3.x)
    - Ollama (Local, FREE)
    - Cline (Local IDE agent)
    - Google Gemini
    - Custom endpoints (via BaseAgentAdapter)

Each adapter implements the BaseAgentAdapter protocol, ensuring consistent
behavior regardless of the underlying provider.

Example:
    >>> from shared.python.ai.adapters import AdapterFactory
    >>> adapter = AdapterFactory.get_best_available(prefer_local=True)
    >>> if adapter:
    ...     response = adapter.send_message("Hello", context, tools)
"""

from src.shared.python.ai.adapters.anthropic_adapter import AnthropicAdapter
from src.shared.python.ai.adapters.base import BaseAgentAdapter, ToolDeclaration
from src.shared.python.ai.adapters.cline_adapter import ClineAdapter
from src.shared.python.ai.adapters.factory import AdapterFactory
from src.shared.python.ai.adapters.ollama_adapter import OllamaAdapter
from src.shared.python.ai.adapters.openai_adapter import OpenAIAdapter

__all__ = [
    "BaseAgentAdapter",
    "ToolDeclaration",
    "OllamaAdapter",
    "OpenAIAdapter",
    "AnthropicAdapter",
    "ClineAdapter",
    "AdapterFactory",
]
