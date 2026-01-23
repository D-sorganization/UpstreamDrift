"""AI Provider Adapters for the Golf Modeling Suite.

This package provides adapters for various AI providers, translating
between the Agent Interface Protocol (AIP) format and provider-specific APIs.

Supported Providers:
    - OpenAI (GPT-4, GPT-4 Turbo)
    - Anthropic (Claude 3.x)
    - Ollama (Local, FREE)
    - Google Gemini (Coming soon)
    - Custom endpoints (via BaseAgentAdapter)

Each adapter implements the BaseAgentAdapter protocol, ensuring consistent
behavior regardless of the underlying provider.

Example:
    >>> from shared.python.ai.adapters import OllamaAdapter
    >>> adapter = OllamaAdapter()  # Free local AI
    >>> success, message = adapter.validate_connection()
    >>> if success:
    ...     response = adapter.send_message("Hello", context, tools)
"""

from src.shared.python.ai.adapters.anthropic_adapter import AnthropicAdapter
from src.shared.python.ai.adapters.base import BaseAgentAdapter, ToolDeclaration
from src.shared.python.ai.adapters.ollama_adapter import OllamaAdapter
from src.shared.python.ai.adapters.openai_adapter import OpenAIAdapter

__all__ = [
    "BaseAgentAdapter",
    "ToolDeclaration",
    "OllamaAdapter",
    "OpenAIAdapter",
    "AnthropicAdapter",
]
