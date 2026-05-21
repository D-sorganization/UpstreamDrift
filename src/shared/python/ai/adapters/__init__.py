"""AI Provider Adapters for the shared AI infrastructure.

This package provides adapters for various AI providers, translating
between the Agent Interface Protocol (AIP) format and provider-specific APIs.

Supported Providers:
    - OpenAI (GPT-4, GPT-4 Turbo) — API
    - Anthropic (Claude 3.x) — API
    - Ollama (Local, FREE) — local server
    - Cline — local IDE agent (HTTP)
    - Google Gemini — API
    - Claude Code CLI — local CLI agent (`claude` binary)
    - OpenAI Codex CLI — local CLI agent (`codex` binary)
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
from src.shared.python.ai.adapters.claude_code_adapter import ClaudeCodeAdapter
from src.shared.python.ai.adapters.cline_adapter import ClineAdapter
from src.shared.python.ai.adapters.codex_cli_adapter import CodexCliAdapter
from src.shared.python.ai.adapters.factory import AdapterFactory
from src.shared.python.ai.adapters.gemini_cli_adapter import GeminiCliAdapter
from src.shared.python.ai.adapters.ollama_adapter import OllamaAdapter
from src.shared.python.ai.adapters.openai_adapter import OpenAIAdapter

__all__ = [
    "AdapterFactory",
    "AnthropicAdapter",
    "BaseAgentAdapter",
    "ClaudeCodeAdapter",
    "ClineAdapter",
    "CodexCliAdapter",
    "GeminiCliAdapter",
    "OllamaAdapter",
    "OpenAIAdapter",
    "ToolDeclaration",
]
