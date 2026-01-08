"""AI Assistant Integration Layer for Golf Modeling Suite.

This package provides an agent-agnostic AI assistant architecture that
enables natural language interaction, guided workflows, and educational
content delivery for users of all skill levels.

Architecture:
    AgentInterfaceProtocol (AIP) <- Provider Adapters <- User's LLM
                |
                v
    ToolRegistry + WorkflowEngine + EducationSystem
                |
                v
    Scientific Validator (enforces physics consistency)

Design Principles:
    1. Agent-Agnostic: Works with any LLM provider (OpenAI, Anthropic, Ollama)
    2. Zero Developer Cost: Users provide their own API keys
    3. Educational Focus: Teaches while executing
    4. Scientific Integrity: AI never bypasses validation
    5. Privacy-First: API keys in OS keyring, no data to developers

Example:
    >>> from shared.python.ai import AgentInterfaceProtocol, OllamaAdapter
    >>> adapter = OllamaAdapter()  # Free local AI
    >>> aip = AgentInterfaceProtocol.create_with_adapter(adapter)
    >>> response = aip.process_message("Help me analyze this C3D file")
"""

from shared.python.ai.exceptions import (
    AIConnectionError,
    AIError,
    AIProviderError,
    AIRateLimitError,
    AITimeoutError,
    ScientificValidationError,
    ToolExecutionError,
    WorkflowError,
)
from shared.python.ai.types import (
    AgentChunk,
    AgentResponse,
    ConversationContext,
    ExpertiseLevel,
    Message,
    ProviderCapabilities,
    ProviderCapability,
    ToolCall,
)

__all__ = [
    # Types
    "AgentChunk",
    "AgentResponse",
    "ConversationContext",
    "ExpertiseLevel",
    "Message",
    "ProviderCapabilities",
    "ProviderCapability",
    "ToolCall",
    # Exceptions
    "AIError",
    "AIProviderError",
    "AIConnectionError",
    "AIRateLimitError",
    "AITimeoutError",
    "ScientificValidationError",
    "WorkflowError",
    "ToolExecutionError",
]

__version__ = "0.1.0"
