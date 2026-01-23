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
    >>> from shared.python.ai import ToolRegistry, WorkflowEngine, EducationSystem
    >>> registry = ToolRegistry()
    >>> engine = WorkflowEngine(registry)
    >>> edu = EducationSystem()
"""

from src.shared.python.ai.education import EducationSystem, GlossaryEntry
from src.shared.python.ai.exceptions import (
    AIConnectionError,
    AIError,
    AIProviderError,
    AIRateLimitError,
    AITimeoutError,
    ScientificValidationError,
    ToolExecutionError,
    WorkflowError,
)
from src.shared.python.ai.tool_registry import (
    Tool,
    ToolCategory,
    ToolParameter,
    ToolRegistry,
    get_global_registry,
)
from src.shared.python.ai.types import (
    AgentChunk,
    AgentResponse,
    ConversationContext,
    ExpertiseLevel,
    Message,
    ProviderCapabilities,
    ProviderCapability,
    ToolCall,
    ToolResult,
)
from src.shared.python.ai.workflow_engine import (
    RecoveryStrategy,
    StepResult,
    StepStatus,
    ValidationResult,
    Workflow,
    WorkflowEngine,
    WorkflowExecution,
    WorkflowStep,
    create_first_analysis_workflow,
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
    "ToolResult",
    # Tool Registry
    "Tool",
    "ToolCategory",
    "ToolParameter",
    "ToolRegistry",
    "get_global_registry",
    # Workflow Engine
    "RecoveryStrategy",
    "StepResult",
    "StepStatus",
    "ValidationResult",
    "Workflow",
    "WorkflowEngine",
    "WorkflowExecution",
    "WorkflowStep",
    "create_first_analysis_workflow",
    # Education
    "EducationSystem",
    "GlossaryEntry",
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
