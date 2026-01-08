# Golf Modeling Suite: AI Assistant Architecture Specification

**Version:** 1.0.0  
**Date:** 2026-01-08  
**Status:** Technical Specification  
**Parent Document:** AI_IMPLEMENTATION_MASTER_PLAN.md

---

## 1. Overview

This document provides the detailed technical architecture for the AI Assistant integration layer. It extends the Project Design Guidelines with Section T requirements and serves as the implementation reference for all AI-related code.

---

## 2. Core Design Principles

### 2.1 Agent-Agnostic Architecture

The Golf Modeling Suite's AI integration is designed to work with **ANY** LLM provider without vendor lock-in:

```
┌──────────────────────────────────────────────────────────────┐
│                    Golf Modeling Suite                        │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌────────────────────────────────────────────────────────┐  │
│  │              Agent Interface Protocol (AIP)            │  │
│  │                                                        │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │  │
│  │  │ Tool Registry│  │   Workflow   │  │  Education   │  │  │
│  │  │              │  │    Engine    │  │    System    │  │  │
│  │  └──────────────┘  └──────────────┘  └──────────────┘  │  │
│  │                                                        │  │
│  │  ┌────────────────────────────────────────────────────┐│  │
│  │  │            Scientific Validator                    ││  │
│  │  └────────────────────────────────────────────────────┘│  │
│  └────────────────────────────────────────────────────────┘  │
│                            │                                  │
│                            ▼                                  │
│  ┌────────────────────────────────────────────────────────┐  │
│  │              Provider Adapter Layer                    │  │
│  │                                                        │  │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────────┐  │  │
│  │  │ OpenAI  │ │Anthropic│ │ Ollama  │ │   Custom    │  │  │
│  │  │ Adapter │ │ Adapter │ │ Adapter │ │   Adapter   │  │  │
│  │  └─────────┘ └─────────┘ └─────────┘ └─────────────┘  │  │
│  └────────────────────────────────────────────────────────┘  │
│                            │                                  │
└────────────────────────────┼──────────────────────────────────┘
                             │
                             ▼
          User's Chosen LLM Provider (User Pays)
```

### 2.2 Zero Developer Cost Model

The architecture ensures developers pay nothing:

| Component      | Cost Bearer     | Rationale                        |
| -------------- | --------------- | -------------------------------- |
| AI Compute     | User            | User's API key → User's provider |
| Infrastructure | None            | No backend servers required      |
| Local LLM      | User (optional) | Ollama runs on user's machine    |
| Data Storage   | User            | All data stays local             |

### 2.3 Privacy-First Design

All sensitive data handling follows these rules:

1. **API Keys**: Stored in OS keyring (Windows Credential Manager / macOS Keychain / Linux Secret Service)
2. **Conversation History**: Local only, never transmitted to developers
3. **Model Data**: Never sent to AI providers beyond current analysis context
4. **Audit Logging**: Local-only logs for troubleshooting

---

## 3. Component Specifications

### 3.1 Agent Interface Protocol (AIP) Server

**Location:** `shared/python/ai/aip_server.py`

```python
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

logger = logging.getLogger(__name__)


@dataclass
class AIPConfig:
    """Configuration for the Agent Interface Protocol server.

    Attributes:
        max_context_tokens: Maximum tokens for conversation context [tokens].
        response_timeout: Timeout for provider responses [seconds].
        enable_streaming: Whether to stream responses incrementally.
        scientific_validation: Enable post-response physics checks.
    """
    max_context_tokens: int = 8192
    response_timeout: float = 60.0
    enable_streaming: bool = True
    scientific_validation: bool = True
    default_expertise_level: ExpertiseLevel = ExpertiseLevel.BEGINNER


@dataclass
class ConversationContext:
    """Manages conversation state for multi-turn interactions.

    Attributes:
        messages: List of previous messages in conversation.
        active_workflow: Currently executing workflow (if any).
        user_expertise: User's current expertise level.
        session_id: Unique identifier for this session.
    """
    messages: list[Message] = field(default_factory=list)
    active_workflow: WorkflowState | None = None
    user_expertise: ExpertiseLevel = ExpertiseLevel.BEGINNER
    session_id: str = ""

    def add_message(self, role: str, content: str) -> None:
        """Add a message to the conversation history.

        Args:
            role: 'user', 'assistant', or 'system'
            content: The message content
        """
        self.messages.append(Message(role=role, content=content))
        self._truncate_if_needed()


class AgentInterfaceProtocol:
    """JSON-RPC 2.0 server for AI agent communication.

    This is the central orchestrator that:
    1. Receives user requests
    2. Translates to provider-specific format
    3. Validates AI responses scientifically
    4. Returns results with educational context

    Example:
        >>> aip = AgentInterfaceProtocol(config, registry, engine, education)
        >>> response = aip.process_message("Help me analyze this C3D file")
        >>> print(response.content)
    """

    def __init__(
        self,
        config: AIPConfig,
        tool_registry: ToolRegistry,
        workflow_engine: WorkflowEngine,
        education_system: EducationSystem,
    ) -> None:
        self._config = config
        self._tool_registry = tool_registry
        self._workflow_engine = workflow_engine
        self._education_system = education_system
        self._validator = ScientificValidator()
        self._active_adapter: BaseAgentAdapter | None = None

    def set_adapter(self, adapter: BaseAgentAdapter) -> None:
        """Configure the AI provider adapter.

        Args:
            adapter: Configured provider adapter instance
        """
        self._active_adapter = adapter
        logger.info(
            "AI provider set to %s",
            type(adapter).__name__
        )

    def process_message(
        self,
        message: str,
        context: ConversationContext,
    ) -> AgentResponse:
        """Process a user message and return AI response.

        Args:
            message: User's input message
            context: Current conversation context

        Returns:
            AI response with optional tool calls and educational content

        Raises:
            AIProviderError: If the AI provider is unavailable
            ScientificValidationError: If AI output fails physics checks
        """
        if self._active_adapter is None:
            raise AIProviderError("No AI provider configured")

        # Get relevant tools for this context
        available_tools = self._tool_registry.get_tools_for_context(
            context,
            self._active_adapter.capabilities
        )

        # Send to provider
        raw_response = self._active_adapter.send_message(
            message, context, available_tools
        )

        # Validate any tool calls scientifically
        if raw_response.tool_calls:
            for tool_call in raw_response.tool_calls:
                self._validate_tool_call(tool_call)

        # Add educational context if appropriate
        enriched_response = self._enrich_with_education(
            raw_response, context.user_expertise
        )

        return enriched_response
```

### 3.2 Provider Adapters

**Location:** `shared/python/ai/adapters/`

Each adapter translates between AIP format and provider-specific API:

#### 3.2.1 Base Adapter Protocol

```python
# shared/python/ai/adapters/base.py

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Iterator


class ProviderCapability(Enum):
    """Capabilities that may vary by provider."""
    FUNCTION_CALLING = auto()
    STREAMING = auto()
    VISION = auto()
    CODE_EXECUTION = auto()
    LONG_CONTEXT = auto()  # >32k tokens


@dataclass
class ProviderCapabilities:
    """Describes a provider's capabilities.

    Attributes:
        supported: Set of supported capabilities
        max_tokens: Maximum context window [tokens]
        model_name: Specific model identifier
    """
    supported: set[ProviderCapability]
    max_tokens: int
    model_name: str


class BaseAgentAdapter(Protocol):
    """Protocol for all AI provider adapters.

    Implementations must handle:
    - Message formatting for their provider
    - Tool call translation
    - Error handling and retries
    - Rate limiting compliance
    """

    @abstractmethod
    def send_message(
        self,
        message: str,
        context: ConversationContext,
        tools: list[ToolDeclaration],
    ) -> AgentResponse:
        """Send a message to the AI provider.

        Args:
            message: User message to process
            context: Conversation context
            tools: Available tools for this request

        Returns:
            Provider response translated to standard format
        """
        ...

    @abstractmethod
    def stream_response(
        self,
        message: str,
        context: ConversationContext,
        tools: list[ToolDeclaration],
    ) -> Iterator[AgentChunk]:
        """Stream response chunks from the AI provider.

        Args:
            message: User message to process
            context: Conversation context
            tools: Available tools

        Yields:
            Response chunks as they arrive
        """
        ...

    @property
    @abstractmethod
    def capabilities(self) -> ProviderCapabilities:
        """Return provider capabilities."""
        ...

    @abstractmethod
    def validate_connection(self) -> tuple[bool, str]:
        """Test connection to the provider.

        Returns:
            Tuple of (success, diagnostic_message)
        """
        ...
```

#### 3.2.2 Ollama Adapter (Free Local)

```python
# shared/python/ai/adapters/ollama_adapter.py

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import httpx

from .base import BaseAgentAdapter, ProviderCapabilities, ProviderCapability

if TYPE_CHECKING:
    from collections.abc import Iterator

logger = logging.getLogger(__name__)

# Ollama endpoint configuration
OLLAMA_DEFAULT_HOST = "http://localhost:11434"
OLLAMA_DEFAULT_MODEL = "llama3.1:8b"


class OllamaAdapter(BaseAgentAdapter):
    """Adapter for local Ollama LLM inference.

    This adapter enables FREE, 100% local AI assistance with no
    API keys or external services required.

    Requirements:
        - Ollama installed (https://ollama.ai)
        - Recommended model: llama3.1:8b (or larger)
        - Minimum RAM: 8GB (16GB+ recommended)

    Example:
        >>> adapter = OllamaAdapter(model="llama3.1:8b")
        >>> if adapter.validate_connection()[0]:
        ...     response = adapter.send_message("Hello", context, tools)
    """

    def __init__(
        self,
        host: str = OLLAMA_DEFAULT_HOST,
        model: str = OLLAMA_DEFAULT_MODEL,
        timeout: float = 120.0,
    ) -> None:
        """Initialize Ollama adapter.

        Args:
            host: Ollama server URL [URL]
            model: Model name to use
            timeout: Request timeout [seconds]
        """
        self._host = host
        self._model = model
        self._timeout = timeout
        self._client = httpx.Client(timeout=timeout)

    def send_message(
        self,
        message: str,
        context: ConversationContext,
        tools: list[ToolDeclaration],
    ) -> AgentResponse:
        """Send message to local Ollama instance."""
        # Format conversation for Ollama
        messages = self._format_messages(context, message)

        # Include tool descriptions in system prompt
        system_prompt = self._build_system_prompt(tools)

        response = self._client.post(
            f"{self._host}/api/chat",
            json={
                "model": self._model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    *messages,
                ],
                "stream": False,
            },
        )
        response.raise_for_status()

        return self._parse_response(response.json())

    @property
    def capabilities(self) -> ProviderCapabilities:
        """Return Ollama capabilities."""
        return ProviderCapabilities(
            supported={
                ProviderCapability.STREAMING,
                # Function calling support varies by model
            },
            max_tokens=8192,  # Varies by model
            model_name=self._model,
        )

    def validate_connection(self) -> tuple[bool, str]:
        """Test connection to local Ollama."""
        try:
            response = self._client.get(f"{self._host}/api/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m["name"] for m in models]
                if self._model in model_names:
                    return True, f"Connected to Ollama with {self._model}"
                return False, (
                    f"Model {self._model} not found. "
                    f"Available: {model_names}"
                )
            return False, f"Ollama returned status {response.status_code}"
        except httpx.ConnectError:
            return False, (
                "Cannot connect to Ollama. "
                "Is it running? (ollama serve)"
            )
```

### 3.3 Tool Registry

**Location:** `shared/python/ai/tool_registry.py`

```python
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger(__name__)


class ExpertiseLevel(Enum):
    """User expertise levels for progressive disclosure."""
    BEGINNER = 1      # No prior knowledge
    INTERMEDIATE = 2  # Basic physics/biomechanics
    ADVANCED = 3      # Graduate-level
    EXPERT = 4        # Publication-ready


@dataclass
class ParameterSpec:
    """Specification for a tool parameter.

    Attributes:
        name: Parameter name
        type: Python type annotation
        description: Human-readable description
        required: Whether parameter is required
        default: Default value if not required
        units: Physical units [SI notation]
        constraints: Validation constraints
    """
    name: str
    type: str
    description: str
    required: bool = True
    default: Any = None
    units: str | None = None
    constraints: dict[str, Any] = field(default_factory=dict)


@dataclass
class QualityCheck:
    """Post-execution quality validation.

    Attributes:
        name: Check name
        check_fn: Function that validates output
        failure_message: Message if check fails
        severity: 'error' or 'warning'
    """
    name: str
    check_fn: Callable[[Any], bool]
    failure_message: str
    severity: str = "error"


@dataclass
class ToolDeclaration:
    """Self-describing tool for AI consumption.

    Each tool exposes Golf Suite functionality to the AI in a
    standardized, discoverable format.

    Attributes:
        name: Unique tool identifier
        description: What the tool does (AI-consumable)
        parameters: Input parameters
        returns: Return value specification
        educational_link: Link to concept explanation
        quality_checks: Post-execution validations
        expertise_level: Minimum user level for this tool
        category: Organizational category
    """
    name: str
    description: str
    parameters: list[ParameterSpec]
    returns: ParameterSpec
    educational_link: str | None = None
    quality_checks: list[QualityCheck] = field(default_factory=list)
    expertise_level: ExpertiseLevel = ExpertiseLevel.BEGINNER
    category: str = "general"


class ToolRegistry:
    """Central registry of all AI-accessible tools.

    The registry provides:
    1. Tool discovery by category and expertise level
    2. Parameter validation
    3. Post-execution quality checks
    4. Educational content linking

    Example:
        >>> registry = ToolRegistry()
        >>> registry.register(load_c3d_tool)
        >>> tools = registry.get_tools_for_context(context, capabilities)
    """

    # Tool categories
    CATEGORIES = {
        "modeling": "URDF generation, model loading, configuration",
        "simulation": "Forward/inverse dynamics, integration",
        "analysis": "IAA, drift-control, ellipsoids, power flow",
        "visualization": "Plots, 3D views, exports",
        "validation": "Cross-engine checks, scientific tests",
        "data": "C3D loading, export, data management",
    }

    def __init__(self) -> None:
        self._tools: dict[str, ToolDeclaration] = {}
        self._load_builtin_tools()

    def register(self, tool: ToolDeclaration) -> None:
        """Register a tool in the registry.

        Args:
            tool: Tool declaration to register

        Raises:
            ValueError: If tool name already registered
        """
        if tool.name in self._tools:
            raise ValueError(f"Tool {tool.name} already registered")
        self._tools[tool.name] = tool
        logger.info("Registered tool: %s", tool.name)

    def get_tools_for_context(
        self,
        context: ConversationContext,
        capabilities: ProviderCapabilities,
    ) -> list[ToolDeclaration]:
        """Get tools appropriate for current context.

        Filters tools based on:
        - User expertise level
        - Provider capabilities
        - Active workflow requirements

        Args:
            context: Current conversation context
            capabilities: Provider's capabilities

        Returns:
            List of available tools
        """
        available: list[ToolDeclaration] = []

        for tool in self._tools.values():
            # Filter by expertise level
            if tool.expertise_level.value > context.user_expertise.value:
                continue

            # Add tool if it passes filters
            available.append(tool)

        return available

    def _load_builtin_tools(self) -> None:
        """Load all built-in Golf Suite tools."""
        # These will be populated from actual implementations
        self._register_modeling_tools()
        self._register_simulation_tools()
        self._register_analysis_tools()
        self._register_visualization_tools()
        self._register_validation_tools()
        self._register_data_tools()
```

### 3.4 Workflow Engine

**Location:** `shared/python/ai/workflow_engine.py`

```python
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)


class WorkflowStatus(Enum):
    """Current status of a workflow execution."""
    NOT_STARTED = auto()
    IN_PROGRESS = auto()
    AWAITING_INPUT = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()


class FailureStrategy(Enum):
    """How to handle step failures."""
    ABORT = auto()           # Stop workflow immediately
    RETRY = auto()           # Retry this step
    SKIP = auto()            # Skip to next step
    ASK_USER = auto()        # Ask user how to proceed
    FALLBACK = auto()        # Use fallback action


@dataclass
class ValidationCheck:
    """Validation to run after a workflow step.

    Attributes:
        name: Check identifier
        check_fn: Function returning (passed, message)
        required: If True, failure blocks progression
    """
    name: str
    check_fn: Callable[[Any], tuple[bool, str]]
    required: bool = True


@dataclass
class WorkflowStep:
    """Single step in a guided workflow.

    Attributes:
        step_id: Unique step identifier
        name: Human-readable step name
        description: What this step accomplishes
        tool_calls: Tools to execute in this step
        validation: Checks to run after execution
        on_failure: How to handle failures
        educational_content: Learning content for this step
        requires_user_input: Whether step needs user action
    """
    step_id: str
    name: str
    description: str
    tool_calls: list[str] = field(default_factory=list)
    validation: list[ValidationCheck] = field(default_factory=list)
    on_failure: FailureStrategy = FailureStrategy.ASK_USER
    educational_content: str | None = None
    requires_user_input: bool = False


@dataclass
class WorkflowState:
    """Current state of an executing workflow.

    Attributes:
        workflow_id: Which workflow is running
        current_step_index: Index of current step
        step_results: Results from completed steps
        status: Current workflow status
        error_message: Error if status is FAILED
    """
    workflow_id: str
    current_step_index: int = 0
    step_results: dict[str, Any] = field(default_factory=dict)
    status: WorkflowStatus = WorkflowStatus.NOT_STARTED
    error_message: str | None = None


class WorkflowEngine:
    """Orchestrates multi-step analysis workflows.

    The workflow engine provides:
    1. Step-by-step guidance through complex analyses
    2. Validation at each step
    3. Error recovery with multiple strategies
    4. Educational content interwoven with execution

    Built-in Workflows:
        first_analysis: Complete beginner walkthrough
        c3d_import: Load and validate C3D files
        inverse_dynamics: Compute joint torques
        cross_engine_validation: Compare engines
        drift_control_decomposition: Causal analysis

    Example:
        >>> engine = WorkflowEngine(tool_registry)
        >>> state = engine.start_workflow("first_analysis")
        >>> while not engine.is_complete(state):
        ...     state = engine.advance(state, user_input)
    """

    def __init__(self, tool_registry: ToolRegistry) -> None:
        self._tool_registry = tool_registry
        self._workflows: dict[str, list[WorkflowStep]] = {}
        self._load_builtin_workflows()

    def start_workflow(
        self,
        workflow_id: str,
        initial_context: dict[str, Any] | None = None,
    ) -> WorkflowState:
        """Start a new workflow execution.

        Args:
            workflow_id: Identifier of workflow to start
            initial_context: Optional initial parameters

        Returns:
            Initial workflow state

        Raises:
            ValueError: If workflow_id not found
        """
        if workflow_id not in self._workflows:
            available = list(self._workflows.keys())
            raise ValueError(
                f"Unknown workflow: {workflow_id}. "
                f"Available: {available}"
            )

        state = WorkflowState(
            workflow_id=workflow_id,
            status=WorkflowStatus.IN_PROGRESS,
        )

        if initial_context:
            state.step_results["_initial"] = initial_context

        logger.info("Started workflow: %s", workflow_id)
        return state

    def get_current_step(self, state: WorkflowState) -> WorkflowStep:
        """Get the current step for a workflow state.

        Args:
            state: Current workflow state

        Returns:
            Current step to execute
        """
        steps = self._workflows[state.workflow_id]
        return steps[state.current_step_index]

    def advance(
        self,
        state: WorkflowState,
        step_result: Any = None,
    ) -> WorkflowState:
        """Advance workflow to next step.

        Args:
            state: Current workflow state
            step_result: Result from current step (if any)

        Returns:
            Updated workflow state
        """
        current_step = self.get_current_step(state)

        # Store result
        if step_result is not None:
            state.step_results[current_step.step_id] = step_result

        # Run validations
        for check in current_step.validation:
            passed, message = check.check_fn(step_result)
            if not passed and check.required:
                return self._handle_failure(state, current_step, message)

        # Advance to next step
        steps = self._workflows[state.workflow_id]
        if state.current_step_index + 1 >= len(steps):
            state.status = WorkflowStatus.COMPLETED
            logger.info("Workflow completed: %s", state.workflow_id)
        else:
            state.current_step_index += 1

        return state

    def is_complete(self, state: WorkflowState) -> bool:
        """Check if workflow has finished.

        Args:
            state: Workflow state to check

        Returns:
            True if workflow completed or failed
        """
        return state.status in {
            WorkflowStatus.COMPLETED,
            WorkflowStatus.FAILED,
            WorkflowStatus.CANCELLED,
        }

    def _load_builtin_workflows(self) -> None:
        """Load all built-in workflow definitions."""
        self._workflows["first_analysis"] = self._create_first_analysis()
        self._workflows["c3d_import"] = self._create_c3d_import()
        self._workflows["inverse_dynamics"] = self._create_inverse_dynamics()
        self._workflows["cross_engine_validation"] = (
            self._create_cross_engine_validation()
        )
        self._workflows["drift_control_decomposition"] = (
            self._create_drift_control()
        )

    def _create_first_analysis(self) -> list[WorkflowStep]:
        """Create the beginner-friendly first analysis workflow."""
        return [
            WorkflowStep(
                step_id="welcome",
                name="Welcome to Golf Modeling Suite",
                description=(
                    "This workflow will guide you through your first "
                    "biomechanical analysis of a golf swing."
                ),
                educational_content="concepts/introduction",
            ),
            WorkflowStep(
                step_id="load_data",
                name="Load Motion Capture Data",
                description="Import your C3D file containing marker data",
                tool_calls=["load_c3d"],
                requires_user_input=True,
                educational_content="concepts/c3d_format",
            ),
            WorkflowStep(
                step_id="select_engine",
                name="Select Physics Engine",
                description="Choose which physics engine to use",
                tool_calls=["list_available_engines", "select_engine"],
                educational_content="concepts/physics_engines",
            ),
            WorkflowStep(
                step_id="run_inverse_dynamics",
                name="Compute Joint Torques",
                description="Calculate the forces that produced this motion",
                tool_calls=["run_inverse_dynamics"],
                validation=[
                    ValidationCheck(
                        name="energy_conservation",
                        check_fn=lambda r: (
                            r.get("energy_drift", 1.0) < 0.01,
                            f"Energy drift: {r.get('energy_drift', 'N/A')}"
                        ),
                    ),
                ],
                educational_content="concepts/inverse_dynamics",
            ),
            WorkflowStep(
                step_id="visualize",
                name="Visualize Results",
                description="Display joint torques and power flow",
                tool_calls=["plot_joint_torques", "plot_power_flow"],
                educational_content="concepts/interpretation",
            ),
            WorkflowStep(
                step_id="export",
                name="Export Analysis",
                description="Save your results for future reference",
                tool_calls=["export_analysis_bundle"],
                requires_user_input=True,
            ),
        ]
```

### 3.5 Scientific Validator

**Location:** `shared/python/ai/scientific_validator.py`

```python
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# Validation tolerances from Project Design Guidelines Section P3
TOLERANCE_POSITION_M = 1e-6     # [m]
TOLERANCE_VELOCITY_MS = 1e-5   # [m/s]
TOLERANCE_ACCEL_MS2 = 1e-4     # [m/s²]
TOLERANCE_TORQUE_NM = 1e-3     # [N·m]
TOLERANCE_ENERGY_FRACTION = 0.01  # [dimensionless, 1%]


@dataclass
class ValidationResult:
    """Result of a scientific validation check.

    Attributes:
        passed: Whether validation passed
        check_name: Name of the validation
        message: Human-readable result
        value: Computed value (if applicable)
        threshold: Threshold used (if applicable)
    """
    passed: bool
    check_name: str
    message: str
    value: float | None = None
    threshold: float | None = None


class ScientificValidator:
    """Validates AI-generated physics outputs.

    This validator ensures that AI responses do not violate
    fundamental physics constraints:

    1. Conservation laws (energy, momentum)
    2. Cross-engine consistency
    3. Numerical stability
    4. Physical plausibility

    Example:
        >>> validator = ScientificValidator()
        >>> result = validator.validate_inverse_dynamics(torques, motion)
        >>> if not result.passed:
        ...     raise ScientificValidationError(result.message)
    """

    def validate_inverse_dynamics(
        self,
        torques: Any,
        motion: Any,
        engine_name: str,
    ) -> list[ValidationResult]:
        """Validate inverse dynamics computation.

        Checks:
        1. Torques are physically plausible
        2. Results are reproducible
        3. Energy balance is maintained

        Args:
            torques: Computed joint torques
            motion: Input motion data
            engine_name: Which engine produced results

        Returns:
            List of validation results
        """
        results: list[ValidationResult] = []

        # Check 1: Torque magnitudes are plausible
        max_torque = self._get_max_torque(torques)
        # Human joint torque limits (approximate)
        torque_limit = 500.0  # [N·m] - high for shoulder joint

        results.append(
            ValidationResult(
                passed=max_torque < torque_limit,
                check_name="torque_magnitude",
                message=(
                    f"Maximum torque: {max_torque:.1f} N·m "
                    f"(limit: {torque_limit} N·m)"
                ),
                value=max_torque,
                threshold=torque_limit,
            )
        )

        # Check 2: Energy conservation
        energy_drift = self._compute_energy_drift(torques, motion)

        results.append(
            ValidationResult(
                passed=energy_drift < TOLERANCE_ENERGY_FRACTION,
                check_name="energy_conservation",
                message=(
                    f"Energy drift: {energy_drift*100:.2f}% "
                    f"(limit: {TOLERANCE_ENERGY_FRACTION*100}%)"
                ),
                value=energy_drift,
                threshold=TOLERANCE_ENERGY_FRACTION,
            )
        )

        return results

    def validate_cross_engine(
        self,
        result_a: Any,
        result_b: Any,
        engine_a: str,
        engine_b: str,
    ) -> list[ValidationResult]:
        """Validate cross-engine consistency.

        Per Section P3 tolerances:
        - Positions: ± 1e-6 m
        - Velocities: ± 1e-5 m/s
        - Accelerations: ± 1e-4 m/s²
        - Torques: ± 1e-3 N·m

        Args:
            result_a: Results from first engine
            result_b: Results from second engine
            engine_a: First engine name
            engine_b: Second engine name

        Returns:
            List of validation results
        """
        results: list[ValidationResult] = []

        # Compare positions
        pos_diff = self._compute_max_difference(
            result_a.get("positions"),
            result_b.get("positions"),
        )
        results.append(
            ValidationResult(
                passed=pos_diff < TOLERANCE_POSITION_M,
                check_name="position_agreement",
                message=(
                    f"{engine_a} vs {engine_b} position diff: "
                    f"{pos_diff*1000:.3f} mm (limit: 0.001 mm)"
                ),
                value=pos_diff,
                threshold=TOLERANCE_POSITION_M,
            )
        )

        # Compare torques
        torque_diff = self._compute_max_difference(
            result_a.get("torques"),
            result_b.get("torques"),
        )
        results.append(
            ValidationResult(
                passed=torque_diff < TOLERANCE_TORQUE_NM,
                check_name="torque_agreement",
                message=(
                    f"{engine_a} vs {engine_b} torque diff: "
                    f"{torque_diff:.4f} N·m (limit: 0.001 N·m)"
                ),
                value=torque_diff,
                threshold=TOLERANCE_TORQUE_NM,
            )
        )

        return results
```

---

## 4. GUI Integration Specification

### 4.1 AI Assistant Panel Layout

```
┌─────────────────────────────────────────────────────────────────────┐
│ Golf Modeling Suite - AI Assistant                              [×] │
├─────────────────────┬───────────────────────────────────────────────┤
│                     │                                               │
│  Conversation       │  AI Response Panel                            │
│  ─────────────      │                                               │
│                     │  Welcome! I'm your AI assistant for the       │
│  [User] Help me     │  Golf Modeling Suite. I can help you:         │
│  analyze this C3D   │                                               │
│  file               │  • Load and analyze C3D motion capture        │
│                     │  • Run biomechanical simulations              │
│  [AI] I'll help     │  • Interpret results using scientific         │
│  you through the    │    analysis techniques                        │
│  analysis...        │  • Compare results across physics engines     │
│                     │                                               │
│  [User] What does   │  ℹ️ Would you like to start with a guided     │
│  "drift" mean?      │  workflow, or do you have a specific task?    │
│                     │                                               │
│  [AI] Great         │  📚 Learn More: [Drift-Control Decomposition] │
│  question! Drift... │                                               │
│                     ├───────────────────────────────────────────────┤
│                     │  Active Workflow: First Analysis (Step 2/6)   │
│                     │  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  │
│                     │  ☑ Welcome  ● Load Data  ○ Select Engine ...  │
│                     │                                               │
├─────────────────────┼───────────────────────────────────────────────┤
│                     │  Tool Execution                               │
│  Quick Actions      │  ─────────────                                │
│  ─────────────      │                                               │
│  [📁 Load C3D]      │  ▶ load_c3d("golf_swing_001.c3d")            │
│  [🔬 Run Analysis]  │    Status: Completed ✓                        │
│  [📊 Visualize]     │    Markers: 42, Frames: 1250                  │
│  [🔄 Compare]       │    Sample Rate: 250 Hz                        │
│                     │                                               │
├─────────────────────┴───────────────────────────────────────────────┤
│ Provider: OpenAI (GPT-4) ● Connected | Expertise: Beginner [▲]     │
└─────────────────────────────────────────────────────────────────────┘
```

### 4.2 Widget Hierarchy

```python
# Location: tools/launcher/ai_assistant_panel.py

class AIAssistantPanel(QWidget):
    """Main AI Assistant panel for the Golf Suite Launcher.

    Components:
    - ConversationWidget: Chat history and input
    - ResponsePanel: AI response display with educational links
    - WorkflowTracker: Visual progress for active workflows
    - ToolVisualizer: Displays executing tools and results
    - QuickActionsBar: Common action shortcuts
    - StatusBar: Provider status and expertise level
    """

    # Signals
    message_sent = pyqtSignal(str)
    workflow_started = pyqtSignal(str)
    tool_executed = pyqtSignal(str, object)

    def __init__(
        self,
        aip_server: AgentInterfaceProtocol,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._aip = aip_server
        self._setup_ui()
        self._connect_signals()
```

---

## 5. Security Specification

### 5.1 API Key Storage

```python
# Location: shared/python/ai/keyring_storage.py

import keyring
import logging

logger = logging.getLogger(__name__)

# Keyring service identifier
SERVICE_NAME = "golf-modeling-suite"


class SecureKeyStorage:
    """Secure storage for API keys using OS keyring.

    Uses:
    - Windows: Windows Credential Manager
    - macOS: Keychain
    - Linux: Secret Service API (GNOME Keyring, KWallet)

    API keys are NEVER:
    - Stored in plain text
    - Logged
    - Transmitted to developers
    - Included in error reports
    """

    def store_api_key(self, provider: str, api_key: str) -> None:
        """Securely store an API key.

        Args:
            provider: Provider identifier (e.g., 'openai')
            api_key: The API key to store
        """
        keyring.set_password(SERVICE_NAME, f"{provider}_api_key", api_key)
        logger.info("Stored API key for provider: %s", provider)

    def get_api_key(self, provider: str) -> str | None:
        """Retrieve an API key.

        Args:
            provider: Provider identifier

        Returns:
            API key or None if not found
        """
        return keyring.get_password(SERVICE_NAME, f"{provider}_api_key")

    def delete_api_key(self, provider: str) -> None:
        """Remove a stored API key.

        Args:
            provider: Provider identifier
        """
        try:
            keyring.delete_password(SERVICE_NAME, f"{provider}_api_key")
            logger.info("Deleted API key for provider: %s", provider)
        except keyring.errors.PasswordDeleteError:
            logger.warning("No API key found for provider: %s", provider)
```

### 5.2 Privacy Guarantees

| Data Type            | Storage           | Transmission            | Developer Access  |
| -------------------- | ----------------- | ----------------------- | ----------------- |
| API Keys             | OS Keyring        | To user's provider only | NEVER             |
| Conversation History | Local JSON        | NEVER                   | NEVER             |
| Model Files          | User's filesystem | NEVER                   | NEVER             |
| Analysis Results     | User's filesystem | NEVER                   | NEVER             |
| Error Reports        | Local logs only   | Optional (user consent) | With consent only |

---

## 6. Testing Requirements

### 6.1 Unit Tests

All AI modules require minimum 50% coverage:

```python
# tests/unit/test_aip_server.py

class TestAIPServer:
    """Tests for Agent Interface Protocol server."""

    def test_process_message_requires_adapter(self) -> None:
        """Test that processing fails without adapter."""
        ...

    def test_tool_filtering_by_expertise(self) -> None:
        """Test that tools are filtered by user expertise level."""
        ...

    def test_scientific_validation_blocks_invalid(self) -> None:
        """Test that invalid physics outputs are blocked."""
        ...
```

### 6.2 Integration Tests

```python
# tests/integration/test_ai_workflows.py

class TestBuiltinWorkflows:
    """Integration tests for built-in workflows."""

    @pytest.mark.integration
    def test_first_analysis_workflow_complete(self) -> None:
        """Test complete first analysis workflow."""
        ...

    @pytest.mark.integration
    def test_cross_engine_workflow_validates(self) -> None:
        """Test cross-engine validation workflow."""
        ...
```

---

## Document Control

| Version | Date       | Author                 | Changes               |
| ------- | ---------- | ---------------------- | --------------------- |
| 1.0.0   | 2026-01-08 | AI Implementation Team | Initial specification |

**Related Documents:**

- AI_IMPLEMENTATION_MASTER_PLAN.md (Parent)
- CRITICAL_EVALUATION.md (Assessment)
- project_design_guidelines.qmd (Section T addition)
