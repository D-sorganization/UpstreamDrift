"""Workflow Engine for guided AI-assisted analysis.

This module provides step-by-step workflow execution with validation,
error recovery, and educational content integration.

The workflow engine ensures that users can complete complex analyses
without expert knowledge, while learning along the way.

Example:
    >>> from shared.python.ai.workflow_engine import WorkflowEngine, Workflow
    >>> engine = WorkflowEngine(tool_registry)
    >>> result = engine.execute_workflow("first_analysis", context)
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum, auto
from typing import Any

from shared.python.ai.exceptions import WorkflowError
from shared.python.ai.tool_registry import ToolRegistry
from shared.python.ai.types import ConversationContext, ExpertiseLevel, ToolResult

logger = logging.getLogger(__name__)


class StepStatus(Enum):
    """Status of a workflow step."""

    PENDING = auto()  # Not yet started
    RUNNING = auto()  # Currently executing
    COMPLETED = auto()  # Successfully finished
    FAILED = auto()  # Failed with error
    SKIPPED = auto()  # Skipped by user or condition


class RecoveryStrategy(Enum):
    """How to handle step failures."""

    ABORT = auto()  # Stop workflow immediately
    RETRY = auto()  # Retry the step
    SKIP = auto()  # Skip and continue
    ASK_USER = auto()  # Ask user what to do
    FALLBACK = auto()  # Try alternate approach


@dataclass
class ValidationResult:
    """Result of a step validation check.

    Attributes:
        passed: Whether validation passed.
        message: Descriptive message.
        details: Additional validation details.
    """

    passed: bool
    message: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowStep:
    """A single step in a workflow.

    Attributes:
        id: Unique step identifier.
        name: Human-readable step name.
        description: What this step does.
        tool_name: Tool to execute (if tool-based step).
        tool_arguments: Arguments for the tool.
        validation: Optional validation function.
        on_failure: How to handle failures.
        educational_content: Learning content for this step.
        condition: Optional condition to check before running.
        timeout: Maximum execution time [s].
    """

    id: str
    name: str
    description: str
    tool_name: str | None = None
    tool_arguments: dict[str, Any] = field(default_factory=dict)
    validation: Callable[[Any], ValidationResult] | None = None
    on_failure: RecoveryStrategy = RecoveryStrategy.ASK_USER
    educational_content: dict[str, str] = field(default_factory=dict)
    condition: Callable[[dict[str, Any]], bool] | None = None
    timeout: float = 300.0  # 5 minutes default


@dataclass
class StepResult:
    """Result of executing a workflow step.

    Attributes:
        step_id: ID of the step that was executed.
        status: Final status of the step.
        result: Result data from execution.
        error: Error message if failed.
        validation: Validation result if performed.
        duration: How long the step took [s].
    """

    step_id: str
    status: StepStatus
    result: Any = None
    error: str | None = None
    validation: ValidationResult | None = None
    duration: float = 0.0


@dataclass
class Workflow:
    """A complete workflow definition.

    Attributes:
        id: Unique workflow identifier.
        name: Human-readable name.
        description: What this workflow accomplishes.
        steps: Ordered list of steps.
        expertise_level: Minimum expertise level.
        estimated_duration: Estimated completion time [minutes].
        tags: Searchable tags.
    """

    id: str
    name: str
    description: str
    steps: list[WorkflowStep] = field(default_factory=list)
    expertise_level: ExpertiseLevel = ExpertiseLevel.BEGINNER
    estimated_duration: int = 15  # minutes
    tags: list[str] = field(default_factory=list)

    def add_step(self, step: WorkflowStep) -> None:
        """Add a step to the workflow.

        Args:
            step: Step to add.
        """
        self.steps.append(step)


@dataclass
class WorkflowExecution:
    """Tracks state of a running workflow.

    Attributes:
        execution_id: Unique execution identifier.
        workflow_id: ID of the workflow being executed.
        context: Conversation context for this execution.
        state: Shared state between steps.
        step_results: Results of completed steps.
        current_step_index: Index of current step.
        started_at: When execution started.
        status: Overall execution status.
    """

    execution_id: str
    workflow_id: str
    context: ConversationContext
    state: dict[str, Any] = field(default_factory=dict)
    step_results: list[StepResult] = field(default_factory=list)
    current_step_index: int = 0
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    status: StepStatus = StepStatus.PENDING

    def get_current_step_result(self) -> StepResult | None:
        """Get the most recent step result.

        Returns:
            Most recent StepResult, or None if no steps completed.
        """
        if self.step_results:
            return self.step_results[-1]
        return None

    def get_step_result(self, step_id: str) -> StepResult | None:
        """Get result for a specific step.

        Args:
            step_id: ID of the step.

        Returns:
            StepResult if found, None otherwise.
        """
        for result in self.step_results:
            if result.step_id == step_id:
                return result
        return None


class WorkflowEngine:
    """Engine for executing guided workflows.

    The workflow engine orchestrates step-by-step execution,
    handling validation, error recovery, and state management.

    Example:
        >>> engine = WorkflowEngine(tool_registry)
        >>> engine.register_workflow(my_workflow)
        >>> execution = engine.start_workflow("my_workflow", context)
        >>> while not engine.is_complete(execution):
        ...     result = engine.execute_next_step(execution)
    """

    def __init__(self, tool_registry: ToolRegistry) -> None:
        """Initialize workflow engine.

        Args:
            tool_registry: Registry of available tools.
        """
        self._tool_registry = tool_registry
        self._workflows: dict[str, Workflow] = {}
        self._executions: dict[str, WorkflowExecution] = {}

        logger.info("Initialized WorkflowEngine")

    def register_workflow(self, workflow: Workflow) -> None:
        """Register a workflow.

        Args:
            workflow: Workflow to register.
        """
        self._workflows[workflow.id] = workflow
        logger.debug("Registered workflow: %s", workflow.id)

    def get_workflow(self, workflow_id: str) -> Workflow | None:
        """Get a workflow by ID.

        Args:
            workflow_id: ID of the workflow.

        Returns:
            Workflow if found, None otherwise.
        """
        return self._workflows.get(workflow_id)

    def list_workflows(
        self,
        max_expertise: ExpertiseLevel = ExpertiseLevel.EXPERT,
    ) -> list[Workflow]:
        """List available workflows.

        Args:
            max_expertise: Maximum expertise level to include.

        Returns:
            List of matching workflows.
        """
        workflows = list(self._workflows.values())
        workflows = [w for w in workflows if w.expertise_level <= max_expertise]
        return sorted(workflows, key=lambda w: w.name)

    def start_workflow(
        self,
        workflow_id: str,
        context: ConversationContext,
        initial_state: dict[str, Any] | None = None,
    ) -> WorkflowExecution:
        """Start executing a workflow.

        Args:
            workflow_id: ID of the workflow to start.
            context: Conversation context.
            initial_state: Initial state values.

        Returns:
            WorkflowExecution tracking the execution.

        Raises:
            WorkflowError: If workflow not found.
        """
        workflow = self.get_workflow(workflow_id)
        if workflow is None:
            raise WorkflowError(
                f"Workflow not found: {workflow_id}",
                workflow_id=workflow_id,
            )

        execution_id = f"exec_{uuid.uuid4().hex[:12]}"
        execution = WorkflowExecution(
            execution_id=execution_id,
            workflow_id=workflow_id,
            context=context,
            state=initial_state or {},
            status=StepStatus.RUNNING,
        )

        self._executions[execution_id] = execution
        logger.info("Started workflow: %s (execution: %s)", workflow_id, execution_id)

        return execution

    def get_execution(self, execution_id: str) -> WorkflowExecution | None:
        """Get an execution by ID.

        Args:
            execution_id: ID of the execution.

        Returns:
            WorkflowExecution if found, None otherwise.
        """
        return self._executions.get(execution_id)

    def is_complete(self, execution: WorkflowExecution) -> bool:
        """Check if a workflow execution is complete.

        Args:
            execution: Execution to check.

        Returns:
            True if complete (success or failure).
        """
        workflow = self.get_workflow(execution.workflow_id)
        if workflow is None:
            return True

        if execution.status in (StepStatus.COMPLETED, StepStatus.FAILED):
            return True

        return execution.current_step_index >= len(workflow.steps)

    def get_current_step(self, execution: WorkflowExecution) -> WorkflowStep | None:
        """Get the current step to execute.

        Args:
            execution: Current execution.

        Returns:
            Current WorkflowStep, or None if complete.
        """
        workflow = self.get_workflow(execution.workflow_id)
        if workflow is None:
            return None

        if execution.current_step_index >= len(workflow.steps):
            return None

        return workflow.steps[execution.current_step_index]

    def execute_next_step(self, execution: WorkflowExecution) -> StepResult:
        """Execute the next step in the workflow.

        Args:
            execution: Current execution.

        Returns:
            Result of the step execution.

        Raises:
            WorkflowError: If workflow is complete or not found.
        """
        import time

        workflow = self.get_workflow(execution.workflow_id)
        if workflow is None:
            raise WorkflowError(
                f"Workflow not found: {execution.workflow_id}",
                workflow_id=execution.workflow_id,
            )

        step = self.get_current_step(execution)
        if step is None:
            raise WorkflowError(
                "Workflow already complete",
                workflow_id=execution.workflow_id,
            )

        start_time = time.perf_counter()
        logger.info(
            "Executing step: %s (workflow: %s)",
            step.id,
            execution.workflow_id,
        )

        # Check condition
        if step.condition is not None:
            try:
                should_run = step.condition(execution.state)
                if not should_run:
                    result = StepResult(
                        step_id=step.id,
                        status=StepStatus.SKIPPED,
                        duration=time.perf_counter() - start_time,
                    )
                    execution.step_results.append(result)
                    execution.current_step_index += 1
                    return result
            except Exception as e:
                logger.warning("Condition check failed for step %s: %s", step.id, e)

        # Execute tool if specified
        tool_result: ToolResult | None = None
        if step.tool_name:
            try:
                # Merge state with explicit arguments
                arguments = {**execution.state, **step.tool_arguments}
                tool_result = self._tool_registry.execute(
                    step.tool_name,
                    arguments,
                )
            except Exception as e:
                result = StepResult(
                    step_id=step.id,
                    status=StepStatus.FAILED,
                    error=str(e),
                    duration=time.perf_counter() - start_time,
                )
                execution.step_results.append(result)
                return self._handle_failure(execution, step, result)

            if not tool_result.success:
                result = StepResult(
                    step_id=step.id,
                    status=StepStatus.FAILED,
                    error=tool_result.error,
                    duration=time.perf_counter() - start_time,
                )
                execution.step_results.append(result)
                return self._handle_failure(execution, step, result)

        # Run validation
        validation_result: ValidationResult | None = None
        if step.validation is not None:
            try:
                validation_result = step.validation(
                    tool_result.result if tool_result else execution.state
                )
                if not validation_result.passed:
                    result = StepResult(
                        step_id=step.id,
                        status=StepStatus.FAILED,
                        error=validation_result.message,
                        validation=validation_result,
                        duration=time.perf_counter() - start_time,
                    )
                    execution.step_results.append(result)
                    return self._handle_failure(execution, step, result)
            except Exception as e:
                logger.warning("Validation failed for step %s: %s", step.id, e)

        # Success!
        result = StepResult(
            step_id=step.id,
            status=StepStatus.COMPLETED,
            result=tool_result.result if tool_result else None,
            validation=validation_result,
            duration=time.perf_counter() - start_time,
        )
        execution.step_results.append(result)
        execution.current_step_index += 1

        # Check if workflow complete
        if execution.current_step_index >= len(workflow.steps):
            execution.status = StepStatus.COMPLETED
            logger.info("Workflow completed: %s", execution.workflow_id)

        return result

    def _handle_failure(
        self,
        execution: WorkflowExecution,
        step: WorkflowStep,
        result: StepResult,
    ) -> StepResult:
        """Handle a step failure based on recovery strategy.

        Args:
            execution: Current execution.
            step: Failed step.
            result: Failure result.

        Returns:
            StepResult (possibly modified based on recovery).
        """
        if step.on_failure == RecoveryStrategy.ABORT:
            execution.status = StepStatus.FAILED
            logger.error(
                "Workflow aborted at step %s: %s",
                step.id,
                result.error,
            )
        elif step.on_failure == RecoveryStrategy.SKIP:
            result.status = StepStatus.SKIPPED
            execution.current_step_index += 1
            logger.warning("Skipped failed step: %s", step.id)
        # For RETRY, ASK_USER, FALLBACK - return result for caller to handle

        return result

    def get_progress(self, execution: WorkflowExecution) -> dict[str, Any]:
        """Get progress information for an execution.

        Args:
            execution: Current execution.

        Returns:
            Progress information dictionary.
        """
        workflow = self.get_workflow(execution.workflow_id)
        if workflow is None:
            return {"error": "Workflow not found"}

        completed = sum(
            1 for r in execution.step_results if r.status == StepStatus.COMPLETED
        )
        total = len(workflow.steps)

        return {
            "execution_id": execution.execution_id,
            "workflow_id": execution.workflow_id,
            "workflow_name": workflow.name,
            "current_step": execution.current_step_index,
            "total_steps": total,
            "completed_steps": completed,
            "progress_percent": (completed / total * 100) if total > 0 else 0,
            "status": execution.status.name,
            "is_complete": self.is_complete(execution),
        }

    def get_step_educational_content(
        self,
        step: WorkflowStep,
        expertise_level: ExpertiseLevel = ExpertiseLevel.BEGINNER,
    ) -> str:
        """Get educational content for a step at the given level.

        Args:
            step: Step to get content for.
            expertise_level: User's expertise level.

        Returns:
            Educational content string.
        """
        level_key = expertise_level.name.lower()
        if level_key in step.educational_content:
            return step.educational_content[level_key]

        # Fall back to lower levels
        for level in ExpertiseLevel:
            if level <= expertise_level:
                key = level.name.lower()
                if key in step.educational_content:
                    return step.educational_content[key]

        return step.description

    def __len__(self) -> int:
        """Return number of registered workflows."""
        return len(self._workflows)


def create_first_analysis_workflow() -> Workflow:
    """Create the 'first_analysis' beginner workflow.

    This workflow guides beginners through their first
    complete analysis of a golf swing.

    Returns:
        Complete Workflow object.
    """
    workflow = Workflow(
        id="first_analysis",
        name="Your First Golf Swing Analysis",
        description=(
            "A step-by-step guide to analyzing your first golf swing. "
            "This workflow will help you load data, run simulations, "
            "and interpret results."
        ),
        expertise_level=ExpertiseLevel.BEGINNER,
        estimated_duration=30,
        tags=["beginner", "tutorial", "inverse-dynamics"],
    )

    # Step 1: Welcome
    workflow.add_step(
        WorkflowStep(
            id="welcome",
            name="Welcome",
            description="Introduction to golf swing analysis",
            educational_content={
                "beginner": (
                    "Welcome to the Golf Modeling Suite! 🏌️\n\n"
                    "This workflow will guide you through analyzing a golf swing "
                    "using physics simulations. By the end, you'll understand:\n"
                    "- How to load motion capture data\n"
                    "- What inverse dynamics tells us\n"
                    "- How to interpret joint torques"
                ),
                "intermediate": (
                    "This workflow covers inverse dynamics analysis using "
                    "your choice of physics engine (MuJoCo, Drake, or Pinocchio)."
                ),
            },
        )
    )

    # Step 2: Select file
    workflow.add_step(
        WorkflowStep(
            id="select_file",
            name="Select Motion Data",
            description="Choose a C3D file containing golf swing motion capture",
            tool_name="list_sample_files",
            educational_content={
                "beginner": (
                    "C3D files contain motion capture data - 3D positions of "
                    "markers placed on the body during a golf swing.\n\n"
                    "Think of it like a detailed recording of exactly how "
                    "the body moved during the swing."
                ),
            },
        )
    )

    # Step 3: Load data
    workflow.add_step(
        WorkflowStep(
            id="load_data",
            name="Load Motion Data",
            description="Load and validate the C3D file",
            tool_name="load_c3d",
            educational_content={
                "beginner": (
                    "When we load the C3D file, we're extracting:\n"
                    "- Marker positions over time\n"
                    "- Frame rate (usually 100-500 Hz)\n"
                    "- Duration of the recording\n\n"
                    "The system will also check that the data is valid."
                ),
            },
        )
    )

    # Step 4: Run simulation
    workflow.add_step(
        WorkflowStep(
            id="run_simulation",
            name="Run Physics Simulation",
            description="Compute joint torques using inverse dynamics",
            tool_name="run_inverse_dynamics",
            timeout=600.0,  # 10 minutes for simulation
            educational_content={
                "beginner": (
                    "Inverse dynamics answers the question:\n"
                    "'What forces caused this motion?'\n\n"
                    "By knowing how the body moved, we can calculate the "
                    "torques (rotational forces) at each joint.\n\n"
                    "This helps us understand muscle contribution and "
                    "potential injury risks."
                ),
                "advanced": (
                    "Using the equation τ = M(q)q̈ + C(q,q̇) + g(q), "
                    "we solve for joint torques given the measured motion."
                ),
            },
        )
    )

    # Step 5: Interpret results
    workflow.add_step(
        WorkflowStep(
            id="interpret_results",
            name="Interpret Results",
            description="Understand what the torque data means",
            tool_name="interpret_torques",
            educational_content={
                "beginner": (
                    "The results show torques at each joint:\n"
                    "- Shoulder: typically 50-150 N·m during downswing\n"
                    "- Hip: often the highest torques (100-200 N·m)\n"
                    "- Wrist: lower but important for club control\n\n"
                    "Higher torques mean more muscle effort at that joint."
                ),
            },
        )
    )

    return workflow
