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

import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Any

from src.shared.python.ai.exceptions import WorkflowError
from src.shared.python.ai.tool_registry import ToolRegistry
from src.shared.python.ai.types import ConversationContext, ExpertiseLevel, ToolResult
from src.shared.python.logging_config import get_logger

UTC = timezone.utc  # noqa: UP017

logger = get_logger(__name__)


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
            except (RuntimeError, ValueError, OSError) as e:
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
            except Exception as e:  # noqa: BLE001 - tools may raise anything
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
            except (RuntimeError, ValueError, OSError) as e:
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


def create_c3d_import_workflow() -> Workflow:
    """Create the 'c3d_import' workflow for importing motion capture data.

    This workflow guides users through importing and validating
    C3D motion capture files.

    Returns:
        Complete Workflow object.
    """
    workflow = Workflow(
        id="c3d_import",
        name="C3D Motion Capture Import",
        description=(
            "Import and validate C3D motion capture data. This workflow "
            "guides you through loading, inspecting, and preparing motion "
            "data for analysis."
        ),
        expertise_level=ExpertiseLevel.INTERMEDIATE,
        estimated_duration=15,
        tags=["c3d", "import", "motion-capture", "data-loading"],
    )

    # Step 1: Introduction
    workflow.add_step(
        WorkflowStep(
            id="intro",
            name="C3D Import Introduction",
            description="Overview of the C3D import process",
            educational_content={
                "beginner": (
                    "C3D (Coordinate 3D) is the standard format for motion "
                    "capture data. It contains 3D marker positions recorded "
                    "during movement.\n\n"
                    "This workflow will help you:\n"
                    "- Load a C3D file\n"
                    "- Inspect its contents\n"
                    "- Verify data quality"
                ),
                "intermediate": (
                    "C3D files contain both analog and point data streams. "
                    "We'll extract marker trajectories, check for gaps, and "
                    "prepare the data for physics simulation."
                ),
            },
        )
    )

    # Step 2: List available files
    workflow.add_step(
        WorkflowStep(
            id="list_files",
            name="Browse Available Files",
            description="List available C3D files for import",
            tool_name="list_sample_files",
            educational_content={
                "beginner": (
                    "Sample files are included to help you learn. "
                    "You can also import your own C3D files."
                ),
            },
        )
    )

    # Step 3: Load the file
    workflow.add_step(
        WorkflowStep(
            id="load_file",
            name="Load C3D File",
            description="Load the selected C3D file",
            tool_name="load_c3d",
            on_failure=RecoveryStrategy.ASK_USER,
            educational_content={
                "beginner": (
                    "Loading extracts marker positions, frame rate, "
                    "and other metadata from the file."
                ),
                "intermediate": (
                    "The loader validates file structure, checks for "
                    "corrupted frames, and converts units if necessary."
                ),
            },
        )
    )

    # Step 4: Inspect markers
    workflow.add_step(
        WorkflowStep(
            id="inspect_markers",
            name="Inspect Marker Configuration",
            description="Examine the markers present in the data",
            tool_name="get_marker_info",
            educational_content={
                "beginner": (
                    "Markers are reflective balls placed on the body. "
                    "Each marker has a name and represents a specific "
                    "body location."
                ),
                "intermediate": (
                    "The marker set determines which body segments can "
                    "be tracked. Common sets include Plug-in Gait and "
                    "Cleveland Clinic marker sets."
                ),
            },
        )
    )

    return workflow


def create_inverse_dynamics_workflow() -> Workflow:
    """Create the 'inverse_dynamics' workflow for full ID analysis.

    This workflow performs complete inverse dynamics analysis
    with detailed configuration options.

    Returns:
        Complete Workflow object.
    """
    workflow = Workflow(
        id="inverse_dynamics",
        name="Inverse Dynamics Analysis",
        description=(
            "Complete inverse dynamics analysis workflow. Calculate joint "
            "torques, muscle contributions, and analyze movement patterns."
        ),
        expertise_level=ExpertiseLevel.INTERMEDIATE,
        estimated_duration=45,
        tags=["inverse-dynamics", "torques", "analysis", "biomechanics"],
    )

    # Step 1: Introduction
    workflow.add_step(
        WorkflowStep(
            id="intro",
            name="Inverse Dynamics Overview",
            description="Introduction to inverse dynamics analysis",
            educational_content={
                "beginner": (
                    "Inverse dynamics calculates the forces that caused "
                    "a movement. Given how the body moved, we determine "
                    "what muscle forces were required."
                ),
                "intermediate": (
                    "Using Newton-Euler equations and measured kinematics, "
                    "we solve for joint torques. This requires accurate "
                    "segment inertial properties and kinematic data."
                ),
                "advanced": (
                    "The inverse dynamics problem solves τ = M(q)q̈ + C(q,q̇)q̇ + g(q) "
                    "where M is the mass matrix, C represents Coriolis/centrifugal "
                    "effects, and g is the gravity vector."
                ),
            },
        )
    )

    # Step 2: Select physics engine
    workflow.add_step(
        WorkflowStep(
            id="select_engine",
            name="Select Physics Engine",
            description="Choose the physics engine for simulation",
            tool_name="list_physics_engines",
            educational_content={
                "intermediate": (
                    "Available engines:\n"
                    "- MuJoCo: Fast, GPU-accelerated, good for muscle models\n"
                    "- Drake: Robust, precise, good for contacts\n"
                    "- Pinocchio: Efficient, analytical derivatives"
                ),
            },
        )
    )

    # Step 3: Load data
    workflow.add_step(
        WorkflowStep(
            id="load_data",
            name="Load Motion Data",
            description="Load C3D data for analysis",
            tool_name="load_c3d",
            on_failure=RecoveryStrategy.ASK_USER,
        )
    )

    # Step 4: Run inverse dynamics
    workflow.add_step(
        WorkflowStep(
            id="run_id",
            name="Run Inverse Dynamics",
            description="Execute inverse dynamics calculation",
            tool_name="run_inverse_dynamics",
            timeout=600.0,
            on_failure=RecoveryStrategy.RETRY,
            educational_content={
                "beginner": (
                    "The simulation is now calculating joint torques. "
                    "This may take a few minutes depending on data length."
                ),
            },
        )
    )

    # Step 5: Check energy
    workflow.add_step(
        WorkflowStep(
            id="check_energy",
            name="Verify Energy Conservation",
            description="Check physical plausibility of results",
            tool_name="check_energy_conservation",
            educational_content={
                "intermediate": (
                    "Energy conservation is a key validation metric. "
                    "Large energy violations indicate simulation errors."
                ),
            },
        )
    )

    # Step 6: Interpret results
    workflow.add_step(
        WorkflowStep(
            id="interpret",
            name="Interpret Torque Results",
            description="Analyze and interpret the calculated torques",
            tool_name="interpret_torques",
            educational_content={
                "beginner": (
                    "Joint torques tell us about muscle effort:\n"
                    "- High torques = high muscle demands\n"
                    "- Timing patterns reveal movement strategy\n"
                    "- Asymmetries may indicate technique issues"
                ),
            },
        )
    )

    return workflow


def create_cross_engine_validation_workflow() -> Workflow:
    """Create the 'cross_engine_validation' workflow.

    This workflow validates results by comparing across multiple
    physics engines to ensure robustness.

    Returns:
        Complete Workflow object.
    """
    workflow = Workflow(
        id="cross_engine_validation",
        name="Cross-Engine Validation",
        description=(
            "Validate simulation results by comparing outputs from multiple "
            "physics engines. Ensures results are robust and not artifacts "
            "of a specific engine implementation."
        ),
        expertise_level=ExpertiseLevel.ADVANCED,
        estimated_duration=60,
        tags=["validation", "cross-engine", "robustness", "quality-assurance"],
    )

    # Step 1: Introduction
    workflow.add_step(
        WorkflowStep(
            id="intro",
            name="Cross-Engine Validation Overview",
            description="Why cross-engine validation matters",
            educational_content={
                "intermediate": (
                    "Different physics engines use different algorithms. "
                    "Comparing results helps identify numerical issues and "
                    "ensures scientific validity."
                ),
                "advanced": (
                    "Cross-validation detects:\n"
                    "- Numerical instabilities\n"
                    "- Integration method artifacts\n"
                    "- Contact model discrepancies\n"
                    "- Solver convergence issues"
                ),
            },
        )
    )

    # Step 2: Check available engines
    workflow.add_step(
        WorkflowStep(
            id="check_engines",
            name="Check Available Engines",
            description="Verify which physics engines are available",
            tool_name="list_physics_engines",
            educational_content={
                "intermediate": (
                    "At least two engines are needed for cross-validation. "
                    "Results are compared based on:\n"
                    "- Joint torque magnitudes\n"
                    "- Peak timing\n"
                    "- Energy conservation"
                ),
            },
        )
    )

    # Step 3: Load data
    workflow.add_step(
        WorkflowStep(
            id="load_data",
            name="Load Motion Data",
            description="Load C3D data for multi-engine analysis",
            tool_name="load_c3d",
        )
    )

    # Step 4: Run validation
    workflow.add_step(
        WorkflowStep(
            id="run_validation",
            name="Run Cross-Engine Validation",
            description="Execute simulations on multiple engines and compare",
            tool_name="validate_cross_engine",
            timeout=900.0,  # 15 minutes for multiple engines
            on_failure=RecoveryStrategy.ASK_USER,
            educational_content={
                "advanced": (
                    "Validation metrics:\n"
                    "- RMS error between engines\n"
                    "- Correlation coefficient\n"
                    "- Peak torque agreement\n"
                    "- Energy drift comparison\n\n"
                    "Acceptable agreement: RMS < 5%, r > 0.99"
                ),
            },
        )
    )

    # Step 5: Energy conservation check
    workflow.add_step(
        WorkflowStep(
            id="energy_check",
            name="Energy Conservation Analysis",
            description="Verify energy conservation across engines",
            tool_name="check_energy_conservation",
            educational_content={
                "advanced": (
                    "Energy conservation violations should be:\n"
                    "- < 1% for constrained systems\n"
                    "- < 5% for systems with contacts\n"
                    "- Consistent across engines"
                ),
            },
        )
    )

    return workflow


def create_drift_control_decomposition_workflow() -> Workflow:
    """Create the 'drift_control_decomposition' workflow.

    This advanced workflow decomposes control contributions
    into drift and corrective components.

    Returns:
        Complete Workflow object.
    """
    workflow = Workflow(
        id="drift_control_decomposition",
        name="Drift-Control Decomposition Analysis",
        description=(
            "Decompose joint torques into passive (drift) and active (control) "
            "components. Advanced analysis for understanding movement control "
            "strategies and neuromuscular coordination."
        ),
        expertise_level=ExpertiseLevel.EXPERT,
        estimated_duration=90,
        tags=["drift", "control", "decomposition", "expert", "neuromuscular"],
    )

    # Step 1: Introduction
    workflow.add_step(
        WorkflowStep(
            id="intro",
            name="Drift-Control Decomposition Overview",
            description="Introduction to drift-control decomposition",
            educational_content={
                "advanced": (
                    "Drift-control decomposition separates torques into:\n"
                    "- Passive drift: gravity, inertial coupling\n"
                    "- Active control: intentional muscle activation\n\n"
                    "This reveals the control strategy used during movement."
                ),
                "expert": (
                    "The decomposition solves:\n"
                    "τ_total = τ_drift + τ_control\n\n"
                    "where τ_drift = M(q)q̈_free + C(q,q̇)q̇ + g(q)\n"
                    "represents the torques needed to allow natural motion, "
                    "and τ_control represents active corrections."
                ),
            },
        )
    )

    # Step 2: Load and prepare data
    workflow.add_step(
        WorkflowStep(
            id="load_data",
            name="Load Motion Data",
            description="Load high-quality motion capture data",
            tool_name="load_c3d",
            on_failure=RecoveryStrategy.ASK_USER,
            educational_content={
                "expert": (
                    "Drift-control analysis requires:\n"
                    "- High sampling rate (≥200 Hz)\n"
                    "- Low marker noise\n"
                    "- Complete marker visibility\n"
                    "- Accurate segment properties"
                ),
            },
        )
    )

    # Step 3: Run inverse dynamics
    workflow.add_step(
        WorkflowStep(
            id="run_id",
            name="Compute Total Joint Torques",
            description="Calculate total torques via inverse dynamics",
            tool_name="run_inverse_dynamics",
            timeout=600.0,
        )
    )

    # Step 4: Compute drift component
    workflow.add_step(
        WorkflowStep(
            id="compute_drift",
            name="Calculate Drift Component",
            description="Compute passive drift torques",
            tool_name="run_inverse_dynamics",
            tool_arguments={"mode": "drift_only"},
            timeout=600.0,
            educational_content={
                "expert": (
                    "Drift torques represent what happens if the nervous "
                    "system provides no active control - the natural dynamics "
                    "of the linked segment system under gravity."
                ),
            },
        )
    )

    # Step 5: Energy verification
    workflow.add_step(
        WorkflowStep(
            id="verify_energy",
            name="Verify Energy Conservation",
            description="Check energy conservation in decomposition",
            tool_name="check_energy_conservation",
            educational_content={
                "expert": (
                    "The control component should:\n"
                    "- Add energy during acceleration phases\n"
                    "- Remove energy during deceleration\n"
                    "- Show minimal energy when coasting"
                ),
            },
        )
    )

    # Step 6: Cross-validate
    workflow.add_step(
        WorkflowStep(
            id="cross_validate",
            name="Cross-Engine Validation",
            description="Validate decomposition across physics engines",
            tool_name="validate_cross_engine",
            timeout=900.0,
            on_failure=RecoveryStrategy.SKIP,
            educational_content={
                "expert": (
                    "Cross-engine validation ensures the decomposition "
                    "is robust to numerical implementation details."
                ),
            },
        )
    )

    return workflow
