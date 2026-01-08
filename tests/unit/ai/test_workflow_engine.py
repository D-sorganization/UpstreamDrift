"""Unit tests for WorkflowEngine."""

from __future__ import annotations

import pytest

from shared.python.ai.exceptions import WorkflowError
from shared.python.ai.tool_registry import ToolRegistry
from shared.python.ai.types import ConversationContext, ExpertiseLevel
from shared.python.ai.workflow_engine import (
    RecoveryStrategy,
    StepStatus,
    Workflow,
    WorkflowEngine,
    WorkflowStep,
    create_first_analysis_workflow,
)


class TestWorkflowStep:
    """Tests for WorkflowStep."""

    def test_step_creation(self) -> None:
        """Test basic step creation."""
        step = WorkflowStep(
            id="test_step",
            name="Test Step",
            description="A test step",
        )
        assert step.id == "test_step"
        assert step.tool_name is None
        assert step.on_failure == RecoveryStrategy.ASK_USER


class TestWorkflow:
    """Tests for Workflow."""

    def test_workflow_creation(self) -> None:
        """Test workflow creation."""
        workflow = Workflow(
            id="test_workflow",
            name="Test Workflow",
            description="A test workflow",
        )
        assert workflow.id == "test_workflow"
        assert len(workflow.steps) == 0

    def test_add_step(self) -> None:
        """Test adding steps to workflow."""
        workflow = Workflow(id="test", name="Test", description="Test")
        workflow.add_step(WorkflowStep(id="step1", name="Step 1", description="First"))
        workflow.add_step(WorkflowStep(id="step2", name="Step 2", description="Second"))
        assert len(workflow.steps) == 2
        assert workflow.steps[0].id == "step1"


class TestWorkflowEngine:
    """Tests for WorkflowEngine."""

    def test_engine_creation(self) -> None:
        """Test engine creation."""
        registry = ToolRegistry()
        engine = WorkflowEngine(registry)
        assert len(engine) == 0

    def test_register_workflow(self) -> None:
        """Test registering a workflow."""
        registry = ToolRegistry()
        engine = WorkflowEngine(registry)
        workflow = Workflow(id="test", name="Test", description="Test")
        engine.register_workflow(workflow)
        assert len(engine) == 1
        assert engine.get_workflow("test") is workflow

    def test_list_workflows(self) -> None:
        """Test listing workflows with expertise filter."""
        registry = ToolRegistry()
        engine = WorkflowEngine(registry)

        beginner_wf = Workflow(
            id="beginner",
            name="Beginner",
            description="",
            expertise_level=ExpertiseLevel.BEGINNER,
        )
        expert_wf = Workflow(
            id="expert",
            name="Expert",
            description="",
            expertise_level=ExpertiseLevel.EXPERT,
        )

        engine.register_workflow(beginner_wf)
        engine.register_workflow(expert_wf)

        # Beginner should only see beginner workflow
        beginner_list = engine.list_workflows(max_expertise=ExpertiseLevel.BEGINNER)
        assert len(beginner_list) == 1
        assert beginner_list[0].id == "beginner"

        # Expert should see both
        expert_list = engine.list_workflows(max_expertise=ExpertiseLevel.EXPERT)
        assert len(expert_list) == 2

    def test_start_workflow(self) -> None:
        """Test starting a workflow."""
        registry = ToolRegistry()
        engine = WorkflowEngine(registry)

        workflow = Workflow(id="test", name="Test", description="Test")
        workflow.add_step(WorkflowStep(id="step1", name="Step 1", description="First"))
        engine.register_workflow(workflow)

        context = ConversationContext()
        execution = engine.start_workflow("test", context)

        assert execution.workflow_id == "test"
        assert execution.status == StepStatus.RUNNING
        assert execution.current_step_index == 0

    def test_start_workflow_not_found(self) -> None:
        """Test starting non-existent workflow raises error."""
        registry = ToolRegistry()
        engine = WorkflowEngine(registry)
        context = ConversationContext()

        with pytest.raises(WorkflowError):
            engine.start_workflow("nonexistent", context)

    def test_get_current_step(self) -> None:
        """Test getting current step."""
        registry = ToolRegistry()
        engine = WorkflowEngine(registry)

        workflow = Workflow(id="test", name="Test", description="Test")
        workflow.add_step(WorkflowStep(id="step1", name="Step 1", description="First"))
        workflow.add_step(WorkflowStep(id="step2", name="Step 2", description="Second"))
        engine.register_workflow(workflow)

        context = ConversationContext()
        execution = engine.start_workflow("test", context)

        current = engine.get_current_step(execution)
        assert current is not None
        assert current.id == "step1"

    def test_execute_step_without_tool(self) -> None:
        """Test executing a step that has no tool."""
        registry = ToolRegistry()
        engine = WorkflowEngine(registry)

        workflow = Workflow(id="test", name="Test", description="Test")
        workflow.add_step(
            WorkflowStep(id="intro", name="Intro", description="Introduction")
        )
        engine.register_workflow(workflow)

        context = ConversationContext()
        execution = engine.start_workflow("test", context)

        result = engine.execute_next_step(execution)
        assert result.status == StepStatus.COMPLETED
        assert execution.current_step_index == 1

    def test_execute_step_with_tool(self) -> None:
        """Test executing a step that runs a tool."""
        registry = ToolRegistry()

        @registry.register("test_tool", "A test tool")
        def test_tool() -> dict[str, str]:
            return {"status": "success"}

        engine = WorkflowEngine(registry)

        workflow = Workflow(id="test", name="Test", description="Test")
        workflow.add_step(
            WorkflowStep(
                id="run_tool",
                name="Run Tool",
                description="Run test tool",
                tool_name="test_tool",
            )
        )
        engine.register_workflow(workflow)

        context = ConversationContext()
        execution = engine.start_workflow("test", context)

        result = engine.execute_next_step(execution)
        assert result.status == StepStatus.COMPLETED
        assert result.result == {"status": "success"}

    def test_execute_step_with_condition_skip(self) -> None:
        """Test that steps with false condition are skipped."""
        registry = ToolRegistry()
        engine = WorkflowEngine(registry)

        workflow = Workflow(id="test", name="Test", description="Test")
        workflow.add_step(
            WorkflowStep(
                id="conditional",
                name="Conditional",
                description="Skipped step",
                condition=lambda state: False,  # Always skip
            )
        )
        engine.register_workflow(workflow)

        context = ConversationContext()
        execution = engine.start_workflow("test", context)

        result = engine.execute_next_step(execution)
        assert result.status == StepStatus.SKIPPED

    def test_is_complete(self) -> None:
        """Test completion check."""
        registry = ToolRegistry()
        engine = WorkflowEngine(registry)

        workflow = Workflow(id="test", name="Test", description="Test")
        workflow.add_step(
            WorkflowStep(id="step1", name="Step 1", description="Only step")
        )
        engine.register_workflow(workflow)

        context = ConversationContext()
        execution = engine.start_workflow("test", context)

        assert engine.is_complete(execution) is False

        engine.execute_next_step(execution)

        assert engine.is_complete(execution) is True
        assert execution.status == StepStatus.COMPLETED

    def test_get_progress(self) -> None:
        """Test progress reporting."""
        registry = ToolRegistry()
        engine = WorkflowEngine(registry)

        workflow = Workflow(id="test", name="Test Workflow", description="Test")
        workflow.add_step(WorkflowStep(id="step1", name="Step 1", description="First"))
        workflow.add_step(WorkflowStep(id="step2", name="Step 2", description="Second"))
        engine.register_workflow(workflow)

        context = ConversationContext()
        execution = engine.start_workflow("test", context)

        progress = engine.get_progress(execution)
        assert progress["total_steps"] == 2
        assert progress["completed_steps"] == 0
        assert progress["progress_percent"] == 0.0

        engine.execute_next_step(execution)

        progress = engine.get_progress(execution)
        assert progress["completed_steps"] == 1
        assert progress["progress_percent"] == 50.0


class TestFirstAnalysisWorkflow:
    """Tests for the first_analysis workflow."""

    def test_workflow_creation(self) -> None:
        """Test creating the first_analysis workflow."""
        workflow = create_first_analysis_workflow()
        assert workflow.id == "first_analysis"
        assert workflow.expertise_level == ExpertiseLevel.BEGINNER
        assert len(workflow.steps) == 5

    def test_workflow_has_educational_content(self) -> None:
        """Test that steps have educational content."""
        workflow = create_first_analysis_workflow()
        welcome_step = workflow.steps[0]
        assert "beginner" in welcome_step.educational_content
        assert len(welcome_step.educational_content["beginner"]) > 0
