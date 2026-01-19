from unittest.mock import MagicMock, Mock

import pytest

from shared.python.ai.exceptions import WorkflowError
from shared.python.ai.types import ConversationContext, ExpertiseLevel, ToolResult
from shared.python.ai.workflow_engine import (
    StepStatus,
    Workflow,
    WorkflowEngine,
    WorkflowExecution,
    WorkflowStep,
    create_first_analysis_workflow,
)


class TestAIWorkflowEngine:

    @pytest.fixture
    def mock_tool_registry(self):
        registry = Mock()
        registry.execute.return_value = ToolResult(
            tool_call_id="mock_id", success=True, result={"data": "test"}
        )
        return registry

    @pytest.fixture
    def engine(self, mock_tool_registry):
        return WorkflowEngine(mock_tool_registry)

    @pytest.fixture
    def basic_workflow(self):
        wf = Workflow(
            id="test_workflow",
            name="Test Workflow",
            description="A simple test workflow",
            expertise_level=ExpertiseLevel.BEGINNER,
        )
        step1 = WorkflowStep(id="step1", name="Step 1", description="First step")
        step2 = WorkflowStep(id="step2", name="Step 2", description="Second step")
        wf.add_step(step1)
        wf.add_step(step2)
        return wf

    def test_workflow_registration(self, engine, basic_workflow):
        """Test registering and retrieving workflows."""
        engine.register_workflow(basic_workflow)
        retrieved = engine.get_workflow("test_workflow")
        assert retrieved == basic_workflow
        assert engine.list_workflows()[0].id == "test_workflow"

    def test_start_workflow_success(self, engine, basic_workflow):
        """Test starting a valid workflow."""
        engine.register_workflow(basic_workflow)
        context = MagicMock(spec=ConversationContext)
        initial_state = {"key": "value"}

        execution = engine.start_workflow("test_workflow", context, initial_state)

        assert isinstance(execution, WorkflowExecution)
        assert execution.workflow_id == "test_workflow"
        assert execution.status == StepStatus.RUNNING
        assert execution.state == initial_state
        assert execution.current_step_index == 0

    def test_start_workflow_not_found(self, engine):
        """Test starting a non-existent workflow raises error."""
        context = MagicMock(spec=ConversationContext)
        with pytest.raises(WorkflowError):
            engine.start_workflow("missing_workflow", context)

    def test_execute_steps_sequential(self, engine, basic_workflow):
        """Test executing steps sequentially."""
        engine.register_workflow(basic_workflow)
        context = MagicMock(spec=ConversationContext)
        execution = engine.start_workflow("test_workflow", context)

        # Step 1
        assert not engine.is_complete(execution)
        result1 = engine.execute_next_step(execution)
        assert result1.status == StepStatus.COMPLETED
        assert result1.step_id == "step1"
        assert execution.current_step_index == 1

        # Step 2
        result2 = engine.execute_next_step(execution)
        assert result2.status == StepStatus.COMPLETED
        assert result2.step_id == "step2"
        assert execution.current_step_index == 2

        # Check completion
        assert engine.is_complete(execution)
        assert execution.status == StepStatus.COMPLETED

    def test_execute_step_with_tool(self, engine, mock_tool_registry):
        """Test executing a step that requires a tool."""
        wf = Workflow(id="tool_wf", name="Tool WF", description="desc")
        tool_step = WorkflowStep(
            id="tool_step",
            name="Tool Step",
            description="Run a tool",
            tool_name="my_tool",
            tool_arguments={"arg": 1},
        )
        wf.add_step(tool_step)
        engine.register_workflow(wf)

        context = MagicMock(spec=ConversationContext)
        execution = engine.start_workflow(
            "tool_wf", context, initial_state={"existing": "data"}
        )

        # Execute
        result = engine.execute_next_step(execution)

        assert result.status == StepStatus.COMPLETED
        mock_tool_registry.execute.assert_called_with(
            "my_tool", {"existing": "data", "arg": 1}
        )

    def test_step_condition_skip(self, engine):
        """Test skipping a step based on condition."""
        wf = Workflow(id="cond_wf", name="Conditional WF", description="desc")

        # Step that only runs if 'run_me' is True
        step1 = WorkflowStep(
            id="conditional_step",
            name="Conditional",
            description="Runs if true",
            condition=lambda state: state.get("run_me", False),
        )
        wf.add_step(step1)
        engine.register_workflow(wf)

        context = MagicMock(spec=ConversationContext)
        execution = engine.start_workflow(
            "cond_wf", context, initial_state={"run_me": False}
        )

        result = engine.execute_next_step(execution)
        assert result.status == StepStatus.SKIPPED
        assert engine.is_complete(execution)

    def test_create_first_analysis_workflow(self):
        """Verify the factory function creates the expected workflow."""
        wf = create_first_analysis_workflow()
        assert wf.id == "first_analysis"
        assert len(wf.steps) == 5
        assert wf.steps[0].id == "welcome"
        assert wf.steps[1].tool_name == "list_sample_files"
