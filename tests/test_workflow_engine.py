
import pytest
from unittest.mock import Mock, MagicMock
from collections import namedtuple
from shared.python.ai.workflow_engine import (
    WorkflowEngine,
    Workflow,
    WorkflowStep,
    StepStatus,
    RecoveryStrategy,
    ExpertiseLevel
)
from shared.python.ai.tool_registry import ToolRegistry, ToolResult

# Minimal Context Mock
Context = namedtuple("Context", ["user_id"])

class TestWorkflowEngine:
    @pytest.fixture
    def mock_registry(self):
        registry = Mock(spec=ToolRegistry)
        registry.execute.return_value = ToolResult("", True, "success")
        return registry

    @pytest.fixture
    def engine(self, mock_registry):
        return WorkflowEngine(mock_registry)

    @pytest.fixture
    def simple_workflow(self):
        wf = Workflow(
            id="test_wf",
            name="Test Workflow",
            description="Testing",
            expertise_level=ExpertiseLevel.BEGINNER
        )
        wf.add_step(WorkflowStep("step1", "Step 1", "Do something"))
        wf.add_step(WorkflowStep("step2", "Step 2", "Do else"))
        return wf

    def test_workflow_registration(self, engine, simple_workflow):
        engine.register_workflow(simple_workflow)
        assert engine.get_workflow("test_wf") == simple_workflow
        assert len(engine.list_workflows()) == 1

    def test_start_workflow(self, engine, simple_workflow):
        engine.register_workflow(simple_workflow)
        ctx = Context(user_id="user1")
        
        execution = engine.start_workflow("test_wf", ctx, initial_state={"key": "val"})
        
        assert execution.workflow_id == "test_wf"
        assert execution.status == StepStatus.RUNNING
        assert execution.state["key"] == "val"
        assert execution.current_step_index == 0
        assert execution.execution_id.startswith("exec_")

    def test_execute_simple_steps(self, engine, simple_workflow):
        engine.register_workflow(simple_workflow)
        ctx = Context("user1")
        exe = engine.start_workflow("test_wf", ctx)
        
        # Step 1
        res1 = engine.execute_next_step(exe)
        assert res1.step_id == "step1"
        assert res1.status == StepStatus.COMPLETED
        assert exe.current_step_index == 1
        
        # Step 2
        res2 = engine.execute_next_step(exe)
        assert res2.step_id == "step2"
        assert res2.status == StepStatus.COMPLETED
        assert exe.current_step_index == 2
        assert engine.is_complete(exe)

    def test_tool_execution(self, engine, mock_registry):
        wf = Workflow("tool_wf", "Tool WF", "Desc")
        wf.add_step(WorkflowStep(
            id="tool_step",
            name="Tool Step",
            description="Run tool",
            tool_name="my_tool",
            tool_arguments={"arg": 1}
        ))
        engine.register_workflow(wf)
        
        exe = engine.start_workflow("tool_wf", Context("u1"))
        res = engine.execute_next_step(exe)
        
        mock_registry.execute.assert_called_with("my_tool", {"arg": 1})
        assert res.status == StepStatus.COMPLETED
        assert res.result == "success"

    def test_step_failure_abort(self, engine):
        wf = Workflow("fail_wf", "Fail WF", "Desc")
        wf.add_step(WorkflowStep(
            id="fail_step",
            name="Fail Step",
            description="Will fail",
            tool_name="bad_tool",
            on_failure=RecoveryStrategy.ABORT
        ))
        engine.register_workflow(wf)
        
        # Mock registry failure
        engine._tool_registry.execute.side_effect = Exception("Tool explosion")
        
        exe = engine.start_workflow("fail_wf", Context("u1"))
        res = engine.execute_next_step(exe)
        
        assert res.status == StepStatus.FAILED
        assert "Tool explosion" in res.error
        assert exe.status == StepStatus.FAILED

    def test_step_condition_skip(self, engine):
        wf = Workflow("cond_wf", "Cond WF", "Desc")
        wf.add_step(WorkflowStep(
            id="cond_step",
            name="Conditional",
            description="Maybe run",
            condition=lambda state: state.get("run_me", False)
        ))
        engine.register_workflow(wf)
        
        # Case 1: Skip
        exe = engine.start_workflow("cond_wf", Context("u1"), initial_state={"run_me": False})
        res = engine.execute_next_step(exe)
        assert res.status == StepStatus.SKIPPED
        
        # Case 2: Run
        exe2 = engine.start_workflow("cond_wf", Context("u1"), initial_state={"run_me": True})
        res2 = engine.execute_next_step(exe2)
        assert res2.status == StepStatus.COMPLETED

    def test_educational_content(self, engine):
        step = WorkflowStep(
            "edu", "Edu", "Desc",
            educational_content={"beginner": "Learn X", "expert": "Review Y"}
        )
        
        assert engine.get_step_educational_content(step, ExpertiseLevel.BEGINNER) == "Learn X"
        assert engine.get_step_educational_content(step, ExpertiseLevel.EXPERT) == "Review Y"
        # Fallback
        assert engine.get_step_educational_content(step, ExpertiseLevel.INTERMEDIATE) == "Learn X"
