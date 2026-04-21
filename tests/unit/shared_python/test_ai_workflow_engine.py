"""Unit tests for shared/python/ai/workflow_engine.py."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.shared.python.ai.exceptions import WorkflowError
from src.shared.python.ai.tool_registry import ToolRegistry
from src.shared.python.ai.types import ConversationContext, ExpertiseLevel, ToolResult
from src.shared.python.ai.workflow_engine import (
    RecoveryStrategy,
    StepResult,
    StepStatus,
    ValidationResult,
    Workflow,
    WorkflowEngine,
    WorkflowExecution,
    WorkflowStep,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def registry() -> ToolRegistry:
    return ToolRegistry()


@pytest.fixture
def engine(registry: ToolRegistry) -> WorkflowEngine:
    return WorkflowEngine(registry)


@pytest.fixture
def context() -> ConversationContext:
    return ConversationContext(session_id="test-session")


def _make_workflow(
    wf_id: str = "wf1",
    num_steps: int = 2,
) -> Workflow:
    wf = Workflow(id=wf_id, name="Test Workflow", description="A test workflow")
    for i in range(num_steps):
        wf.add_step(
            WorkflowStep(
                id=f"step_{i}",
                name=f"Step {i}",
                description=f"Step {i} description",
            )
        )
    return wf


def _make_tool_result(success: bool = True, result: object = None) -> ToolResult:
    tr = MagicMock(spec=ToolResult)
    tr.success = success
    tr.result = result or {}
    tr.error = None if success else "tool error"
    return tr


# ---------------------------------------------------------------------------
# Dataclass tests
# ---------------------------------------------------------------------------


class TestValidationResult:
    def test_passed(self) -> None:
        vr = ValidationResult(passed=True, message="OK")
        assert vr.passed is True
        assert vr.message == "OK"

    def test_failed(self) -> None:
        vr = ValidationResult(passed=False, message="Failed", details={"key": "val"})
        assert vr.passed is False
        assert vr.details == {"key": "val"}


class TestWorkflowStep:
    def test_defaults(self) -> None:
        s = WorkflowStep(id="s1", name="Step 1", description="Desc")
        assert s.tool_name is None
        assert s.on_failure == RecoveryStrategy.ASK_USER
        assert s.timeout == 300.0

    def test_custom_on_failure(self) -> None:
        s = WorkflowStep(
            id="s1", name="S", description="D", on_failure=RecoveryStrategy.ABORT
        )
        assert s.on_failure == RecoveryStrategy.ABORT


class TestWorkflow:
    def test_add_step(self) -> None:
        wf = Workflow(id="wf", name="WF", description="D")
        wf.add_step(WorkflowStep(id="s1", name="S", description="D"))
        assert len(wf.steps) == 1

    def test_default_expertise_level(self) -> None:
        wf = Workflow(id="wf", name="WF", description="D")
        assert wf.expertise_level == ExpertiseLevel.BEGINNER


class TestWorkflowExecution:
    def test_get_current_step_result_empty(self, context: ConversationContext) -> None:
        exe = WorkflowExecution(
            execution_id="e1",
            workflow_id="wf1",
            context=context,
        )
        assert exe.get_current_step_result() is None

    def test_get_current_step_result_with_results(
        self, context: ConversationContext
    ) -> None:
        exe = WorkflowExecution(
            execution_id="e1",
            workflow_id="wf1",
            context=context,
        )
        r1 = StepResult(step_id="s1", status=StepStatus.COMPLETED)
        r2 = StepResult(step_id="s2", status=StepStatus.COMPLETED)
        exe.step_results = [r1, r2]
        assert exe.get_current_step_result() is r2

    def test_get_step_result_by_id(self, context: ConversationContext) -> None:
        exe = WorkflowExecution(
            execution_id="e1",
            workflow_id="wf1",
            context=context,
        )
        r1 = StepResult(step_id="s1", status=StepStatus.COMPLETED)
        exe.step_results = [r1]
        assert exe.get_step_result("s1") is r1

    def test_get_step_result_missing(self, context: ConversationContext) -> None:
        exe = WorkflowExecution(
            execution_id="e1",
            workflow_id="wf1",
            context=context,
        )
        assert exe.get_step_result("nonexistent") is None

    def test_get_step_result_requires_id(self, context: ConversationContext) -> None:
        exe = WorkflowExecution(
            execution_id="e1",
            workflow_id="wf1",
            context=context,
        )
        with pytest.raises((AssertionError, ValueError)):
            exe.get_step_result(None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# WorkflowEngine — registration
# ---------------------------------------------------------------------------


class TestWorkflowEngineRegistration:
    def test_register_workflow(self, engine: WorkflowEngine) -> None:
        wf = _make_workflow()
        engine.register_workflow(wf)
        assert engine.get_workflow("wf1") is wf

    def test_get_nonexistent_returns_none(self, engine: WorkflowEngine) -> None:
        assert engine.get_workflow("missing") is None

    def test_len(self, engine: WorkflowEngine) -> None:
        assert len(engine) == 0
        engine.register_workflow(_make_workflow("w1"))
        engine.register_workflow(_make_workflow("w2"))
        assert len(engine) == 2

    def test_list_workflows_empty(self, engine: WorkflowEngine) -> None:
        assert engine.list_workflows() == []

    def test_list_workflows_returns_sorted(self, engine: WorkflowEngine) -> None:
        wf_b = Workflow(id="b", name="B Workflow", description="B")
        wf_a = Workflow(id="a", name="A Workflow", description="A")
        engine.register_workflow(wf_b)
        engine.register_workflow(wf_a)
        names = [w.name for w in engine.list_workflows()]
        assert names == sorted(names)

    def test_list_workflows_filters_by_expertise(self, engine: WorkflowEngine) -> None:
        beginner_wf = Workflow(
            id="beg",
            name="Beginner",
            description="B",
            expertise_level=ExpertiseLevel.BEGINNER,
        )
        expert_wf = Workflow(
            id="exp",
            name="Expert",
            description="E",
            expertise_level=ExpertiseLevel.EXPERT,
        )
        engine.register_workflow(beginner_wf)
        engine.register_workflow(expert_wf)
        results = engine.list_workflows(max_expertise=ExpertiseLevel.INTERMEDIATE)
        ids = [w.id for w in results]
        assert "beg" in ids
        assert "exp" not in ids


# ---------------------------------------------------------------------------
# WorkflowEngine — start_workflow
# ---------------------------------------------------------------------------


class TestStartWorkflow:
    def test_start_registered_workflow(
        self, engine: WorkflowEngine, context: ConversationContext
    ) -> None:
        wf = _make_workflow()
        engine.register_workflow(wf)
        exe = engine.start_workflow("wf1", context)
        assert exe.workflow_id == "wf1"
        assert exe.status == StepStatus.RUNNING

    def test_start_not_found_raises(
        self, engine: WorkflowEngine, context: ConversationContext
    ) -> None:
        with pytest.raises(WorkflowError):
            engine.start_workflow("nonexistent", context)

    def test_start_with_initial_state(
        self, engine: WorkflowEngine, context: ConversationContext
    ) -> None:
        wf = _make_workflow()
        engine.register_workflow(wf)
        exe = engine.start_workflow("wf1", context, initial_state={"key": "value"})
        assert exe.state == {"key": "value"}

    def test_execution_is_stored(
        self, engine: WorkflowEngine, context: ConversationContext
    ) -> None:
        wf = _make_workflow()
        engine.register_workflow(wf)
        exe = engine.start_workflow("wf1", context)
        assert engine.get_execution(exe.execution_id) is exe

    def test_unique_execution_ids(
        self, engine: WorkflowEngine, context: ConversationContext
    ) -> None:
        wf = _make_workflow()
        engine.register_workflow(wf)
        exe1 = engine.start_workflow("wf1", context)
        exe2 = engine.start_workflow("wf1", context)
        assert exe1.execution_id != exe2.execution_id


# ---------------------------------------------------------------------------
# WorkflowEngine — is_complete / get_current_step
# ---------------------------------------------------------------------------


class TestIsCompleteAndCurrentStep:
    def test_not_complete_at_start(
        self, engine: WorkflowEngine, context: ConversationContext
    ) -> None:
        wf = _make_workflow(num_steps=2)
        engine.register_workflow(wf)
        exe = engine.start_workflow("wf1", context)
        assert not engine.is_complete(exe)

    def test_complete_when_status_completed(
        self, engine: WorkflowEngine, context: ConversationContext
    ) -> None:
        wf = _make_workflow(num_steps=2)
        engine.register_workflow(wf)
        exe = engine.start_workflow("wf1", context)
        exe.status = StepStatus.COMPLETED
        assert engine.is_complete(exe)

    def test_complete_when_status_failed(
        self, engine: WorkflowEngine, context: ConversationContext
    ) -> None:
        wf = _make_workflow(num_steps=2)
        engine.register_workflow(wf)
        exe = engine.start_workflow("wf1", context)
        exe.status = StepStatus.FAILED
        assert engine.is_complete(exe)

    def test_complete_when_index_exceeds_steps(
        self, engine: WorkflowEngine, context: ConversationContext
    ) -> None:
        wf = _make_workflow(num_steps=1)
        engine.register_workflow(wf)
        exe = engine.start_workflow("wf1", context)
        exe.current_step_index = 99
        assert engine.is_complete(exe)

    def test_get_current_step_returns_first(
        self, engine: WorkflowEngine, context: ConversationContext
    ) -> None:
        wf = _make_workflow(num_steps=2)
        engine.register_workflow(wf)
        exe = engine.start_workflow("wf1", context)
        step = engine.get_current_step(exe)
        assert step is not None
        assert step.id == "step_0"

    def test_get_current_step_none_when_complete(
        self, engine: WorkflowEngine, context: ConversationContext
    ) -> None:
        wf = _make_workflow(num_steps=1)
        engine.register_workflow(wf)
        exe = engine.start_workflow("wf1", context)
        exe.current_step_index = 10
        assert engine.get_current_step(exe) is None


# ---------------------------------------------------------------------------
# WorkflowEngine — execute_next_step (no-tool steps)
# ---------------------------------------------------------------------------


class TestExecuteNextStep:
    def test_executes_no_tool_step(
        self, engine: WorkflowEngine, context: ConversationContext
    ) -> None:
        wf = _make_workflow(num_steps=1)
        engine.register_workflow(wf)
        exe = engine.start_workflow("wf1", context)
        result = engine.execute_next_step(exe)
        assert result.status == StepStatus.COMPLETED
        assert result.step_id == "step_0"

    def test_completed_after_all_steps(
        self, engine: WorkflowEngine, context: ConversationContext
    ) -> None:
        wf = _make_workflow(num_steps=2)
        engine.register_workflow(wf)
        exe = engine.start_workflow("wf1", context)
        engine.execute_next_step(exe)
        engine.execute_next_step(exe)
        assert exe.status == StepStatus.COMPLETED

    def test_raises_when_workflow_complete(
        self, engine: WorkflowEngine, context: ConversationContext
    ) -> None:
        wf = _make_workflow(num_steps=1)
        engine.register_workflow(wf)
        exe = engine.start_workflow("wf1", context)
        engine.execute_next_step(exe)
        with pytest.raises(WorkflowError):
            engine.execute_next_step(exe)

    def test_step_condition_false_skips(
        self, engine: WorkflowEngine, context: ConversationContext
    ) -> None:
        wf = Workflow(id="wf", name="WF", description="D")
        wf.add_step(
            WorkflowStep(
                id="s1",
                name="S1",
                description="D",
                condition=lambda _state: False,
            )
        )
        engine.register_workflow(wf)
        exe = engine.start_workflow("wf", context)
        result = engine.execute_next_step(exe)
        assert result.status == StepStatus.SKIPPED

    def test_step_condition_true_runs(
        self, engine: WorkflowEngine, context: ConversationContext
    ) -> None:
        wf = Workflow(id="wf2", name="WF2", description="D")
        wf.add_step(
            WorkflowStep(
                id="s1",
                name="S1",
                description="D",
                condition=lambda _state: True,
            )
        )
        engine.register_workflow(wf)
        exe = engine.start_workflow("wf2", context)
        result = engine.execute_next_step(exe)
        assert result.status == StepStatus.COMPLETED

    def test_validation_passes(
        self, engine: WorkflowEngine, context: ConversationContext
    ) -> None:
        wf = Workflow(id="wf3", name="WF3", description="D")
        wf.add_step(
            WorkflowStep(
                id="s1",
                name="S1",
                description="D",
                validation=lambda _: ValidationResult(passed=True, message="OK"),
            )
        )
        engine.register_workflow(wf)
        exe = engine.start_workflow("wf3", context)
        result = engine.execute_next_step(exe)
        assert result.status == StepStatus.COMPLETED

    def test_validation_fails_abort_strategy(
        self, engine: WorkflowEngine, context: ConversationContext
    ) -> None:
        wf = Workflow(id="wf4", name="WF4", description="D")
        wf.add_step(
            WorkflowStep(
                id="s1",
                name="S1",
                description="D",
                validation=lambda _: ValidationResult(passed=False, message="Bad"),
                on_failure=RecoveryStrategy.ABORT,
            )
        )
        engine.register_workflow(wf)
        exe = engine.start_workflow("wf4", context)
        result = engine.execute_next_step(exe)
        assert result.status == StepStatus.FAILED
        assert exe.status == StepStatus.FAILED

    def test_validation_fails_skip_strategy(
        self, engine: WorkflowEngine, context: ConversationContext
    ) -> None:
        wf = Workflow(id="wf5", name="WF5", description="D")
        wf.add_step(
            WorkflowStep(
                id="s1",
                name="S1",
                description="D",
                validation=lambda _: ValidationResult(passed=False, message="Bad"),
                on_failure=RecoveryStrategy.SKIP,
            )
        )
        engine.register_workflow(wf)
        exe = engine.start_workflow("wf5", context)
        result = engine.execute_next_step(exe)
        assert result.status == StepStatus.SKIPPED


# ---------------------------------------------------------------------------
# WorkflowEngine — get_progress
# ---------------------------------------------------------------------------


class TestGetProgress:
    def test_progress_at_start(
        self, engine: WorkflowEngine, context: ConversationContext
    ) -> None:
        wf = _make_workflow(num_steps=4)
        engine.register_workflow(wf)
        exe = engine.start_workflow("wf1", context)
        prog = engine.get_progress(exe)
        assert prog["total_steps"] == 4
        assert prog["completed_steps"] == 0
        assert prog["progress_percent"] == 0

    def test_progress_after_steps(
        self, engine: WorkflowEngine, context: ConversationContext
    ) -> None:
        wf = _make_workflow(num_steps=4)
        engine.register_workflow(wf)
        exe = engine.start_workflow("wf1", context)
        engine.execute_next_step(exe)
        engine.execute_next_step(exe)
        prog = engine.get_progress(exe)
        assert prog["completed_steps"] == 2
        assert prog["progress_percent"] == 50.0

    def test_progress_workflow_not_found(
        self, engine: WorkflowEngine, context: ConversationContext
    ) -> None:
        exe = WorkflowExecution(
            execution_id="e1",
            workflow_id="nonexistent",
            context=context,
        )
        prog = engine.get_progress(exe)
        assert "error" in prog


# ---------------------------------------------------------------------------
# WorkflowEngine — get_step_educational_content
# ---------------------------------------------------------------------------


class TestGetStepEducationalContent:
    def test_returns_level_content(self, engine: WorkflowEngine) -> None:
        step = WorkflowStep(
            id="s1",
            name="S",
            description="Default desc",
            educational_content={
                "beginner": "Easy explanation",
                "expert": "Technical depth",
            },
        )
        content = engine.get_step_educational_content(step, ExpertiseLevel.BEGINNER)
        assert content == "Easy explanation"

    def test_falls_back_to_description(self, engine: WorkflowEngine) -> None:
        step = WorkflowStep(id="s1", name="S", description="Fallback desc")
        content = engine.get_step_educational_content(step, ExpertiseLevel.EXPERT)
        assert content == "Fallback desc"

    def test_falls_back_to_lower_level(self, engine: WorkflowEngine) -> None:
        step = WorkflowStep(
            id="s1",
            name="S",
            description="Default",
            educational_content={"beginner": "Beginner level"},
        )
        content = engine.get_step_educational_content(step, ExpertiseLevel.INTERMEDIATE)
        assert content == "Beginner level"
