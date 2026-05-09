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


# ---------------------------------------------------------------------------
# WorkflowEngine — registration
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# WorkflowEngine — start_workflow
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# WorkflowEngine — is_complete / get_current_step
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# WorkflowEngine — get_step_educational_content
# ---------------------------------------------------------------------------
