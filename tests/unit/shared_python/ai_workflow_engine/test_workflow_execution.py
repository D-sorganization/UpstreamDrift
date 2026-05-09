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
        with pytest.raises(ValueError):
            exe.get_step_result(None)  # type: ignore[arg-type]


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


# ---------------------------------------------------------------------------
# WorkflowEngine — get_progress
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# WorkflowEngine — get_step_educational_content
# ---------------------------------------------------------------------------
