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


# ---------------------------------------------------------------------------
# WorkflowEngine — get_progress
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# WorkflowEngine — get_step_educational_content
# ---------------------------------------------------------------------------
