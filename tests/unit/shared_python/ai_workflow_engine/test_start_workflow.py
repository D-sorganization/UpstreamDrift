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


# ---------------------------------------------------------------------------
# WorkflowEngine — execute_next_step (no-tool steps)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# WorkflowEngine — get_progress
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# WorkflowEngine — get_step_educational_content
# ---------------------------------------------------------------------------
