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
