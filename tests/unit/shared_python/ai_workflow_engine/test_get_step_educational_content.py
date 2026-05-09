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
