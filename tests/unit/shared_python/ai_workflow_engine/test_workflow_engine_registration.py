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
