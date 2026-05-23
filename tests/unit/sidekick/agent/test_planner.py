"""Tests for sidekick.agent.planner (epic #5967 / S5 / #5974).

TDD: contract pinned before implementation. The planner sits between
the LLM-emitted tool calls and the action service; it validates each
proposed action against the registered descriptors and produces
PlannedSteps that the chat layer can render or execute.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pytest

from sidekick.agent.action_service import (
    ActionDescriptor,
    ActionResult,
    SidekickActionService,
)
from sidekick.agent.planner import (
    PlannedStep,
    PlannerError,
    SidekickAgentPlanner,
    ToolCall,
    build_sidekick_system_prompt,
)
from sidekick.agent.subtab_adapter import SubtabAdapter

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class _FakeSubtabPort:
    def list_tabs(self) -> Sequence[str]:
        return ("calculator", "workspace")

    def active_tab(self) -> str | None:
        return "calculator"

    def focus(self, tab_id: str) -> None:
        if tab_id not in self.list_tabs():
            raise KeyError(tab_id)

    def set_visible(self, tab_id: str, visible: bool) -> None:
        if tab_id not in self.list_tabs():
            raise KeyError(tab_id)

    def workspace_snapshot(self):  # type: ignore[no-untyped-def]
        from sidekick.agent.subtab_adapter import WorkspaceSnapshot

        return WorkspaceSnapshot(values={})

    def workspace_set_variable(self, name: str, value: Any) -> Any:
        return None

    def calculator_run(self, calculator_id: str, inputs: Mapping[str, Any]):  # type: ignore[no-untyped-def]
        from sidekick.agent.subtab_adapter import CalculatorRun

        return CalculatorRun(values={"answer": 1.0})

    def state_profile_save(self, name: str, payload: Mapping[str, Any]) -> None:
        pass

    def state_profile_load(self, name: str):  # type: ignore[no-untyped-def]
        from sidekick.agent.subtab_adapter import StateProfile

        return StateProfile(name=name, payload={})


def _build_service_with_subtab() -> SidekickActionService:
    service = SidekickActionService()
    service.register(SubtabAdapter(port=_FakeSubtabPort()))
    return service


# ---------------------------------------------------------------------------
# ToolCall + PlannedStep DbC
# ---------------------------------------------------------------------------


def test_tool_call_rejects_non_string_id() -> None:
    with pytest.raises(TypeError):
        ToolCall(action_id=123, params={})  # type: ignore[arg-type]


def test_tool_call_rejects_empty_id() -> None:
    with pytest.raises(ValueError, match="action_id"):
        ToolCall(action_id="", params={})


def test_planned_step_rejects_empty_action_id() -> None:
    with pytest.raises(ValueError):
        PlannedStep(action_id="", params={}, rationale="x")


def test_planned_step_is_frozen() -> None:
    import dataclasses

    step = PlannedStep(action_id="subtab.list", params={}, rationale="show me tabs")
    with pytest.raises(dataclasses.FrozenInstanceError):
        step.rationale = "no"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Planner construction
# ---------------------------------------------------------------------------


def test_planner_rejects_non_service() -> None:
    with pytest.raises(TypeError):
        SidekickAgentPlanner(service="not-a-service")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# plan_from_tool_calls
# ---------------------------------------------------------------------------


def test_plan_emits_one_step_per_known_action() -> None:
    service = _build_service_with_subtab()
    planner = SidekickAgentPlanner(service=service)
    calls = (
        ToolCall(action_id="subtab.list", params={}),
        ToolCall(action_id="subtab.focus", params={"tab_id": "workspace"}),
    )
    steps = planner.plan_from_tool_calls(calls)
    assert [s.action_id for s in steps] == ["subtab.list", "subtab.focus"]


def test_plan_rejects_unknown_action_with_error_step() -> None:
    service = _build_service_with_subtab()
    planner = SidekickAgentPlanner(service=service)
    calls = (ToolCall(action_id="nope.nada", params={}),)
    steps = planner.plan_from_tool_calls(calls)
    assert len(steps) == 1
    assert steps[0].is_error
    assert "nope.nada" in steps[0].error_message


def test_plan_validates_against_descriptor_schema() -> None:
    service = _build_service_with_subtab()
    planner = SidekickAgentPlanner(service=service)
    # subtab.focus requires tab_id
    calls = (ToolCall(action_id="subtab.focus", params={}),)
    steps = planner.plan_from_tool_calls(calls)
    assert len(steps) == 1
    assert steps[0].is_error
    assert "tab_id" in steps[0].error_message


def test_plan_preserves_rationale_text() -> None:
    service = _build_service_with_subtab()
    planner = SidekickAgentPlanner(service=service)
    calls = (
        ToolCall(
            action_id="subtab.list",
            params={},
            rationale="user asked what tabs exist",
        ),
    )
    steps = planner.plan_from_tool_calls(calls)
    assert steps[0].rationale == "user asked what tabs exist"


def test_plan_is_deterministic_for_same_input() -> None:
    service = _build_service_with_subtab()
    planner = SidekickAgentPlanner(service=service)
    calls = (ToolCall(action_id="subtab.list", params={}),)
    a = planner.plan_from_tool_calls(calls)
    b = planner.plan_from_tool_calls(calls)
    assert a == b


# ---------------------------------------------------------------------------
# execute
# ---------------------------------------------------------------------------


def test_execute_dispatches_to_action_service() -> None:
    service = _build_service_with_subtab()
    planner = SidekickAgentPlanner(service=service)
    step = planner.plan_from_tool_calls(
        (ToolCall(action_id="subtab.list", params={}),)
    )[0]
    result = planner.execute(step)
    assert isinstance(result, ActionResult)
    assert result.ok is True
    assert result.value == ["calculator", "workspace"]


def test_execute_on_error_step_raises_planner_error() -> None:
    service = _build_service_with_subtab()
    planner = SidekickAgentPlanner(service=service)
    step = planner.plan_from_tool_calls((ToolCall(action_id="nope.nada", params={}),))[
        0
    ]
    assert step.is_error
    with pytest.raises(PlannerError):
        planner.execute(step)


def test_execute_respects_dry_run_flag() -> None:
    service = _build_service_with_subtab()
    planner = SidekickAgentPlanner(service=service)
    step = planner.plan_from_tool_calls(
        (ToolCall(action_id="subtab.focus", params={"tab_id": "workspace"}),)
    )[0]
    result = planner.execute(step, dry_run=True)
    assert result.ok is True
    assert "dry_run" in result.metadata


# ---------------------------------------------------------------------------
# Tool registry bridge
# ---------------------------------------------------------------------------


def test_export_for_tool_registry_returns_one_tool_per_action() -> None:
    service = _build_service_with_subtab()
    planner = SidekickAgentPlanner(service=service)
    tools = planner.export_for_tool_registry()
    # One per registered action.
    assert len(tools) == len(service.list_actions())
    # Every tool name is sidekick-namespaced and matches a real action.
    action_ids = {d.action_id for d in service.list_actions()}
    tool_names = {t["name"] for t in tools}
    assert all(name.startswith("sidekick.action.") for name in tool_names)
    for tool in tools:
        bare = tool["name"].removeprefix("sidekick.action.")
        assert bare in action_ids


def test_export_for_tool_registry_is_deterministic() -> None:
    service = _build_service_with_subtab()
    planner = SidekickAgentPlanner(service=service)
    a = planner.export_for_tool_registry()
    b = planner.export_for_tool_registry()
    assert a == b


def test_export_includes_schema_and_description() -> None:
    service = _build_service_with_subtab()
    planner = SidekickAgentPlanner(service=service)
    tools = planner.export_for_tool_registry()
    for tool in tools:
        assert "name" in tool
        assert "description" in tool
        assert "parameters" in tool
        assert tool["parameters"]["type"] == "object"


# ---------------------------------------------------------------------------
# System prompt builder
# ---------------------------------------------------------------------------


def test_system_prompt_lists_every_action_id() -> None:
    service = _build_service_with_subtab()
    prompt = build_sidekick_system_prompt(service=service)
    for descriptor in service.list_actions():
        assert descriptor.action_id in prompt


def test_system_prompt_marks_destructive_actions() -> None:
    """Destructive actions are clearly flagged so the LLM knows to ask."""
    service = SidekickActionService()
    service.register(
        _ActionsHandler(
            [
                ActionDescriptor(
                    action_id="x.boom",
                    summary="Erase all data.",
                    params_schema={"type": "object"},
                    side_effects="destructive",
                    reversible=False,
                ),
            ]
        )
    )
    prompt = build_sidekick_system_prompt(service=service)
    assert "destructive" in prompt.lower()
    assert "x.boom" in prompt


def test_system_prompt_is_deterministic() -> None:
    service = _build_service_with_subtab()
    a = build_sidekick_system_prompt(service=service)
    b = build_sidekick_system_prompt(service=service)
    assert a == b


def test_system_prompt_empty_service_still_returns_baseline() -> None:
    prompt = build_sidekick_system_prompt(service=SidekickActionService())
    assert prompt  # never empty
    assert "Sidekick" in prompt


class _ActionsHandler:
    namespace = "x"

    def __init__(self, descs: Sequence[ActionDescriptor]) -> None:
        self._descs = tuple(descs)

    def describe(self) -> Sequence[ActionDescriptor]:
        return self._descs

    def invoke(self, action_id: str, params: Mapping[str, Any]) -> ActionResult:
        return ActionResult(ok=True)
