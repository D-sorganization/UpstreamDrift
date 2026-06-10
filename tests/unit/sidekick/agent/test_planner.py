"""Focused coverage for Sidekick agent planner."""

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

pytestmark = pytest.mark.unit


class _Handler:
    namespace = "test"

    def describe(self) -> Sequence[ActionDescriptor]:
        return (
            ActionDescriptor(
                action_id="test.echo",
                summary="Echo a value.",
                params_schema={
                    "type": "object",
                    "properties": {"value": {"type": "integer"}},
                    "required": ["value"],
                },
                side_effects="read",
                reversible=False,
            ),
            ActionDescriptor(
                action_id="test.write",
                summary="Write a value.",
                params_schema={"type": "object", "properties": {}},
                side_effects="write",
                reversible=True,
            ),
        )

    def invoke(self, action_id: str, params: Mapping[str, Any]) -> ActionResult:
        return ActionResult(
            ok=True, value={"action_id": action_id, "params": dict(params)}
        )


def _planner() -> SidekickAgentPlanner:
    service = SidekickActionService()
    service.register(_Handler())
    return SidekickAgentPlanner(service=service)


def test_tool_call_and_planned_step_validate_invariants() -> None:
    with pytest.raises(TypeError, match="action_id"):
        ToolCall(action_id=123, params={})  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="params"):
        ToolCall(action_id="test.echo", params=[])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="error_message"):
        PlannedStep(action_id="test.echo", params={}, is_error=True)


def test_planner_emits_normal_unknown_and_schema_error_steps() -> None:
    planner = _planner()

    ok, unknown, invalid = planner.plan_from_tool_calls(
        [
            ToolCall("test.echo", {"value": 4}, rationale="because"),
            ToolCall("test.missing", {}),
            ToolCall("test.echo", {"value": True}),
        ]
    )

    assert ok.is_error is False
    assert ok.rationale == "because"
    assert unknown.is_error is True
    assert "unknown action" in unknown.error_message
    assert invalid.is_error is True
    assert "expected type" in invalid.error_message


def test_execute_refuses_error_steps_and_dispatches_valid_step() -> None:
    planner = _planner()
    (step,) = planner.plan_from_tool_calls([ToolCall("test.echo", {"value": 5})])

    result = planner.execute(step, dry_run=True)

    assert result.ok is True
    assert result.metadata["dry_run"] == {"value": 5}
    assert result.metadata["would_call"] == "test.echo"
    with pytest.raises(PlannerError, match="cannot execute error step"):
        planner.execute(
            PlannedStep("test.echo", {}, is_error=True, error_message="bad")
        )


def test_export_and_system_prompt_are_generated_from_service() -> None:
    planner = _planner()
    exported = planner.export_for_tool_registry()
    prompt = build_sidekick_system_prompt(service=planner._service)  # noqa: SLF001

    assert exported[0]["name"] == "sidekick.action.test.echo"
    assert exported[0]["metadata"]["original_action_id"] == "test.echo"
    assert "`test.write` [write, reversible]" in prompt


def test_system_prompt_handles_empty_service() -> None:
    assert "(No actions registered.)" in build_sidekick_system_prompt(
        service=SidekickActionService()
    )
