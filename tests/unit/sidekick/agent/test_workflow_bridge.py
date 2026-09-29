"""Tests for sidekick.agent.workflow_bridge (epic #5967 / S7 / #5976)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pytest

from sidekick.agent.action_service import (
    ActionDescriptor,
    ActionResult,
    SidekickActionService,
)
from sidekick.agent.workflow_bridge import (
    PendingUserDecision,
    SidekickWorkflow,
    WorkflowOutcome,
    WorkflowStepStatus,
    action_step,
    run_sidekick_workflow,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Fake handler
# ---------------------------------------------------------------------------


class _CountingHandler:
    namespace = "t"

    def __init__(self, fail_action: str | None = None) -> None:
        self._fail_action = fail_action
        self.invocations: list[str] = []

    def describe(self) -> Sequence[ActionDescriptor]:
        return (
            ActionDescriptor(
                action_id="t.step_a",
                summary="Step A",
                params_schema={"type": "object"},
                side_effects="read",
                reversible=False,
            ),
            ActionDescriptor(
                action_id="t.step_b",
                summary="Step B",
                params_schema={"type": "object"},
                side_effects="read",
                reversible=False,
            ),
            ActionDescriptor(
                action_id="t.flaky",
                summary="Flaky",
                params_schema={"type": "object"},
                side_effects="read",
                reversible=False,
            ),
        )

    def invoke(self, action_id: str, params: Mapping[str, Any]) -> ActionResult:
        self.invocations.append(action_id)
        if action_id == self._fail_action:
            return ActionResult(ok=False, error="simulated")
        return ActionResult(ok=True, value=f"ran {action_id}")


def _build(service: SidekickActionService, handler: _CountingHandler) -> None:
    service.register(handler)


# ---------------------------------------------------------------------------
# action_step builder DbC
# ---------------------------------------------------------------------------


def test_action_step_rejects_empty_action_id() -> None:
    with pytest.raises(ValueError):
        action_step("", {})


def test_action_step_default_on_failure_is_abort() -> None:
    step = action_step("t.x", {})
    assert step.on_failure == "abort"


def test_action_step_rejects_unknown_recovery_strategy() -> None:
    with pytest.raises(ValueError, match="on_failure"):
        action_step("t.x", {}, on_failure="explode")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# run_sidekick_workflow: happy path
# ---------------------------------------------------------------------------


def test_run_workflow_executes_all_steps_in_order() -> None:
    service = SidekickActionService()
    handler = _CountingHandler()
    _build(service, handler)
    workflow = SidekickWorkflow(
        name="seq",
        steps=(action_step("t.step_a", {}), action_step("t.step_b", {})),
    )
    outcome = run_sidekick_workflow(workflow, service=service)
    assert outcome.completed is True
    assert handler.invocations == ["t.step_a", "t.step_b"]
    assert [r.status for r in outcome.step_results] == [
        WorkflowStepStatus.COMPLETED,
        WorkflowStepStatus.COMPLETED,
    ]


def test_workflow_outcome_aggregates_step_values() -> None:
    service = SidekickActionService()
    _build(service, _CountingHandler())
    workflow = SidekickWorkflow(
        name="seq",
        steps=(action_step("t.step_a", {}), action_step("t.step_b", {})),
    )
    outcome = run_sidekick_workflow(workflow, service=service)
    assert outcome.outputs == {
        "t.step_a": "ran t.step_a",
        "t.step_b": "ran t.step_b",
    }


# ---------------------------------------------------------------------------
# Failure recovery strategies
# ---------------------------------------------------------------------------


def test_failure_with_abort_stops_workflow() -> None:
    service = SidekickActionService()
    handler = _CountingHandler(fail_action="t.flaky")
    _build(service, handler)
    workflow = SidekickWorkflow(
        name="abort",
        steps=(
            action_step("t.step_a", {}),
            action_step("t.flaky", {}, on_failure="abort"),
            action_step("t.step_b", {}),
        ),
    )
    outcome = run_sidekick_workflow(workflow, service=service)
    assert outcome.completed is False
    assert handler.invocations == ["t.step_a", "t.flaky"]


def test_failure_with_skip_continues_workflow() -> None:
    service = SidekickActionService()
    handler = _CountingHandler(fail_action="t.flaky")
    _build(service, handler)
    workflow = SidekickWorkflow(
        name="skip",
        steps=(
            action_step("t.step_a", {}),
            action_step("t.flaky", {}, on_failure="skip"),
            action_step("t.step_b", {}),
        ),
    )
    outcome = run_sidekick_workflow(workflow, service=service)
    assert outcome.completed is True
    assert handler.invocations == ["t.step_a", "t.flaky", "t.step_b"]
    assert outcome.step_results[1].status == WorkflowStepStatus.SKIPPED


def test_failure_with_retry_retries_exactly_once() -> None:
    """Retry strategy: one extra attempt; same call hits handler twice."""
    service = SidekickActionService()
    handler = _CountingHandler(fail_action="t.flaky")
    _build(service, handler)
    workflow = SidekickWorkflow(
        name="retry",
        steps=(action_step("t.flaky", {}, on_failure="retry"),),
    )
    outcome = run_sidekick_workflow(workflow, service=service)
    assert outcome.completed is False  # second attempt also failed
    assert handler.invocations == ["t.flaky", "t.flaky"]


def test_failure_with_ask_user_raises_pending_user_decision() -> None:
    service = SidekickActionService()
    handler = _CountingHandler(fail_action="t.flaky")
    _build(service, handler)
    workflow = SidekickWorkflow(
        name="ask",
        steps=(
            action_step("t.step_a", {}),
            action_step("t.flaky", {}, on_failure="ask_user"),
            action_step("t.step_b", {}),  # should not run
        ),
    )
    with pytest.raises(PendingUserDecision) as exc:
        run_sidekick_workflow(workflow, service=service)
    assert exc.value.action_id == "t.flaky"
    # Step b must not have run.
    assert handler.invocations == ["t.step_a", "t.flaky"]


# ---------------------------------------------------------------------------
# DbC: registration-time validation
# ---------------------------------------------------------------------------


def test_action_step_validates_against_catalog_at_run_time() -> None:
    """An action_step pointing at an unknown action surfaces a clear
    error in WorkflowStepResult rather than crashing."""
    service = SidekickActionService()
    _build(service, _CountingHandler())
    workflow = SidekickWorkflow(
        name="nope",
        steps=(action_step("t.does_not_exist", {}, on_failure="abort"),),
    )
    outcome = run_sidekick_workflow(workflow, service=service)
    assert outcome.completed is False
    assert outcome.step_results[0].status == WorkflowStepStatus.FAILED
    assert "unknown" in (
        outcome.step_results[0].error_message or ""
    ).lower() or "t.does_not_exist" in (outcome.step_results[0].error_message or "")


def test_workflow_with_empty_steps_completes_immediately() -> None:
    service = SidekickActionService()
    workflow = SidekickWorkflow(name="empty", steps=())
    outcome = run_sidekick_workflow(workflow, service=service)
    assert outcome.completed is True
    assert outcome.step_results == ()


def test_workflow_outcome_is_frozen() -> None:
    import dataclasses

    outcome = WorkflowOutcome(
        workflow_name="x", completed=True, step_results=(), outputs={}
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        outcome.completed = False  # type: ignore[misc]
