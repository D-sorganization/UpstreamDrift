"""Tests for sidekick.agent.chat_surface (epic #5967 / S8 / #5977).

The chat surface module owns the wire-shaped data model for action
chips — the per-step UI affordance the chat layer renders before
executing. Both the PyQt6 assistant panel and the React/Tauri
ChatPanel consume the same model so confirmation behaviour is
identical across surfaces.

This sub-issue lands the headless-testable contract; the per-surface
widget code (which sits on the currently in-flux tools_sidebar
package) lands in follow-up PRs and consumes this model verbatim.
"""

from __future__ import annotations

import json

import pytest

from sidekick.agent.action_service import (
    ActionDescriptor,
    ActionResult,
    SidekickActionService,
)
from sidekick.agent.chat_surface import (
    ActionChipState,
    ChatActionEnvelope,
    build_chip,
    serialize_envelope,
)
from sidekick.agent.planner import PlannedStep, SidekickAgentPlanner, ToolCall

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class _OneActionHandler:
    namespace = "x"

    def __init__(self, side_effects: str = "write", reversible: bool = False) -> None:
        self._desc = ActionDescriptor(
            action_id="x.do",
            summary="Do a thing.",
            params_schema={
                "type": "object",
                "properties": {"a": {"type": "integer"}},
                "required": ["a"],
            },
            side_effects=side_effects,  # type: ignore[arg-type]
            reversible=reversible,
        )

    def describe(self):  # type: ignore[no-untyped-def]
        return (self._desc,)

    def invoke(self, action_id, params):  # type: ignore[no-untyped-def]
        return ActionResult(ok=True, value=params)


def _service(side_effects: str = "write") -> SidekickActionService:
    s = SidekickActionService()
    s.register(_OneActionHandler(side_effects=side_effects))
    return s


# ---------------------------------------------------------------------------
# ActionChipModel DbC
# ---------------------------------------------------------------------------


def test_chip_initial_state_for_read_action_is_ready() -> None:
    s = _service(side_effects="read")
    step = PlannedStep(action_id="x.do", params={"a": 1}, rationale="r")
    chip = build_chip(step=step, service=s)
    assert chip.state == ActionChipState.READY


def test_chip_initial_state_for_write_action_is_ready() -> None:
    s = _service(side_effects="write")
    step = PlannedStep(action_id="x.do", params={"a": 1})
    chip = build_chip(step=step, service=s)
    assert chip.state == ActionChipState.READY


def test_chip_initial_state_for_destructive_action_is_locked() -> None:
    s = _service(side_effects="destructive")
    step = PlannedStep(action_id="x.do", params={"a": 1})
    chip = build_chip(step=step, service=s)
    assert chip.state == ActionChipState.LOCKED


def test_chip_for_error_step_is_error_state() -> None:
    s = _service()
    step = PlannedStep(
        action_id="x.do",
        params={"a": "string-not-int"},
        is_error=True,
        error_message="params validation failed",
    )
    chip = build_chip(step=step, service=s)
    assert chip.state == ActionChipState.ERROR
    assert chip.error_message == "params validation failed"


def test_chip_for_unknown_action_is_error_state() -> None:
    s = _service()
    step = PlannedStep(action_id="unknown.x", params={})
    chip = build_chip(step=step, service=s)
    assert chip.state == ActionChipState.ERROR


def test_chip_is_frozen() -> None:
    import dataclasses

    s = _service()
    chip = build_chip(step=PlannedStep(action_id="x.do", params={"a": 1}), service=s)
    with pytest.raises(dataclasses.FrozenInstanceError):
        chip.state = ActionChipState.RUNNING  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Confirmation / state transitions
# ---------------------------------------------------------------------------


def test_chip_with_confirmation_transitions_locked_to_ready() -> None:
    s = _service(side_effects="destructive")
    chip = build_chip(step=PlannedStep(action_id="x.do", params={"a": 1}), service=s)
    confirmed = chip.with_confirmation()
    assert confirmed.state == ActionChipState.READY
    # Confirmation flag is recorded in params so execute() honours it.
    assert confirmed.params.get("_confirmed") is True


def test_with_confirmation_on_non_destructive_is_a_noop() -> None:
    s = _service(side_effects="write")
    chip = build_chip(step=PlannedStep(action_id="x.do", params={"a": 1}), service=s)
    after = chip.with_confirmation()
    # Same chip; nothing to confirm.
    assert after.params.get("_confirmed") is None


def test_with_confirmation_on_error_chip_raises() -> None:
    s = _service()
    chip = build_chip(
        step=PlannedStep(
            action_id="x.do",
            params={},
            is_error=True,
            error_message="bad",
        ),
        service=s,
    )
    with pytest.raises(RuntimeError, match="error"):
        chip.with_confirmation()


# ---------------------------------------------------------------------------
# Side-effects label, reversible flag exposed
# ---------------------------------------------------------------------------


def test_chip_exposes_side_effects_label() -> None:
    s = _service(side_effects="destructive")
    chip = build_chip(step=PlannedStep(action_id="x.do", params={"a": 1}), service=s)
    assert chip.side_effects == "destructive"


def test_chip_exposes_reversible_flag() -> None:
    handler = _OneActionHandler(side_effects="write", reversible=True)
    s = SidekickActionService()
    s.register(handler)
    chip = build_chip(step=PlannedStep(action_id="x.do", params={"a": 1}), service=s)
    assert chip.reversible is True


# ---------------------------------------------------------------------------
# Envelope + serialization
# ---------------------------------------------------------------------------


def test_envelope_carries_every_chip() -> None:
    s = _service()
    planner = SidekickAgentPlanner(service=s)
    steps = planner.plan_from_tool_calls((ToolCall(action_id="x.do", params={"a": 1}),))
    envelope = ChatActionEnvelope(
        steps=steps,
        chips=tuple(build_chip(step=step, service=s) for step in steps),
    )
    assert len(envelope.chips) == 1
    assert envelope.chips[0].action_id == "x.do"


def test_serialize_envelope_is_json_round_trippable() -> None:
    s = _service(side_effects="destructive")
    chip = build_chip(
        step=PlannedStep(action_id="x.do", params={"a": 1}, rationale="why"),
        service=s,
    )
    envelope = ChatActionEnvelope(steps=(), chips=(chip,))
    payload = serialize_envelope(envelope)
    encoded = json.dumps(payload)
    decoded = json.loads(encoded)
    assert decoded["chips"][0]["action_id"] == "x.do"
    assert decoded["chips"][0]["side_effects"] == "destructive"
    assert decoded["chips"][0]["state"] == "locked"
    assert decoded["chips"][0]["rationale"] == "why"


def test_serialize_envelope_redacts_sensitive_params() -> None:
    s = _service()
    chip = build_chip(
        step=PlannedStep(
            action_id="x.do",
            params={"a": 1, "password": "hunter2"},
        ),
        service=s,
    )
    payload = serialize_envelope(ChatActionEnvelope(steps=(), chips=(chip,)))
    assert payload["chips"][0]["params"]["password"] == "***"
    assert payload["chips"][0]["params"]["a"] == 1


def test_serialize_envelope_includes_error_chip_message() -> None:
    s = _service()
    chip = build_chip(
        step=PlannedStep(
            action_id="x.do",
            params={},
            is_error=True,
            error_message="missing 'a'",
        ),
        service=s,
    )
    payload = serialize_envelope(ChatActionEnvelope(steps=(), chips=(chip,)))
    assert payload["chips"][0]["state"] == "error"
    assert payload["chips"][0]["error_message"] == "missing 'a'"
