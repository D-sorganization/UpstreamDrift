"""Tests for sidekick.agent.action_service (epic #5967 / S2 / #5971).

TDD: contract pinned before implementation. The service is the single
audited choke-point through which every agentic action flows; every
subsequent adapter (subtab, host, feature-catalog) implements one Protocol.
"""

from __future__ import annotations

from typing import Any
from collections.abc import Mapping, Sequence

import pytest

from sidekick.agent.action_service import (
    ActionDescriptor,
    ActionResult,
    SidekickActionHandler,
    SidekickActionService,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _RecordingHandler:
    """Minimal SidekickActionHandler with two read actions and one write."""

    namespace = "test"

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self._descriptors: tuple[ActionDescriptor, ...] = (
            ActionDescriptor(
                action_id="test.echo",
                summary="Return the input value.",
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
                summary="Pretend to mutate state.",
                params_schema={"type": "object", "properties": {}},
                side_effects="write",
                reversible=True,
            ),
        )

    def describe(self) -> Sequence[ActionDescriptor]:
        return self._descriptors

    def invoke(self, action_id: str, params: Mapping[str, Any]) -> ActionResult:
        self.calls.append((action_id, dict(params)))
        if action_id == "test.echo":
            return ActionResult(ok=True, value=params["value"])
        if action_id == "test.write":
            return ActionResult(ok=True, value=None, undo_token="tok-1")
        return ActionResult(ok=False, error=f"unknown:{action_id}")


class _RaisingHandler:
    namespace = "boom"

    def describe(self) -> Sequence[ActionDescriptor]:
        return (
            ActionDescriptor(
                action_id="boom.fail",
                summary="Always raises.",
                params_schema={"type": "object", "properties": {}},
                side_effects="read",
                reversible=False,
            ),
        )

    def invoke(self, action_id: str, params: Mapping[str, Any]) -> ActionResult:
        raise RuntimeError("kaboom")


class _CollidingHandler:
    namespace = "test"

    def describe(self) -> Sequence[ActionDescriptor]:
        return (
            ActionDescriptor(
                action_id="test.echo",  # collides with _RecordingHandler
                summary="duplicate",
                params_schema={"type": "object", "properties": {}},
                side_effects="read",
                reversible=False,
            ),
        )

    def invoke(self, action_id: str, params: Mapping[str, Any]) -> ActionResult:
        return ActionResult(ok=True)


# ---------------------------------------------------------------------------
# ActionDescriptor — DbC
# ---------------------------------------------------------------------------


def test_action_descriptor_rejects_bad_side_effects() -> None:
    with pytest.raises(ValueError, match="side_effects"):
        ActionDescriptor(
            action_id="x.y",
            summary="s",
            params_schema={"type": "object"},
            side_effects="erase_the_planet",  # type: ignore[arg-type]
            reversible=False,
        )


def test_action_descriptor_requires_jsonschema_shaped_params() -> None:
    with pytest.raises(ValueError, match="params_schema"):
        ActionDescriptor(
            action_id="x.y",
            summary="s",
            params_schema={"not": "a schema"},  # missing "type"
            side_effects="read",
            reversible=False,
        )


def test_action_descriptor_requires_dotted_id() -> None:
    with pytest.raises(ValueError, match="action_id"):
        ActionDescriptor(
            action_id="nodot",
            summary="s",
            params_schema={"type": "object"},
            side_effects="read",
            reversible=False,
        )


# ---------------------------------------------------------------------------
# ActionResult — invariants
# ---------------------------------------------------------------------------


def test_action_result_ok_implies_no_error() -> None:
    with pytest.raises(ValueError, match="ok=True"):
        ActionResult(ok=True, error="should not be set")


def test_action_result_not_ok_requires_error() -> None:
    with pytest.raises(ValueError, match="error"):
        ActionResult(ok=False)


# ---------------------------------------------------------------------------
# Service registration
# ---------------------------------------------------------------------------


def test_register_exposes_actions() -> None:
    service = SidekickActionService()
    service.register(_RecordingHandler())
    ids = [d.action_id for d in service.list_actions()]
    assert ids == ["test.echo", "test.write"]


def test_list_actions_is_sorted_alphabetically() -> None:
    service = SidekickActionService()
    service.register(_RecordingHandler())
    ids = [d.action_id for d in service.list_actions()]
    assert ids == sorted(ids)


def test_duplicate_action_id_raises_at_register() -> None:
    service = SidekickActionService()
    service.register(_RecordingHandler())
    with pytest.raises(ValueError, match="duplicate action_id"):
        service.register(_CollidingHandler())


def test_handler_must_implement_protocol() -> None:
    service = SidekickActionService()
    with pytest.raises(TypeError):
        service.register("not a handler")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Dispatch — validation + audit
# ---------------------------------------------------------------------------


def test_invoke_dispatches_to_handler_on_valid_params() -> None:
    handler = _RecordingHandler()
    service = SidekickActionService()
    service.register(handler)
    result = service.invoke("test.echo", {"value": 7})
    assert result.ok is True
    assert result.value == 7
    assert handler.calls == [("test.echo", {"value": 7})]


def test_invoke_unknown_action_returns_error_result() -> None:
    service = SidekickActionService()
    result = service.invoke("nope.nada", {})
    assert result.ok is False
    assert result.error is not None
    assert "nope.nada" in result.error


def test_invoke_invalid_params_does_not_call_handler() -> None:
    handler = _RecordingHandler()
    service = SidekickActionService()
    service.register(handler)
    result = service.invoke("test.echo", {"value": "not-an-int"})
    assert result.ok is False
    assert result.error is not None
    assert handler.calls == [], "handler must not be called on schema failure"


def test_invoke_handler_exception_is_translated_to_error_result() -> None:
    service = SidekickActionService()
    service.register(_RaisingHandler())
    result = service.invoke("boom.fail", {})
    assert result.ok is False
    assert result.error is not None
    assert "kaboom" in result.error or "boom.fail" in result.error


def test_invoke_records_to_audit_sink() -> None:
    events: list[tuple[str, bool]] = []
    service = SidekickActionService(
        audit_sink=lambda call: events.append((call.action_id, call.result.ok))
    )
    service.register(_RecordingHandler())
    service.invoke("test.echo", {"value": 1})
    service.invoke("nope.nada", {})
    assert events == [("test.echo", True), ("nope.nada", False)]


def test_invoke_with_dry_run_does_not_call_handler() -> None:
    handler = _RecordingHandler()
    service = SidekickActionService()
    service.register(handler)
    result = service.invoke("test.write", {}, dry_run=True)
    assert result.ok is True
    assert handler.calls == []
    # The dry-run payload tells callers what would have happened.
    assert "dry_run" in result.metadata


# ---------------------------------------------------------------------------
# Protocol runtime check
# ---------------------------------------------------------------------------


def test_recording_handler_satisfies_protocol() -> None:
    handler = _RecordingHandler()
    assert isinstance(handler, SidekickActionHandler)


def test_string_does_not_satisfy_protocol() -> None:
    assert not isinstance("hello", SidekickActionHandler)
