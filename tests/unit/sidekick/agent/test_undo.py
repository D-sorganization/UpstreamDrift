"""Tests for SidekickActionService.undo (epic #5967 / S6 / #5975)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pytest

from sidekick.agent.action_service import (
    ActionDescriptor,
    ActionResult,
    SidekickActionService,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Test double — adapter with undo support
# ---------------------------------------------------------------------------


class _ToggleHandler:
    """A handler that tracks a single boolean flag and supports undo."""

    namespace = "t"

    def __init__(self) -> None:
        self.flag = False

    def describe(self) -> Sequence[ActionDescriptor]:
        return (
            ActionDescriptor(
                action_id="t.set",
                summary="set flag to given value",
                params_schema={
                    "type": "object",
                    "properties": {"value": {"type": "boolean"}},
                    "required": ["value"],
                },
                side_effects="write",
                reversible=True,
            ),
        )

    def invoke(self, action_id: str, params: Mapping[str, Any]) -> ActionResult:
        prior = self.flag
        self.flag = bool(params["value"])
        # The "_undo" metadata key tells the service how to reverse this.
        return ActionResult(
            ok=True,
            value=None,
            undo_token="any-token-the-service-replaces",
            metadata={"_undo": {"action_id": "t.set", "params": {"value": prior}}},
        )


class _NoUndoHandler:
    namespace = "n"

    def describe(self) -> Sequence[ActionDescriptor]:
        return (
            ActionDescriptor(
                action_id="n.write",
                summary="non-reversible write",
                params_schema={"type": "object"},
                side_effects="write",
                reversible=False,
            ),
        )

    def invoke(self, action_id: str, params: Mapping[str, Any]) -> ActionResult:
        return ActionResult(ok=True, value=None)


# ---------------------------------------------------------------------------
# Service.undo
# ---------------------------------------------------------------------------


def test_undo_round_trip_restores_state() -> None:
    handler = _ToggleHandler()
    service = SidekickActionService()
    service.register(handler)
    # Apply: flag flips False → True
    r = service.invoke("t.set", {"value": True})
    assert r.ok is True
    assert handler.flag is True
    assert r.undo_token  # service issued a real token
    # Undo: flag flips back to False
    u = service.undo(r.undo_token)
    assert u.ok is True
    assert handler.flag is False


def test_undo_unknown_token_returns_error() -> None:
    service = SidekickActionService()
    result = service.undo("bogus-token")
    assert result.ok is False
    assert "unknown" in (result.error or "").lower()


def test_undo_twice_on_same_token_returns_error_second_time() -> None:
    handler = _ToggleHandler()
    service = SidekickActionService()
    service.register(handler)
    r = service.invoke("t.set", {"value": True})
    first = service.undo(r.undo_token)
    assert first.ok is True
    second = service.undo(r.undo_token)
    assert second.ok is False


def test_non_reversible_action_has_no_undo_token() -> None:
    handler = _NoUndoHandler()
    service = SidekickActionService()
    service.register(handler)
    r = service.invoke("n.write", {})
    assert r.ok is True
    assert r.undo_token is None  # nothing to undo


def test_service_assigns_unique_undo_tokens() -> None:
    handler = _ToggleHandler()
    service = SidekickActionService()
    service.register(handler)
    r1 = service.invoke("t.set", {"value": True})
    r2 = service.invoke("t.set", {"value": False})
    assert r1.undo_token != r2.undo_token


def test_undo_validates_input() -> None:
    service = SidekickActionService()
    with pytest.raises(ValueError, match="token"):
        service.undo("")
