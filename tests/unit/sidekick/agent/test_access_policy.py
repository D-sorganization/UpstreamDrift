"""Tests for sidekick.agent.access_policy (epic #5967 / S6 / #5975)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pytest

from sidekick.agent.access_policy import (
    PolicyDecision,
    SidekickActionPolicy,
)
from sidekick.agent.action_service import (
    ActionDescriptor,
    ActionResult,
    SidekickActionService,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Default policy: read OK, others denied
# ---------------------------------------------------------------------------


def _desc(action_id: str, side_effects: str = "read") -> ActionDescriptor:
    return ActionDescriptor(
        action_id=action_id,
        summary="s",
        params_schema={"type": "object"},
        side_effects=side_effects,  # type: ignore[arg-type]
        reversible=False,
    )


def test_default_policy_allows_read_actions() -> None:
    policy = SidekickActionPolicy()
    decision = policy.decide(_desc("x.read", "read"), params={})
    assert decision.allowed
    assert decision.reason == "default-allow-read"


def test_default_policy_denies_write_actions() -> None:
    policy = SidekickActionPolicy()
    decision = policy.decide(_desc("x.write", "write"), params={})
    assert not decision.allowed


def test_default_policy_denies_destructive_actions() -> None:
    policy = SidekickActionPolicy()
    decision = policy.decide(_desc("x.boom", "destructive"), params={})
    assert not decision.allowed


# ---------------------------------------------------------------------------
# Allowlist
# ---------------------------------------------------------------------------


def test_explicit_allowlist_overrides_default_deny() -> None:
    policy = SidekickActionPolicy(allow_write={"x.write"})
    assert policy.decide(_desc("x.write", "write"), params={}).allowed


def test_destructive_requires_confirmation_even_when_allowed() -> None:
    policy = SidekickActionPolicy(allow_destructive={"x.boom"})
    decision = policy.decide(_desc("x.boom", "destructive"), params={})
    # Allowed in principle but not yet confirmed → still denied.
    assert not decision.allowed
    assert "confirm" in decision.reason.lower()


def test_destructive_with_confirmation_passes() -> None:
    policy = SidekickActionPolicy(allow_destructive={"x.boom"})
    decision = policy.decide(
        _desc("x.boom", "destructive"), params={"_confirmed": True}
    )
    assert decision.allowed


def test_unknown_action_under_destructive_not_in_allowlist() -> None:
    policy = SidekickActionPolicy()
    decision = policy.decide(_desc("y.boom", "destructive"), params={})
    assert not decision.allowed
    assert "destructive" in decision.reason.lower()


# ---------------------------------------------------------------------------
# Permissive preset
# ---------------------------------------------------------------------------


def test_permissive_preset_allows_everything_after_confirmation() -> None:
    policy = SidekickActionPolicy.permissive()
    # write: allowed
    assert policy.decide(_desc("x.w", "write"), params={}).allowed
    # destructive without confirmation: still denied
    assert not policy.decide(_desc("x.b", "destructive"), params={}).allowed
    # destructive with confirmation: allowed
    assert policy.decide(
        _desc("x.b", "destructive"), params={"_confirmed": True}
    ).allowed


# ---------------------------------------------------------------------------
# Integration with SidekickActionService
# ---------------------------------------------------------------------------


class _Handler:
    namespace = "t"

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def describe(self) -> Sequence[ActionDescriptor]:
        return (
            _desc("t.read", "read"),
            _desc("t.write", "write"),
        )

    def invoke(self, action_id: str, params: Mapping[str, Any]) -> ActionResult:
        self.calls.append((action_id, dict(params)))
        return ActionResult(ok=True)


def test_service_consults_policy_before_dispatch() -> None:
    handler = _Handler()
    service = SidekickActionService(policy=SidekickActionPolicy())
    service.register(handler)
    # read OK
    r1 = service.invoke("t.read", {})
    assert r1.ok is True
    # write denied
    r2 = service.invoke("t.write", {})
    assert r2.ok is False
    assert (
        "forbidden" in (r2.error or "").lower() or "denied" in (r2.error or "").lower()
    )
    # Critical: write handler must NOT have been called.
    assert all(c[0] != "t.write" for c in handler.calls)


def test_service_without_policy_dispatches_freely() -> None:
    handler = _Handler()
    service = SidekickActionService()  # no policy
    service.register(handler)
    assert service.invoke("t.write", {}).ok is True


# ---------------------------------------------------------------------------
# Policy decision dataclass DbC
# ---------------------------------------------------------------------------


def test_policy_decision_requires_reason() -> None:
    with pytest.raises(ValueError, match="reason"):
        PolicyDecision(allowed=True, reason="")
