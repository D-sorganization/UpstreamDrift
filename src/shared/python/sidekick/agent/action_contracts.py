"""Action-layer contracts for Sidekick agent dispatch."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal, Protocol, runtime_checkable

SideEffect = Literal["read", "write", "destructive"]
_VALID_SIDE_EFFECTS: frozenset[str] = frozenset({"read", "write", "destructive"})


@dataclass(frozen=True, slots=True)
class ActionDescriptor:
    """Self-describing record for one agentic action."""

    action_id: str
    summary: str
    params_schema: Mapping[str, Any]
    side_effects: SideEffect
    reversible: bool = False

    def __post_init__(self) -> None:
        if not self.action_id or "." not in self.action_id:
            raise ValueError(
                f"action_id must be '<namespace>.<verb>'; got {self.action_id!r}"
            )
        if not self.summary:
            raise ValueError("summary must be non-empty")
        if not isinstance(self.params_schema, Mapping):
            raise ValueError("params_schema must be a Mapping")
        if "type" not in self.params_schema:
            raise ValueError(
                "params_schema must be JSON-Schema-shaped (missing 'type')"
            )
        if self.side_effects not in _VALID_SIDE_EFFECTS:
            raise ValueError(
                f"side_effects={self.side_effects!r} not in "
                f"{sorted(_VALID_SIDE_EFFECTS)}"
            )


@dataclass(frozen=True, slots=True)
class ActionResult:
    """Outcome of one invocation. Either successful or carrying an error."""

    ok: bool
    value: Any = None
    error: str | None = None
    undo_token: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.ok and self.error is not None:
            raise ValueError("ok=True forbids setting error")
        if not self.ok and not self.error:
            raise ValueError("ok=False requires a non-empty error message")


@dataclass(frozen=True, slots=True)
class RecordedCall:
    """One immutable entry passed to an audit sink."""

    timestamp: datetime
    action_id: str
    params: Mapping[str, Any]
    descriptor: ActionDescriptor | None
    result: ActionResult
    dry_run: bool


@runtime_checkable
class SidekickActionHandler(Protocol):
    """Contract implemented by every action adapter."""

    namespace: str

    def describe(self) -> Sequence[ActionDescriptor]:
        """Return the actions this handler publishes."""
        ...

    def invoke(self, action_id: str, params: Mapping[str, Any]) -> ActionResult:
        """Run one action and translate user errors to ``ActionResult``."""
        ...


AuditSink = Callable[[RecordedCall], None]
"""Audit sink signature. Sinks must be cheap and non-raising."""

ActionDispatcher = Callable[[Callable[[], ActionResult]], ActionResult]
"""Runs an action thunk through a host-provided dispatch boundary."""
