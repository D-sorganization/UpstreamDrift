"""Canonical lifecycle kernel and state machine transitions.

Standard-library-only contract supporting job states, cancellation races,
and immutable result snapshots.
"""

from __future__ import annotations

from collections.abc import Mapping
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

__all__ = [
    "CANONICAL_LIFECYCLE_TRANSITIONS",
    "CANONICAL_LIFECYCLE_WORDS",
    "IllegalTransitionError",
    "LifecycleFailure",
    "LifecycleSnapshot",
    "LifecycleStatus",
    "TERMINAL_LIFECYCLE_STATES",
    "is_terminal",
    "transition",
]


class LifecycleStatus(StrEnum):
    """Canonical lifecycle states across services and computation kernels."""

    QUEUED = "queued"
    RUNNING = "running"
    CANCEL_REQUESTED = "cancel_requested"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


CANONICAL_LIFECYCLE_WORDS: frozenset[str] = frozenset(
    {"QUEUED", "RUNNING", "CANCEL_REQUESTED", "SUCCEEDED", "FAILED", "CANCELLED"}
)

TERMINAL_LIFECYCLE_STATES: frozenset[LifecycleStatus] = frozenset(
    {LifecycleStatus.SUCCEEDED, LifecycleStatus.FAILED, LifecycleStatus.CANCELLED}
)


class IllegalTransitionError(ValueError):
    """Raised when an attempted lifecycle transition is not permitted."""


_ls = LifecycleStatus
_empty: frozenset[LifecycleStatus] = frozenset()
CANONICAL_LIFECYCLE_TRANSITIONS: dict[LifecycleStatus, frozenset[LifecycleStatus]] = {
    _ls.QUEUED: frozenset(
        {_ls.RUNNING, _ls.CANCEL_REQUESTED, _ls.CANCELLED, _ls.FAILED}
    ),
    _ls.RUNNING: frozenset(
        {_ls.CANCEL_REQUESTED, _ls.SUCCEEDED, _ls.FAILED, _ls.CANCELLED}
    ),
    _ls.CANCEL_REQUESTED: frozenset({_ls.CANCELLED, _ls.SUCCEEDED, _ls.FAILED}),
    _ls.SUCCEEDED: _empty,
    _ls.FAILED: _empty,
    _ls.CANCELLED: _empty,
}


def transition(table: Mapping[Any, Any], current: Any, target_or_event: Any) -> Any:
    """Validate and execute a state transition against a transition table."""
    if not table:
        raise IllegalTransitionError(f"Empty transition table from {current!r}")
    if isinstance(next(iter(table)), tuple):
        key = (current, target_or_event)
        if key in table:
            return table[key]
        raise IllegalTransitionError(f"Illegal transition for {key!r}")
    if current not in table:
        raise IllegalTransitionError(f"State {current!r} not defined in table")
    dest = table[current]
    if isinstance(dest, Mapping) and target_or_event in dest:
        return dest[target_or_event]
    if isinstance(dest, (AbstractSet, tuple, list)) and target_or_event in dest:
        return target_or_event
    if dest == target_or_event:
        return target_or_event
    raise IllegalTransitionError(
        f"Illegal transition: {current!r} -> {target_or_event!r}"
    )


def is_terminal(state: Any) -> bool:
    """Return True if state is a terminal lifecycle state."""
    if isinstance(state, LifecycleStatus):
        return state in TERMINAL_LIFECYCLE_STATES
    if isinstance(state, str):
        if (norm := state.strip().upper()) in LifecycleStatus.__members__:
            return LifecycleStatus[norm] in TERMINAL_LIFECYCLE_STATES
        if (lower := state.strip().lower()) in LifecycleStatus._value2member_map_:
            return LifecycleStatus(lower) in TERMINAL_LIFECYCLE_STATES
    raise ValueError(f"Unknown lifecycle state: {state!r}")


@dataclass(frozen=True)
class LifecycleFailure:
    """Stable structured failure record."""

    code: str
    stage: str
    message: str

    def __post_init__(self) -> None:
        items = ((self.code, "code"), (self.stage, "stage"), (self.message, "message"))
        for val, name in items:
            if not isinstance(val, str) or not val.strip():
                raise ValueError(f"failure {name} must be a non-empty string")


@dataclass(frozen=True)
class LifecycleSnapshot:
    """Immutable validated point-in-time lifecycle status projection."""

    status: LifecycleStatus
    outcome: str | None = None
    failure: LifecycleFailure | None = None
    schema_version: int = 1

    def __post_init__(self) -> None:
        raw_status: Any = self.status
        if not isinstance(raw_status, LifecycleStatus):
            t = type(raw_status).__name__
            raise TypeError(f"status must be an instance of LifecycleStatus: {t}")
        v = self.schema_version
        if isinstance(v, bool) or not isinstance(v, int) or v < 1:
            raise ValueError("schema_version must be an integer >= 1")
        if self.outcome is not None and not isinstance(self.outcome, str):
            raise TypeError("outcome must be a string or None")

        if self.status is LifecycleStatus.FAILED:
            if self.failure is None:
                raise ValueError("failure required on FAILED")
            if not isinstance(self.failure, LifecycleFailure):
                raise TypeError("failure must be a LifecycleFailure")
        elif self.status is LifecycleStatus.SUCCEEDED and self.failure is not None:
            raise ValueError("SUCCEEDED snapshot cannot carry a failure")
        elif self.failure is not None and not isinstance(
            self.failure, LifecycleFailure
        ):
            raise TypeError("failure must be a LifecycleFailure or None")

    @property
    def done(self) -> bool:
        return is_terminal(self.status)

    @property
    def ok(self) -> bool:
        return self.status is LifecycleStatus.SUCCEEDED

    @property
    def cancelled(self) -> bool:
        return self.status is LifecycleStatus.CANCELLED

    @property
    def success(self) -> bool:
        return self.ok
