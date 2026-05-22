"""Audited action-dispatch service for the Sidekick agent layer.

Epic #5967 / sub-issue #5971 (S2).

This module owns the single dispatch path that every agentic action flows
through. Adapters (subtab, host, feature-catalog) implement one Protocol
(:class:`SidekickActionHandler`); the planner sees only the facade
(:class:`SidekickActionService`).

Why one service, not many: per-adapter dispatchers would each grow their
own validation, error mapping, audit, and dry-run logic — Law-of-Demeter
violations and the kind of fork-then-diverge maintenance burden that the
2026-05-21 review (#5907) flagged repeatedly.

Design contracts:

* **DbC.** Descriptors validate themselves; the service refuses duplicate
  ``action_id``s at registration time (invariant: ``list_actions()``
  contains no duplicates).
* **LOD.** Planner code calls ``service.invoke(action_id, params)`` and
  never reaches into a handler. Audit sinks see a ``RecordedCall`` value;
  they cannot see the handler instance.
* **DRY.** JSON-Schema validation is a single private helper, reused by
  every action. The dry-run flag is owned here, never duplicated in
  adapters.
* **Headless-safe.** Zero PyQt6 imports.
* **Error handling.** Per ADR-0016: handler-raised exceptions are caught
  via :func:`~core.process_safety.narrow_catch` and translated into
  :class:`ActionResult` (no bare ``except Exception``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal, Protocol, runtime_checkable
from collections.abc import Callable, Mapping, Sequence

from src.shared.python.core.contracts.exceptions import StateError

logger = logging.getLogger(__name__)

UTC = timezone.utc  # noqa: UP017

__all__ = [
    "ActionDescriptor",
    "ActionResult",
    "AuditSink",
    "RecordedCall",
    "SideEffect",
    "SidekickActionHandler",
    "SidekickActionService",
]


SideEffect = Literal["read", "write", "destructive"]
_VALID_SIDE_EFFECTS: frozenset[str] = frozenset({"read", "write", "destructive"})


# ---------------------------------------------------------------------------
# Public data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ActionDescriptor:
    """Self-describing record for one agentic action.

    Attributes:
        action_id: Fully qualified id of the form ``"<namespace>.<verb>"``.
        summary: One-sentence human description.
        params_schema: JSON Schema (subset) for the ``params`` mapping
            passed to :meth:`SidekickActionService.invoke`.
        side_effects: ``"read"``, ``"write"``, or ``"destructive"``.
            Drives the chat-side confirmation UX (S8).
        reversible: ``True`` if the adapter exposes an ``undo`` hook
            for this action (consumed by the undo subsystem in S6).
    """

    action_id: str
    summary: str
    params_schema: Mapping[str, Any]
    side_effects: SideEffect
    reversible: bool = False

    def __post_init__(self) -> None:
        # DbC: pin every invariant the rest of the system relies on.
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
    """Outcome of one invocation. Either successful or carrying an error.

    Attributes:
        ok: ``True`` if the action completed successfully.
        value: Action-specific payload. May be ``None``.
        error: Human-readable error message. Must be present iff ``ok``
            is ``False``.
        undo_token: Opaque token an adapter can later use to reverse this
            action. Optional; only set for reversible actions that have
            something to undo.
        metadata: Open-ended bag for diagnostics (timings, dry-run
            previews, etc.). Must not be used to smuggle real return
            values — those go in ``value``.
    """

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
    """One entry passed to an audit sink. Immutable.

    The audit sink receives the descriptor and result but NOT the handler
    instance — that's LOD by construction.
    """

    timestamp: datetime
    action_id: str
    params: Mapping[str, Any]
    descriptor: ActionDescriptor | None
    result: ActionResult
    dry_run: bool


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------


@runtime_checkable
class SidekickActionHandler(Protocol):
    """Contract implemented by every action adapter.

    Implementations register a stable ``namespace`` (used only for
    diagnostics; the per-action ``action_id`` is the actual key).
    """

    namespace: str

    def describe(self) -> Sequence[ActionDescriptor]:
        """Return the actions this handler publishes."""
        ...

    def invoke(self, action_id: str, params: Mapping[str, Any]) -> ActionResult:
        """Run one action. Implementations should never raise on user
        errors — translate to ``ActionResult(ok=False, error=...)``."""
        ...


AuditSink = Callable[[RecordedCall], None]
"""Audit sink signature. Sinks must be cheap and non-raising."""


def _noop_audit_sink(call: RecordedCall) -> None:  # pragma: no cover - trivial
    return None


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class SidekickActionService:
    """Registry + dispatcher + audit choke-point.

    The service does four things and no more:

    1. Register handlers, refusing duplicate ``action_id``s.
    2. Validate ``params`` against each action's JSON Schema.
    3. Dispatch to the owning handler (or short-circuit on ``dry_run``).
    4. Hand a :class:`RecordedCall` to the audit sink, on success or
       failure.

    Anything more complex (undo, access policy, planner integration) lives
    in dedicated modules and composes with this service.
    """

    def __init__(self, *, audit_sink: AuditSink | None = None) -> None:
        self._handlers: dict[str, SidekickActionHandler] = {}
        self._descriptors: dict[str, ActionDescriptor] = {}
        self._audit_sink: AuditSink = audit_sink or _noop_audit_sink

    # ---- Registration ----------------------------------------------------

    def register(self, handler: SidekickActionHandler) -> None:
        """Register every action published by ``handler``.

        Precondition: ``handler`` satisfies :class:`SidekickActionHandler`.
        Postcondition: every descriptor's ``action_id`` is reachable via
        :meth:`invoke`.

        Raises:
            TypeError: If ``handler`` does not satisfy the Protocol.
            ValueError: If any ``action_id`` collides with one already
                registered.
        """
        if not isinstance(handler, SidekickActionHandler):
            raise TypeError(
                f"handler must satisfy SidekickActionHandler, got {type(handler).__name__}"
            )
        new_descs = list(handler.describe())
        # Reject duplicates atomically — fail before mutating state.
        for desc in new_descs:
            if desc.action_id in self._descriptors:
                raise ValueError(
                    f"duplicate action_id {desc.action_id!r}; already registered"
                )
        for desc in new_descs:
            self._descriptors[desc.action_id] = desc
            self._handlers[desc.action_id] = handler

    # ---- Discovery -------------------------------------------------------

    def list_actions(self) -> tuple[ActionDescriptor, ...]:
        """Return every registered descriptor sorted by ``action_id``."""
        return tuple(self._descriptors[k] for k in sorted(self._descriptors))

    # ---- Dispatch --------------------------------------------------------

    def invoke(
        self,
        action_id: str,
        params: Mapping[str, Any],
        *,
        dry_run: bool = False,
    ) -> ActionResult:
        """Validate and dispatch one action.

        Args:
            action_id: Must be a registered descriptor's id.
            params: Mapping validated against the descriptor's
                ``params_schema``.
            dry_run: If ``True``, skip the handler entirely and return a
                synthetic success result whose ``metadata['dry_run']`` is
                the supplied params. Audit is still recorded.

        Returns:
            An :class:`ActionResult`. Errors at any layer (unknown id,
            schema failure, handler exception) are returned as
            ``ok=False`` results — this method never raises on user input.
        """
        descriptor = self._descriptors.get(action_id)
        if descriptor is None:
            result = ActionResult(ok=False, error=f"unknown action_id: {action_id!r}")
            self._record(action_id, params, None, result, dry_run)
            return result

        schema_error = _validate_against_schema(params, descriptor.params_schema)
        if schema_error is not None:
            result = ActionResult(
                ok=False, error=f"params validation failed: {schema_error}"
            )
            self._record(action_id, params, descriptor, result, dry_run)
            return result

        if dry_run:
            result = ActionResult(
                ok=True,
                value=None,
                metadata={"dry_run": dict(params), "would_call": action_id},
            )
            self._record(action_id, params, descriptor, result, dry_run)
            return result

        result = self._safe_invoke(action_id, params)
        self._record(action_id, params, descriptor, result, dry_run)
        return result

    # ---- Internals -------------------------------------------------------

    def _safe_invoke(self, action_id: str, params: Mapping[str, Any]) -> ActionResult:
        """Call the handler, translating known exceptions to error results.

        Per ADR-0016 we catch only the narrow set of exceptions a
        well-behaved handler can plausibly raise. Anything else
        (:class:`KeyboardInterrupt`, :class:`SystemExit`, ...) propagates.

        We use an explicit try/except chain rather than
        :func:`~src.shared.python.core.process_safety.narrow_catch` because
        we need the exception value to produce the user-facing error
        string — ``narrow_catch`` is a suppress-and-log helper, the wrong
        tool for translate-and-return.
        """
        handler = self._handlers[action_id]
        try:
            outcome = handler.invoke(action_id, params)
        except StateError as exc:
            logger.warning("action %s raised StateError: %s", action_id, exc)
            return ActionResult(ok=False, error=f"state error: {exc}")
        except ValueError as exc:
            logger.warning("action %s raised ValueError: %s", action_id, exc)
            return ActionResult(ok=False, error=f"value error: {exc}")
        except (RuntimeError, LookupError) as exc:
            logger.warning(
                "action %s raised %s: %s", action_id, type(exc).__name__, exc
            )
            return ActionResult(ok=False, error=f"{type(exc).__name__}: {exc}")
        if not isinstance(outcome, ActionResult):
            return ActionResult(
                ok=False,
                error=(
                    f"handler for {action_id!r} returned "
                    f"{type(outcome).__name__}, expected ActionResult"
                ),
            )
        return outcome

    def _record(
        self,
        action_id: str,
        params: Mapping[str, Any],
        descriptor: ActionDescriptor | None,
        result: ActionResult,
        dry_run: bool,
    ) -> None:
        """Hand a RecordedCall to the audit sink. Sink failures are logged
        but never propagated — auditing is observability, not gating."""
        call = RecordedCall(
            timestamp=datetime.now(UTC),
            action_id=action_id,
            params=dict(params),
            descriptor=descriptor,
            result=result,
            dry_run=dry_run,
        )
        try:
            self._audit_sink(call)
        except Exception:  # noqa: BLE001 - audit sink must never break dispatch
            logger.exception("audit sink failed for action %s", action_id)


# ---------------------------------------------------------------------------
# Minimal JSON-Schema validator
# ---------------------------------------------------------------------------


def _validate_against_schema(
    params: Mapping[str, Any], schema: Mapping[str, Any]
) -> str | None:
    """Return ``None`` on success or a human-readable error string.

    A full draft-7 validator is out of scope; we cover the subset that
    every adapter in this epic uses:

    * ``type: object`` with optional ``properties`` and ``required``
    * per-property primitive types: ``string``, ``integer``, ``number``,
      ``boolean``, ``object``, ``array``

    Adding a third-party JSON Schema library is deliberately deferred —
    when an adapter genuinely needs richer keywords we'll lift this into
    a shared helper and pin a library version once across the fleet.
    """
    if not isinstance(params, Mapping):
        return f"params must be a Mapping, got {type(params).__name__}"
    if schema.get("type") != "object":
        return None  # nothing we can validate; accept

    properties = schema.get("properties", {}) or {}
    required = schema.get("required", []) or []

    for key in required:
        if key not in params:
            return f"missing required property: {key!r}"

    for key, value in params.items():
        prop_schema = properties.get(key)
        if prop_schema is None:
            continue  # additional properties allowed
        prop_type = prop_schema.get("type")
        if prop_type is None:
            continue
        if not _type_matches(value, prop_type):
            return (
                f"property {key!r} expected type {prop_type!r}, "
                f"got {type(value).__name__}"
            )
    return None


_TYPE_MAP: Mapping[str, tuple[type, ...]] = {
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "object": (Mapping,),
    "array": (list, tuple),
    "null": (type(None),),
}


def _type_matches(value: Any, type_name: str) -> bool:
    """Return ``True`` if ``value`` satisfies the JSON Schema primitive
    ``type_name``. ``bool`` is rejected for numeric types because
    Python's ``bool`` is a subclass of ``int`` and conflating the two
    causes silent action-misrouting in our experience."""
    if type_name == "integer" and isinstance(value, bool):
        return False
    if type_name == "number" and isinstance(value, bool):
        return False
    expected = _TYPE_MAP.get(type_name)
    if expected is None:
        return True
    return isinstance(value, expected)
