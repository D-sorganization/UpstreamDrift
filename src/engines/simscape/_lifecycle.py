"""State machine helper for :class:`SimscapeAdapter`.

The adapter's lifecycle is::

    UNINITIALIZED  --load_from_path-->  LOADED
    LOADED         --reset/step/...-->  RUNNING
    RUNNING        --step/...-------->  RUNNING
    {LOADED, RUNNING}  --close-------->  STOPPED
    UNINITIALIZED  --close----------->  STOPPED  (idempotent no-op semantics)

Transitions outside of this graph raise
:class:`src.engines.simscape._errors.SimscapeStateError`. The state-machine
logic lives in this small module, not on the adapter itself, so the adapter
keeps a low Law-of-Demeter footprint and can be unit-tested in isolation.
"""

from __future__ import annotations

from enum import Enum

from src.engines.simscape._errors import SimscapeStateError

__all__ = [
    "AdapterState",
    "LifecycleGuard",
]


class AdapterState(str, Enum):
    """Discrete lifecycle states of :class:`SimscapeAdapter`."""

    UNINITIALIZED = "uninitialized"
    LOADED = "loaded"
    RUNNING = "running"
    STOPPED = "stopped"


# ---------------------------------------------------------------------------
# Allowed transitions. Any (from_state, to_state) pair not in this set is
# rejected by ``LifecycleGuard.transition``.
# ---------------------------------------------------------------------------
_ALLOWED_TRANSITIONS: frozenset[tuple[AdapterState, AdapterState]] = frozenset(
    {
        # Load
        (AdapterState.UNINITIALIZED, AdapterState.LOADED),
        # Run
        (AdapterState.LOADED, AdapterState.RUNNING),
        (AdapterState.RUNNING, AdapterState.RUNNING),
        # Reset (RUNNING -> LOADED is the natural "back to t=0" semantics)
        (AdapterState.RUNNING, AdapterState.LOADED),
        (AdapterState.LOADED, AdapterState.LOADED),
        # Close (idempotent: any state -> STOPPED)
        (AdapterState.UNINITIALIZED, AdapterState.STOPPED),
        (AdapterState.LOADED, AdapterState.STOPPED),
        (AdapterState.RUNNING, AdapterState.STOPPED),
        (AdapterState.STOPPED, AdapterState.STOPPED),
    }
)


class LifecycleGuard:
    """Tiny state-machine helper used by :class:`SimscapeAdapter`.

    The guard tracks the current state, validates transitions against the
    fixed transition table, and exposes ``require`` for guarding operations
    that must be invoked in a particular state (e.g. ``step`` requires
    ``LOADED`` or ``RUNNING``).

    The guard does **not** know about the adapter; it is intentionally
    decoupled so its tests can run with no MATLAB and no adapter dependency.
    """

    def __init__(self) -> None:
        self._state: AdapterState = AdapterState.UNINITIALIZED

    @property
    def state(self) -> AdapterState:
        """Return the current lifecycle state."""
        return self._state

    def is_loaded(self) -> bool:
        """Return ``True`` if a model has been loaded and the engine is live."""
        return self._state in (AdapterState.LOADED, AdapterState.RUNNING)

    def is_stopped(self) -> bool:
        """Return ``True`` if the engine has been shut down."""
        return self._state == AdapterState.STOPPED

    # ------------------------------------------------------------------
    # Transition primitives
    # ------------------------------------------------------------------

    def transition(self, target: AdapterState, *, operation: str) -> None:
        """Move the state machine to ``target`` if the transition is allowed.

        Args:
            target: Desired next state.
            operation: Human-readable name of the operation that triggered
                this transition; used in error messages.

        Raises:
            SimscapeStateError: If the (current, target) pair is not in the
                allowed transition table.
        """
        if (self._state, target) not in _ALLOWED_TRANSITIONS:
            raise SimscapeStateError(
                operation,
                current_state=self._state.value,
                required_state=target.value,
            )
        self._state = target

    def require(
        self,
        *allowed: AdapterState,
        operation: str,
    ) -> None:
        """Assert the current state is one of ``allowed``.

        Used by adapter methods that are read-only with respect to the
        lifecycle but still need a particular state (e.g. ``get_state``
        requires ``LOADED`` or ``RUNNING``).

        Args:
            *allowed: Acceptable states for ``operation``.
            operation: Human-readable operation name for the error message.

        Raises:
            SimscapeStateError: If the current state is not in ``allowed``.
        """
        if self._state not in allowed:
            required = "|".join(s.value for s in allowed)
            raise SimscapeStateError(
                operation,
                current_state=self._state.value,
                required_state=required,
            )
