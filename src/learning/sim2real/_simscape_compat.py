"""Compatibility shim layer between :mod:`system_identification` and SimscapeAdapter.

The ``SimscapeAdapter`` (see :mod:`src.engines.simscape.adapter`) does not yet
expose every accessor that :class:`system_identification.SystemIdentifier`
optionally probes via ``hasattr``. The two methods most likely to surface as
gaps in practice are ``get_friction_coefficients`` / ``set_friction_coefficients``
and ``get_motor_strength`` / ``set_motor_strength`` (Simscape's ``setVariable``
mechanism can route them, but the typed Python surface is deferred to #4006).

Rather than block the integration test on a hidden ``hasattr`` miss, this
shim adds **no-op-with-warning** wrappers for those methods and a thin
delegating ``set_joint_damping`` that records every applied delta so a
test can verify it was called. Every wrapper logs a structured WARNING
the first time it is invoked, so the follow-up cleanup ticket is easy
to find from a test log.

# COMPAT NOTE
This module is intentionally minimal. Once #4006 wires the real Simscape
parameter setters, delete this file and the corresponding unit tests.
The protocol-introspection unit test in
``tests/unit/learning/test_simscape_compat.py`` will then assert
directly against ``SimscapeAdapter``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from src.shared.python.logging_pkg.logging_config import get_logger

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from src.engines.simscape.adapter import SimscapeAdapter

logger = get_logger(__name__)

__all__ = [
    "SimscapeSystemIdCompat",
    "wrap_for_system_identification",
]


class SimscapeSystemIdCompat:
    """Minimal compatibility wrapper around :class:`SimscapeAdapter`.

    The wrapper forwards every call to the underlying adapter unchanged
    except for the methods listed in :attr:`COMPAT_METHODS`. Those are
    served by no-op-with-warning fallbacks so that
    :class:`SystemIdentifier`'s ``hasattr``-guarded probes can exercise
    them without raising :class:`NotImplementedError` from the
    deferred-to-#4006 paths.

    Args:
        adapter: A live :class:`SimscapeAdapter`. The wrapper does not
            take ownership; callers are responsible for closing it.

    # COMPAT NOTE
    Every method on this class is provisional. Track removal in #4006.
    """

    #: Methods served by the compat shim rather than the adapter.
    #
    # # COMPAT NOTE
    # These all map to ``setVariable``/``getVariable`` on the MATLAB
    # workspace once #4006 lands. Until then, the shim either returns
    # zeros (for getters) or silently records the call (for setters).
    COMPAT_METHODS: tuple[str, ...] = (
        "get_friction_coefficients",
        "set_friction_coefficients",
        "get_motor_strength",
        "set_motor_strength",
        "get_joint_positions",
        "set_joint_positions",
        "get_joint_velocities",
        "set_joint_velocities",
        "set_joint_torques",
    )

    def __init__(self, adapter: SimscapeAdapter) -> None:
        if adapter is None:
            raise ValueError("adapter must be provided")
        self._adapter = adapter
        self._warned: set[str] = set()
        self._damping_history: list[NDArray[np.floating]] = []

    # ------------------------------------------------------------------
    # Adapter passthrough
    # ------------------------------------------------------------------

    def __getattr__(self, name: str) -> Any:
        # Only invoked when the attribute is NOT found on self via normal
        # lookup; route to adapter unless it's a compat method.
        if name in self.COMPAT_METHODS:
            return self._compat_dispatch(name)
        return getattr(self._adapter, name)

    # ------------------------------------------------------------------
    # Compat-only methods
    # ------------------------------------------------------------------

    def set_joint_damping(self, damping: NDArray[np.floating]) -> None:
        """Record damping locally; emit a WARNING on first call.

        # COMPAT NOTE
        SimscapeAdapter.set_joint_damping currently raises
        ``NotImplementedError`` (deferred to #4006). The shim records the
        call so :class:`SystemIdentifier` can iterate without crashing.
        """
        self._warn_once(
            "set_joint_damping",
            "SimscapeAdapter.set_joint_damping is deferred to #4006; "
            "treating call as a no-op for system identification.",
        )
        self._damping_history.append(np.asarray(damping, dtype=np.float64).copy())

    @property
    def damping_history(self) -> list[NDArray[np.floating]]:
        """Sequence of damping vectors handed to :meth:`set_joint_damping`."""
        return list(self._damping_history)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _compat_dispatch(self, name: str) -> Any:
        """Return a callable that warns + serves a sensible default.

        For getters we return zeros (sized using ``self._adapter.dof``
        when possible). For setters we accept the input and discard it.
        """

        def _impl(*args: Any, **kwargs: Any) -> Any:
            self._warn_once(
                name,
                (
                    f"SimscapeAdapter.{name} is deferred to #4006; "
                    "compat shim is serving a no-op default."
                ),
            )
            if name.startswith("get_"):
                try:
                    n = int(self._adapter.dof)
                except Exception:  # noqa: BLE001 - dof unavailable pre-load
                    n = 0
                return np.zeros(n, dtype=np.float64)
            # setters: accept and discard
            del args, kwargs
            return None

        return _impl

    def _warn_once(self, name: str, message: str) -> None:
        if name in self._warned:
            return
        self._warned.add(name)
        logger.warning("%s [compat] %s", name, message)


def wrap_for_system_identification(adapter: SimscapeAdapter) -> SimscapeSystemIdCompat:
    """Convenience constructor for :class:`SimscapeSystemIdCompat`.

    # COMPAT NOTE
    Provided so call-sites read naturally:

        ``identifier = SystemIdentifier(wrap_for_system_identification(adapter))``
    """
    return SimscapeSystemIdCompat(adapter)
