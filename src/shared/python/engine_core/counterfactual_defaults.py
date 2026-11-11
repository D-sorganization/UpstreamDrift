"""Default counterfactual accelerations for engines without native hooks (#14943)."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol, TypeVar

import numpy as np

if TYPE_CHECKING:

    class _MassGravityEngine(Protocol):
        def compute_mass_matrix(self) -> np.ndarray: ...
        def compute_gravity_forces(self) -> np.ndarray: ...
        def compute_control_acceleration(self, tau: np.ndarray) -> np.ndarray: ...


_T = TypeVar("_T")


def _configuration_snapshot(engine: object) -> tuple[np.ndarray, np.ndarray] | None:
    """Return a copy of ``(q, v)`` when the engine exposes state readers."""
    get_positions = getattr(engine, "get_joint_positions", None)
    get_velocities = getattr(engine, "get_joint_velocities", None)
    if callable(get_positions) and callable(get_velocities):
        return (
            np.asarray(get_positions(), dtype=float).copy(),
            np.asarray(get_velocities(), dtype=float).copy(),
        )

    get_state = getattr(engine, "get_state", None)
    if not callable(get_state):
        return None

    state = get_state()
    if state is None:
        return None
    if isinstance(state, tuple) and len(state) >= 2:
        return (
            np.asarray(state[0], dtype=float).copy(),
            np.asarray(state[1], dtype=float).copy(),
        )

    q = getattr(state, "q", None)
    v = getattr(state, "v", None)
    if q is not None and v is not None:
        return (np.asarray(q, dtype=float).copy(), np.asarray(v, dtype=float).copy())
    return None


def _apply_configuration(engine: object, q: np.ndarray, v: np.ndarray | None) -> None:
    """Apply generalized coordinates; requires a known state writer on *engine*."""
    q_arr = np.asarray(q, dtype=float)
    set_state = getattr(engine, "set_state", None)
    if callable(set_state):
        if v is None:
            snap = _configuration_snapshot(engine)
            if snap is None:
                raise NotImplementedError(
                    "set_state(q, v) needs velocity snapshot or explicit v"
                )
            v = np.zeros_like(snap[1])
        try:
            set_state(q_arr, np.asarray(v, dtype=float))
        except TypeError:
            # EngineState-only engines are not supported by the default hook.
            raise NotImplementedError(
                "Engine set_state does not accept (q, v); "
                "override compute_control_acceleration_at_state"
            ) from None
        return

    set_positions = getattr(engine, "set_joint_positions", None)
    if not callable(set_positions):
        raise NotImplementedError(
            "Engine has no set_state(q, v) or set_joint_positions; "
            "override compute_control_acceleration_at_state"
        )
    set_positions(q_arr)
    if v is not None:
        set_velocities = getattr(engine, "set_joint_velocities", None)
        if callable(set_velocities):
            set_velocities(np.asarray(v, dtype=float))


def _evaluate_at_fixed_configuration(
    engine: object, q: np.ndarray, evaluate: Callable[[], _T]
) -> _T:
    """Run *evaluate* after applying counterfactual ``q`` with ``v = 0``, then restore."""
    q_arr = np.asarray(q, dtype=float)
    snapshot = _configuration_snapshot(engine)
    if snapshot is None:
        raise NotImplementedError(
            "Engine does not expose readable joint/state configuration; "
            "override compute_control_acceleration_at_state"
        )
    q_prev, v_prev = snapshot
    v_zero = np.zeros_like(v_prev) if v_prev.size else np.zeros_like(q_arr)
    try:
        _apply_configuration(engine, q_arr, v_zero)
        return evaluate()
    finally:
        _apply_configuration(engine, q_prev, v_prev)


def default_gravity_only_acceleration(
    engine: _MassGravityEngine, q: np.ndarray
) -> np.ndarray:
    """Legacy gravity-only diagnostic at fixed q with v = u = 0 (not SPEC ZVCF)."""

    def _gravity_only() -> np.ndarray:
        mass = engine.compute_mass_matrix()
        gravity = engine.compute_gravity_forces()
        return np.linalg.solve(mass, -gravity)

    return _evaluate_at_fixed_configuration(engine, q, _gravity_only)


def default_control_acceleration_at_state(
    engine: _MassGravityEngine, q: np.ndarray, tau: np.ndarray
) -> np.ndarray:
    """SPEC ZVCF control component M(q)⁻¹ τ at fixed state (per §4)."""
    tau_v = np.asarray(tau, dtype=float).reshape(-1)

    def _control() -> np.ndarray:
        return engine.compute_control_acceleration(tau_v)

    return _evaluate_at_fixed_configuration(engine, q, _control)


def deprecated_compute_zvcf_shim(
    engine: _MassGravityEngine, q: np.ndarray
) -> np.ndarray:
    """One-release shim: old compute_zvcf → gravity-only diagnostic."""
    warnings.warn(
        "compute_zvcf(q) names a legacy gravity-only diagnostic; "
        "SPEC §4 ZVCF is the control component M(q)⁻¹τ — use "
        "compute_control_acceleration_at_state(q, tau) or "
        "compute_gravity_only_acceleration(q) explicitly.",
        DeprecationWarning,
        stacklevel=3,
    )
    return default_gravity_only_acceleration(engine, q)
