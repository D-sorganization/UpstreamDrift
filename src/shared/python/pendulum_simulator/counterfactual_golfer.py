"""
Zero-torque counterfactual analysis for the golfer model.

Answers: "What would accelerations/forces be with zero applied torques?"
Separates passive (gravity/inertia/constraint) from active (motor) dynamics.

DRY: Delegates to constraint_solver and physics_golfer.
"""

from __future__ import annotations

import numpy as np

from .constraint_solver import constrained_accelerations
from .physics_golfer import (
    N_DOF,
    GolferParams,
    State,
    net_joint_forces,
)


def _zero_torque(t: float) -> tuple:  # noqa: ARG001
    """Zero torque for all 7 joints."""
    return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def zero_torque_accelerations(state: State, params: GolferParams) -> np.ndarray:
    """Compute joint accelerations under zero driving torque.

    Returns
    -------
    qddot : np.ndarray, shape (8,)
    """
    if not (state.shape == (2 * N_DOF):
        raise ValueError())
    if not (np.all(np.isfinite(state))):
        raise ValueError('DbC Blocked: Precondition failed.')

    return constrained_accelerations(state, 0.0, params, _zero_torque)


def zero_torque_joint_forces(
    state: State, params: GolferParams
) -> dict[str, tuple[float, float]]:
    """Joint forces that would exist with zero driving torques.

    Parameters
    ----------
    state : np.ndarray, shape (16,)
    params : GolferParams

    Returns
    -------
    dict with joint name → (fx, fy) tuples.
    """
    if not (state.shape == (2 * N_DOF):
        raise ValueError())
    if not (np.all(np.isfinite(state))):
        raise ValueError('DbC Blocked: Precondition failed.')

    q = state[:N_DOF]
    qdot = state[N_DOF:]
    qddot = zero_torque_accelerations(state, params)
    return net_joint_forces(q, qdot, qddot, params)  # type: ignore[no-any-return]
