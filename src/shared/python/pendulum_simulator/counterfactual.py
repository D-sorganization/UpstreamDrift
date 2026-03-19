"""
Zero-torque counterfactual joint force computation.

Answers: "What would the joint forces be at this instant if all applied
driving torques were zero?" — i.e., the passive / drift component of the
motion due purely to gravity, inertia, and current velocities.

Design by Contract
------------------
Preconditions:
  - state.shape == (4,) for double, (6,) for triple with all finite values.
  - params has positive masses, lengths, and non-negative g.
Postconditions:
  - Returns a dict with (fx, fy) finite float tuples at each joint.

DRY
---
Delegates all physics to the existing physics.py / physics_triple.py
modules.  No new physics is introduced — only the torque term is zeroed.
"""

from __future__ import annotations

import numpy as np

from .physics import (
    PendulumParams,
    coriolis_vector,
    friction_torque_vector,
    gravity_vector,
    mass_matrix,
    net_joint_forces,
)
from .physics_triple import (
    TriplePendulumParams,
)
from .physics_triple import (
    coriolis_vector as coriolis_vector_triple,
)
from .physics_triple import (
    gravity_vector as gravity_vector_triple,
)
from .physics_triple import (
    mass_matrix as mass_matrix_triple,
)
from .physics_triple import (
    net_joint_forces as net_joint_forces_triple,
)

# Note: physics_triple does not model joint friction (no friction_torque_vector)


# ---------------------------------------------------------------------------
# Double pendulum
# ---------------------------------------------------------------------------


def _zero_torque_qddot_double(state: np.ndarray, params: PendulumParams) -> np.ndarray:
    """Compute angular accel under zero driving torque (passive dynamics).

    Passive RHS: M * q̈ = τ_friction - C(q,q̇)q̇ - G(q)
    Driving torques τ_drive are explicitly set to zero.

    Preconditions
    -------------
    state.shape == (4,) and all values finite.

    Returns
    -------
    qddot : np.ndarray  shape (2,)  — [ddtheta1, ddphi]
    """
    assert state.shape == (4,), f"Double state must be shape (4,), got {state.shape}"
    assert np.all(np.isfinite(state)), "State contains non-finite values"

    _theta1, phi, dtheta1, dphi = state

    M = mass_matrix(phi, params)
    C = coriolis_vector(phi, dtheta1, dphi, params)
    G = gravity_vector(_theta1, phi, params)
    tau_friction = friction_torque_vector(dtheta1, dphi, params)

    # Zero driving torque: rhs = 0 + tau_friction - C - G
    rhs = tau_friction - C - G
    qddot: np.ndarray = np.linalg.solve(M, rhs)

    assert np.all(np.isfinite(qddot)), "Zero-torque qddot is non-finite"
    return qddot


def zero_torque_joint_forces_double(
    state: np.ndarray, params: PendulumParams
) -> dict[str, tuple[float, float]]:
    """Joint forces that would exist if all driving torques were zero.

    This is the *passive drift* component: gravity + Coriolis + friction,
    without any applied motor torques.

    Parameters
    ----------
    state : np.ndarray, shape (4,)
        Current state [theta1, phi, dtheta1, dphi] in radians / rad·s⁻¹.
    params : PendulumParams
        Physical parameters (masses, lengths, gravity).

    Returns
    -------
    dict with keys 'shoulder', 'wrist' — each a (fx, fy) tuple in Newtons.

    Preconditions
    -------------
    state.shape == (4,) with all finite values.

    Postconditions
    --------------
    All returned force components are finite.
    """
    assert state.shape == (4,), f"Expected state shape (4,), got {state.shape}"
    assert np.all(np.isfinite(state)), "State must be finite"

    qddot = _zero_torque_qddot_double(state, params)
    forces = net_joint_forces(state, qddot, params)

    # Postcondition: all outputs finite
    for key, (fx, fy) in forces.items():
        assert np.isfinite(fx) and np.isfinite(
            fy
        ), f"Non-finite zero-torque force at {key}: ({fx}, {fy})"
    return forces


# ---------------------------------------------------------------------------
# Triple pendulum
# ---------------------------------------------------------------------------


def _zero_torque_qddot_triple(
    state: np.ndarray, params: TriplePendulumParams
) -> np.ndarray:
    """Compute angular accel under zero driving torque for triple pendulum.

    Preconditions
    -------------
    state.shape == (6,) and all values finite.

    Returns
    -------
    qddot : np.ndarray  shape (3,)  — [ddtheta1, ddphi1, ddphi2]
    """
    assert state.shape == (6,), f"Triple state must be shape (6,), got {state.shape}"
    assert np.all(np.isfinite(state)), "State contains non-finite values"

    _theta1, phi1, phi2, dtheta1, dphi1, dphi2 = state

    M = mass_matrix_triple(phi1, phi2, params)
    C = coriolis_vector_triple(phi1, phi2, dtheta1, dphi1, dphi2, params)
    G = gravity_vector_triple(_theta1, phi1, phi2, params)
    # physics_triple has no friction model → passive rhs = -C - G only
    rhs = -C - G
    qddot: np.ndarray = np.linalg.solve(M, rhs)

    assert np.all(np.isfinite(qddot)), "Zero-torque qddot (triple) is non-finite"
    return qddot


def zero_torque_joint_forces_triple(
    state: np.ndarray, params: TriplePendulumParams
) -> dict[str, tuple[float, float]]:
    """Joint forces that would exist if all driving torques were zero (triple).

    Parameters
    ----------
    state : np.ndarray, shape (6,)
        Current state [theta1, phi1, phi2, dtheta1, dphi1, dphi2].
    params : TriplePendulumParams

    Returns
    -------
    dict with keys 'shoulder', 'wrist1', 'wrist2' — each (fx, fy) in Newtons.
    """
    assert state.shape == (6,), f"Expected state shape (6,), got {state.shape}"
    assert np.all(np.isfinite(state)), "State must be finite"

    qddot = _zero_torque_qddot_triple(state, params)
    forces = net_joint_forces_triple(state, qddot, params)

    for key, (fx, fy) in forces.items():
        assert np.isfinite(fx) and np.isfinite(
            fy
        ), f"Non-finite zero-torque force at {key}: ({fx}, {fy})"
    return forces
