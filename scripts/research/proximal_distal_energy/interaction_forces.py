"""Exact interaction-force mechanics for the planar double pendulum.

The functions in this module separate the wrist reaction force acting on the
club into kinematically interpretable terms.  They also distinguish an
instantaneous zero-torque counterfactual from a forward-integrated matched-
state torque killswitch.  Coordinates and units follow :mod:`swing_model`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from scripts.research.proximal_distal_energy.swing_model import (
    PlanarInertials,
    hand_velocity,
)
from src.shared.python.simulation_backends import GolfModelParams, make_backend
from src.shared.python.simulation_backends.protocol import SimState, Trace


@dataclass(frozen=True)
class ReactionForceDecomposition:
    """Force on the club at the wrist and its exact inertial components."""

    total: np.ndarray
    club_com_acceleration: np.ndarray
    components: dict[str, np.ndarray]


@dataclass(frozen=True)
class ForcePowerDecomposition:
    """Power transmitted to the club by wrist force and component terms."""

    total: np.ndarray
    hand_velocity: np.ndarray
    components: dict[str, np.ndarray]


@dataclass(frozen=True)
class MatchedStateKillswitch:
    """Commanded and zero-torque trajectories departing one measured state."""

    cut_index: int
    commanded: Trace
    zero_torque: Trace
    commanded_initial_qdd: np.ndarray
    zero_torque_initial_qdd: np.ndarray


def _validated_trace_arrays(
    q: np.ndarray, v: np.ndarray, qdd: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arrays = tuple(np.asarray(value, dtype=float) for value in (q, v, qdd))
    if any(value.ndim != 2 or value.shape[1] != 2 for value in arrays):
        raise ValueError("q, v, and qdd must each have shape (samples, 2)")
    if not (arrays[0].shape == arrays[1].shape == arrays[2].shape):
        raise ValueError("q, v, and qdd must have identical shapes")
    if not all(np.all(np.isfinite(value)) for value in arrays):
        raise ValueError("q, v, and qdd must contain only finite values")
    return arrays


def _unit(theta: np.ndarray) -> np.ndarray:
    return np.column_stack((np.sin(theta), -np.cos(theta)))


def _unit_perp(theta: np.ndarray) -> np.ndarray:
    return np.column_stack((np.cos(theta), np.sin(theta)))


def reaction_force_decomposition(
    inertials: PlanarInertials,
    q: np.ndarray,
    v: np.ndarray,
    qdd: np.ndarray,
) -> ReactionForceDecomposition:
    """Return the wrist force on the club as five exact vector components.

    The club COM acceleration is the sum of proximal tangential, proximal
    centripetal, distal tangential, and distal centripetal accelerations.  The
    joint force follows from ``F_wrist + m g = m a_COM``.  Thus the reported
    gravity-reaction term belongs to the *force* decomposition, while gravity
    also influences the accelerations through the equations of motion.

    Postcondition:
        The component vectors sum to ``total`` to floating-point precision.
    """
    q_arr, v_arr, qdd_arr = _validated_trace_arrays(q, v, qdd)
    theta1 = q_arr[:, 0]
    phi = q_arr.sum(axis=1)
    omega1 = v_arr[:, 0]
    omega_phi = v_arr.sum(axis=1)
    alpha1 = qdd_arr[:, 0]
    alpha_phi = qdd_arr.sum(axis=1)

    a_prox_tan = inertials.l1 * alpha1[:, None] * _unit_perp(theta1)
    a_prox_cen = -inertials.l1 * omega1[:, None] ** 2 * _unit(theta1)
    a_dist_tan = inertials.lc2 * alpha_phi[:, None] * _unit_perp(phi)
    a_dist_cen = -inertials.lc2 * omega_phi[:, None] ** 2 * _unit(phi)
    a_com = a_prox_tan + a_prox_cen + a_dist_tan + a_dist_cen

    gravity_reaction = np.broadcast_to(
        np.array([0.0, inertials.m2 * inertials.g_proj]), a_com.shape
    ).copy()
    components = {
        "proximal_tangential": inertials.m2 * a_prox_tan,
        "proximal_centripetal": inertials.m2 * a_prox_cen,
        "distal_tangential": inertials.m2 * a_dist_tan,
        "distal_centripetal": inertials.m2 * a_dist_cen,
        "gravity_reaction": gravity_reaction,
    }
    total = sum(components.values())
    return ReactionForceDecomposition(
        total=total,
        club_com_acceleration=a_com,
        components=components,
    )


def force_power_decomposition(
    inertials: PlanarInertials,
    q: np.ndarray,
    v: np.ndarray,
    forces: ReactionForceDecomposition,
) -> ForcePowerDecomposition:
    """Project every wrist-force component onto wrist velocity.

    Positive power means that the wrist reaction force transfers energy into
    the club segment at the moving wrist point; negative power removes energy
    from that segment through the same interface.
    """
    q_arr = np.asarray(q, dtype=float)
    v_arr = np.asarray(v, dtype=float)
    if q_arr.shape != v_arr.shape or q_arr.ndim != 2 or q_arr.shape[1] != 2:
        raise ValueError("q and v must have identical shape (samples, 2)")
    if forces.total.shape != q_arr.shape:
        raise ValueError("force history must match the q and v sample shape")
    v_hand = hand_velocity(inertials, q_arr, v_arr)
    components = {
        name: np.einsum("ij,ij->i", vector, v_hand)
        for name, vector in forces.components.items()
    }
    total = np.einsum("ij,ij->i", forces.total, v_hand)
    return ForcePowerDecomposition(
        total=total,
        hand_velocity=v_hand,
        components=components,
    )


def matched_state_killswitch(
    params: GolfModelParams,
    t: np.ndarray,
    q: np.ndarray,
    v: np.ndarray,
    u: np.ndarray,
    *,
    cut_index: int,
    horizon: int,
    dt: float,
) -> MatchedStateKillswitch:
    """Integrate commanded and zero-torque futures from the same state.

    This is a trajectory-level counterfactual.  It must not be conflated with
    a pointwise ZTCF calculation, which changes acceleration while holding the
    observed state fixed at every sample.
    """
    t_arr = np.asarray(t, dtype=float)
    q_arr = np.asarray(q, dtype=float)
    v_arr = np.asarray(v, dtype=float)
    u_arr = np.asarray(u, dtype=float)
    samples = t_arr.size
    if q_arr.shape != (samples, 2) or v_arr.shape != (samples, 2):
        raise ValueError("q and v must have shape (len(t), 2)")
    if u_arr.shape != (samples, 2):
        raise ValueError("u must have shape (len(t), 2)")
    if not 0 <= cut_index < samples:
        raise ValueError("cut_index must identify a trace sample")
    if horizon <= 0 or cut_index + horizon > samples - 1:
        raise ValueError("horizon must fit after cut_index in the source trace")
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be positive and finite")

    state = SimState(q=q_arr[cut_index], v=v_arr[cut_index], time=t_arr[cut_index])
    controls = u_arr[cut_index : cut_index + horizon]
    commanded_backend = make_backend("ode", params)
    commanded_backend.reset(state)
    commanded_qdd = commanded_backend.forward_dynamics(state.q, state.v, controls[0])
    commanded = commanded_backend.rollout(controls, horizon, dt)

    zero_backend = make_backend("ode", params)
    zero_backend.reset(state)
    zero_qdd = zero_backend.forward_dynamics(state.q, state.v, np.zeros(2))
    zero_torque = zero_backend.rollout(None, horizon, dt)
    return MatchedStateKillswitch(
        cut_index=cut_index,
        commanded=commanded,
        zero_torque=zero_torque,
        commanded_initial_qdd=commanded_qdd,
        zero_torque_initial_qdd=zero_qdd,
    )
