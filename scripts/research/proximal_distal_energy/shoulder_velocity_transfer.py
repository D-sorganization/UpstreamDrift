"""Pointwise shoulder-velocity tests for drift-mediated distal transfer.

This module changes velocity while holding configuration and applied control fixed.
It therefore answers a local state-field question, not how a player should create
that state and not whether an anatomical shoulder or torso should move at a given
speed. Forward trajectory and empirical tests remain separate evidence tiers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from scripts.research.proximal_distal_energy.interaction_forces import (
    reaction_force_decomposition,
)
from scripts.research.proximal_distal_energy.swing_model import (
    PlanarInertials,
    clubhead_speed,
    hand_velocity,
    segment_kinetic_energies,
)
from src.shared.python.simulation_backends import GolfModelParams, make_backend

FloatArray = npt.NDArray[np.float64]

_VELOCITY_CONSTRAINTS = {
    "preserve_relative_club_rate",
    "preserve_absolute_club_rate",
    "preserve_total_kinetic_energy",
}
_PHASES = (
    (0.10, "Transition"),
    (0.30, "Early Downswing"),
    (0.60, "Mid-Downswing"),
    (0.85, "Delivery and Release"),
    (1.00, "Pre-Impact"),
)


def _state_vector(name: str, value: object) -> FloatArray:
    array = np.asarray(value, dtype=float)
    if array.shape != (2,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite with shape (2,)")
    return array


@dataclass(frozen=True, slots=True)
class VelocitySweepRequest:
    """Inputs for a state-matched proximal-link velocity sweep."""

    q_rad: FloatArray
    reference_velocity_rad_s: FloatArray
    control_nm: FloatArray
    proximal_velocity_rad_s: FloatArray
    velocity_constraint: str = "preserve_relative_club_rate"

    def __post_init__(self) -> None:
        for name in ("q_rad", "reference_velocity_rad_s", "control_nm"):
            object.__setattr__(self, name, _state_vector(name, getattr(self, name)))
        proximal = np.asarray(self.proximal_velocity_rad_s, dtype=float).reshape(-1)
        if proximal.size < 2 or not np.all(np.isfinite(proximal)):
            raise ValueError("proximal_velocity_rad_s needs at least two finite values")
        if np.unique(proximal).size != proximal.size:
            raise ValueError("proximal_velocity_rad_s values must be unique")
        if self.velocity_constraint not in _VELOCITY_CONSTRAINTS:
            raise ValueError(
                "velocity_constraint must preserve relative club rate, absolute "
                "club rate, or total kinetic energy"
            )
        object.__setattr__(self, "proximal_velocity_rad_s", proximal)


@dataclass(frozen=True, slots=True)
class VelocitySweepRow:
    """One pointwise drift/control evaluation at a declared velocity state."""

    proximal_velocity_rad_s: float
    relative_club_velocity_rad_s: float
    club_angular_velocity_rad_s: float
    clubhead_speed_m_s: float
    distal_kinetic_energy_j: float
    total_kinetic_energy_j: float
    reference_total_kinetic_energy_j: float
    kinetic_energy_residual_j: float
    total_club_angular_acceleration_rad_s2: float
    drift_club_angular_acceleration_rad_s2: float
    control_club_angular_acceleration_rad_s2: float
    acceleration_closure_residual_rad_s2: float
    total_grip_power_w: float
    drift_grip_power_w: float
    control_grip_power_w: float
    braking_grip_power_w: float
    grip_force_total_n: tuple[float, float]
    grip_force_drift_n: tuple[float, float]
    grip_force_control_n: tuple[float, float]
    force_closure_residual_n: float
    model_tier: str = "exact_planar_double_pendulum"
    counterfactual_kind: str = "pointwise_state_matched_zero_applied_control"
    proximal_coordinate_meaning: str = (
        "first_link_angular_velocity_not_anatomical_shoulder"
    )


def classify_transfer_phase(normalized_downswing_time: float) -> str:
    """Return a declared phase label for a normalized downswing fraction."""
    value = float(normalized_downswing_time)
    if not np.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("normalized_downswing_time must lie in [0, 1]")
    return next(name for upper, name in _PHASES if value <= upper)


def _total_kinetic_energy(mass: FloatArray, velocity: FloatArray) -> float:
    return 0.5 * float(velocity @ mass @ velocity)


def _velocity_for(
    request: VelocitySweepRequest, proximal_rate: float, mass: FloatArray
) -> FloatArray:
    if request.velocity_constraint == "preserve_relative_club_rate":
        return np.array([proximal_rate, request.reference_velocity_rad_s[1]])
    if request.velocity_constraint == "preserve_absolute_club_rate":
        absolute_rate = float(np.sum(request.reference_velocity_rad_s))
        return np.array([proximal_rate, absolute_rate - proximal_rate])

    target_energy = _total_kinetic_energy(mass, request.reference_velocity_rad_s)
    quadratic_a = 0.5 * float(mass[1, 1])
    quadratic_b = float(mass[0, 1]) * proximal_rate
    quadratic_c = 0.5 * float(mass[0, 0]) * proximal_rate**2 - target_energy
    discriminant = quadratic_b**2 - 4.0 * quadratic_a * quadratic_c
    tolerance = 1e-12 * max(1.0, quadratic_b**2, abs(4.0 * quadratic_a * quadratic_c))
    if discriminant < -tolerance:
        raise ValueError(
            "proximal rate is infeasible under the total-kinetic-energy constraint"
        )
    root_term = np.sqrt(max(discriminant, 0.0))
    roots = np.array(
        [
            (-quadratic_b - root_term) / (2.0 * quadratic_a),
            (-quadratic_b + root_term) / (2.0 * quadratic_a),
        ]
    )
    reference_relative_rate = float(request.reference_velocity_rad_s[1])
    relative_rate = float(roots[np.argmin(np.abs(roots - reference_relative_rate))])
    return np.array([proximal_rate, relative_rate])


def _row(
    request: VelocitySweepRequest,
    params: GolfModelParams,
    proximal_rate: float,
) -> VelocitySweepRow:
    backend = make_backend("ode", params)
    q = request.q_rad
    mass = np.asarray(backend.mass_matrix(q), dtype=float)
    velocity = _velocity_for(request, proximal_rate, mass)
    reference_energy = _total_kinetic_energy(mass, request.reference_velocity_rad_s)
    total_energy = _total_kinetic_energy(mass, velocity)
    total_qdd = backend.forward_dynamics(q, velocity, request.control_nm)
    drift_qdd = backend.forward_dynamics(q, velocity, np.zeros(2))
    control_qdd = total_qdd - drift_qdd
    inertials = PlanarInertials.from_params(params)
    q_row, velocity_row = q[None, :], velocity[None, :]
    total_force = reaction_force_decomposition(
        inertials, q_row, velocity_row, total_qdd[None, :]
    ).total[0]
    drift_force = reaction_force_decomposition(
        inertials, q_row, velocity_row, drift_qdd[None, :]
    ).total[0]
    control_force = total_force - drift_force
    grip_velocity = hand_velocity(inertials, q_row, velocity_row)[0]
    powers = np.array(
        [
            np.dot(total_force, grip_velocity),
            np.dot(drift_force, grip_velocity),
            np.dot(control_force, grip_velocity),
        ]
    )
    _, distal_energy = segment_kinetic_energies(inertials, q_row, velocity_row)
    club_acceleration = np.array(
        [np.sum(total_qdd), np.sum(drift_qdd), np.sum(control_qdd)]
    )
    return VelocitySweepRow(
        proximal_velocity_rad_s=float(proximal_rate),
        relative_club_velocity_rad_s=float(velocity[1]),
        club_angular_velocity_rad_s=float(np.sum(velocity)),
        clubhead_speed_m_s=float(clubhead_speed(inertials, q_row, velocity_row)[0]),
        distal_kinetic_energy_j=float(distal_energy[0]),
        total_kinetic_energy_j=total_energy,
        reference_total_kinetic_energy_j=reference_energy,
        kinetic_energy_residual_j=total_energy - reference_energy,
        total_club_angular_acceleration_rad_s2=float(club_acceleration[0]),
        drift_club_angular_acceleration_rad_s2=float(club_acceleration[1]),
        control_club_angular_acceleration_rad_s2=float(club_acceleration[2]),
        acceleration_closure_residual_rad_s2=float(
            club_acceleration[0] - club_acceleration[1] - club_acceleration[2]
        ),
        total_grip_power_w=float(powers[0]),
        drift_grip_power_w=float(powers[1]),
        control_grip_power_w=float(powers[2]),
        braking_grip_power_w=float(min(powers[0], 0.0)),
        grip_force_total_n=tuple(float(value) for value in total_force),
        grip_force_drift_n=tuple(float(value) for value in drift_force),
        grip_force_control_n=tuple(float(value) for value in control_force),
        force_closure_residual_n=float(
            np.linalg.norm(total_force - drift_force - control_force)
        ),
    )


def evaluate_velocity_sweep(
    request: VelocitySweepRequest, params: GolfModelParams
) -> tuple[VelocitySweepRow, ...]:
    """Evaluate total, drift, and control outcomes at matched configurations."""
    if not isinstance(request, VelocitySweepRequest):
        raise TypeError("request must be a VelocitySweepRequest")
    if not isinstance(params, GolfModelParams):
        raise TypeError("params must be GolfModelParams")
    return tuple(
        _row(request, params, rate) for rate in request.proximal_velocity_rad_s
    )


def nondominated_indices(
    values: object, *, maximize: tuple[bool, ...]
) -> npt.NDArray[np.int64]:
    """Return nondominated indices for mixed maximum/minimum objectives."""
    objectives = np.asarray(values, dtype=float)
    if objectives.ndim != 2 or objectives.shape[0] == 0:
        raise ValueError("values must be a non-empty two-dimensional array")
    if objectives.shape[1] != len(maximize):
        raise ValueError("maximize length must match the objective count")
    if not np.all(np.isfinite(objectives)):
        raise ValueError("values must be finite")
    oriented = objectives * np.where(np.asarray(maximize), -1.0, 1.0)
    keep = np.ones(oriented.shape[0], dtype=bool)
    for index, candidate in enumerate(oriented):
        dominates = np.all(oriented <= candidate, axis=1)
        dominates &= np.any(oriented < candidate, axis=1)
        keep[index] = not np.any(dominates)
    return np.flatnonzero(keep)


__all__ = [
    "VelocitySweepRequest",
    "VelocitySweepRow",
    "classify_transfer_phase",
    "evaluate_velocity_sweep",
    "nondominated_indices",
]
