"""Pointwise proximal-acceleration interventions at an identical mechanical state.

The intervention holds configuration, velocity, model, gravity, and distal
actuator torque fixed. It solves the proximal actuator torque required to hit
each declared proximal acceleration. Consequently state and instantaneous
kinetic energy are matched, while actuator torque, power, and prior work are not.
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
    hand_velocity,
)
from src.shared.python.simulation_backends import GolfModelParams, make_backend

FloatArray = npt.NDArray[np.float64]


def _vector(name: str, value: object) -> FloatArray:
    array = np.asarray(value, dtype=float)
    if array.shape != (2,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite with shape (2,)")
    return array


@dataclass(frozen=True, slots=True)
class AccelerationSweepRequest:
    """Inputs for an identical-state proximal-acceleration intervention."""

    q_rad: FloatArray
    velocity_rad_s: FloatArray
    reference_control_nm: FloatArray
    proximal_acceleration_rad_s2: FloatArray

    def __post_init__(self) -> None:
        for name in ("q_rad", "velocity_rad_s", "reference_control_nm"):
            object.__setattr__(self, name, _vector(name, getattr(self, name)))
        targets = np.asarray(self.proximal_acceleration_rad_s2, dtype=float).reshape(-1)
        if targets.size < 2 or not np.all(np.isfinite(targets)):
            raise ValueError(
                "proximal_acceleration_rad_s2 needs at least two finite values"
            )
        if np.unique(targets).size != targets.size:
            raise ValueError("proximal_acceleration_rad_s2 values must be unique")
        object.__setattr__(self, "proximal_acceleration_rad_s2", targets)


@dataclass(frozen=True, slots=True)
class AccelerationSweepRow:
    """One pointwise acceleration intervention and its closed force ledger."""

    proximal_acceleration_rad_s2: float
    distal_relative_acceleration_rad_s2: float
    total_club_angular_acceleration_rad_s2: float
    drift_club_angular_acceleration_rad_s2: float
    control_club_angular_acceleration_rad_s2: float
    proximal_acceleration_residual_rad_s2: float
    acceleration_closure_residual_rad_s2: float
    proximal_control_nm: float
    distal_control_nm: float
    proximal_control_power_w: float
    distal_control_power_w: float
    total_control_power_w: float
    total_kinetic_energy_j: float
    total_grip_power_w: float
    drift_grip_power_w: float
    control_grip_power_w: float
    braking_grip_power_w: float
    peak_grip_force_n: float
    force_closure_residual_n: float
    model_tier: str = "exact_planar_double_pendulum"
    counterfactual_kind: str = "pointwise_identical_state_fixed_distal_torque"


def _required_state(
    request: AccelerationSweepRequest,
    params: GolfModelParams,
    target: float,
) -> tuple[FloatArray, FloatArray, FloatArray, float]:
    backend = make_backend("ode", params)
    mass = np.asarray(backend.mass_matrix(request.q_rad), dtype=float)
    drift = backend.forward_dynamics(request.q_rad, request.velocity_rad_s, np.zeros(2))
    bias = -(mass @ drift)
    distal_torque = float(request.reference_control_nm[1])
    distal_acceleration = (
        distal_torque - float(bias[1]) - float(mass[1, 0]) * target
    ) / float(mass[1, 1])
    acceleration = np.array([target, distal_acceleration])
    proximal_torque = float(mass[0] @ acceleration + bias[0])
    control = np.array([proximal_torque, distal_torque])
    energy = 0.5 * float(request.velocity_rad_s @ mass @ request.velocity_rad_s)
    return acceleration, drift, control, energy


def _row(
    request: AccelerationSweepRequest,
    params: GolfModelParams,
    target: float,
) -> AccelerationSweepRow:
    acceleration, drift_acceleration, control, energy = _required_state(
        request, params, target
    )
    backend = make_backend("ode", params)
    evaluated = backend.forward_dynamics(request.q_rad, request.velocity_rad_s, control)
    control_acceleration = evaluated - drift_acceleration
    inertials = PlanarInertials.from_params(params)
    q = request.q_rad[None, :]
    velocity = request.velocity_rad_s[None, :]
    total_force = reaction_force_decomposition(
        inertials, q, velocity, evaluated[None, :]
    ).total[0]
    drift_force = reaction_force_decomposition(
        inertials, q, velocity, drift_acceleration[None, :]
    ).total[0]
    control_force = total_force - drift_force
    grip_velocity = hand_velocity(inertials, q, velocity)[0]
    grip_powers = np.array(
        [
            total_force @ grip_velocity,
            drift_force @ grip_velocity,
            control_force @ grip_velocity,
        ]
    )
    control_power = control * request.velocity_rad_s
    club_acceleration = np.array(
        [
            np.sum(evaluated),
            np.sum(drift_acceleration),
            np.sum(control_acceleration),
        ]
    )
    return AccelerationSweepRow(
        proximal_acceleration_rad_s2=float(target),
        distal_relative_acceleration_rad_s2=float(evaluated[1]),
        total_club_angular_acceleration_rad_s2=float(club_acceleration[0]),
        drift_club_angular_acceleration_rad_s2=float(club_acceleration[1]),
        control_club_angular_acceleration_rad_s2=float(club_acceleration[2]),
        proximal_acceleration_residual_rad_s2=float(evaluated[0] - target),
        acceleration_closure_residual_rad_s2=float(
            club_acceleration[0] - club_acceleration[1] - club_acceleration[2]
        ),
        proximal_control_nm=float(control[0]),
        distal_control_nm=float(control[1]),
        proximal_control_power_w=float(control_power[0]),
        distal_control_power_w=float(control_power[1]),
        total_control_power_w=float(np.sum(control_power)),
        total_kinetic_energy_j=energy,
        total_grip_power_w=float(grip_powers[0]),
        drift_grip_power_w=float(grip_powers[1]),
        control_grip_power_w=float(grip_powers[2]),
        braking_grip_power_w=float(min(grip_powers[0], 0.0)),
        peak_grip_force_n=float(np.linalg.norm(total_force)),
        force_closure_residual_n=float(
            np.linalg.norm(total_force - drift_force - control_force)
        ),
    )


def evaluate_acceleration_sweep(
    request: AccelerationSweepRequest, params: GolfModelParams
) -> tuple[AccelerationSweepRow, ...]:
    """Evaluate declared proximal accelerations at one identical state."""
    if not isinstance(request, AccelerationSweepRequest):
        raise TypeError("request must be an AccelerationSweepRequest")
    if not isinstance(params, GolfModelParams):
        raise TypeError("params must be GolfModelParams")
    return tuple(
        _row(request, params, float(target))
        for target in request.proximal_acceleration_rad_s2
    )


__all__ = [
    "AccelerationSweepRequest",
    "AccelerationSweepRow",
    "evaluate_acceleration_sweep",
]
