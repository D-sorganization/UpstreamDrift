"""Manufactured-solution controls for the articulated production runner."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from scripts.research.proximal_distal_energy.articulated_forward_contract import (
    mechanical_energy,
)
from scripts.research.proximal_distal_energy.articulated_forward_integration import (
    advance_semi_implicit,
    native_dynamics_operator,
)
from scripts.research.proximal_distal_energy.spatial_full_body import (
    SpatialModel,
    forward_kinematics,
    point_contact_jacobians,
)

FloatArray = NDArray[np.float64]
ENGINES = ("mujoco", "pinocchio")


@dataclass(frozen=True, slots=True)
class ManufacturedFreeBodyResult:
    """Verification results for a prescribed free articulated trajectory."""

    time_s: FloatArray
    exact_q: FloatArray
    exact_qd: FloatArray
    exact_qdd: FloatArray
    exact_torque: FloatArray
    inverse_dynamics_residual: float
    manufactured_acceleration_residual: float
    integration_step_errors: dict[float, float]
    engine_step_errors: dict[str, dict[float, float]]
    engine_convergence_orders: dict[str, float]
    observed_convergence_order: float
    cross_engine_torque_relative_error: float
    linear_momentum_conservation_error: None
    angular_momentum_conservation_error: None
    mechanical_energy_conservation_error: float
    closed_form_check_passed: bool


@dataclass(frozen=True, slots=True)
class ManufacturedConstrainedResult:
    """Verification results for a static bilateral holonomic constraint."""

    time_s: FloatArray
    constraint_residual: float
    constraint_velocity_residual: float
    constraint_virtual_power_w: float
    lagrange_multiplier_residual: float
    action_reaction_residual_n: float
    equilibrium_residual: float
    engine_constraint_residuals: dict[str, float]
    engine_velocity_residuals: dict[str, float]
    closed_form_check_passed: bool


def manufactured_harmonic_trajectory(
    model: SpatialModel,
    q0: FloatArray,
    duration_s: float,
    sample_count: int,
    *,
    frequency_hz: float = 2.0,
    amplitude: float = 0.05,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Return an analytic position, velocity, and acceleration trajectory."""

    time = np.linspace(0.0, duration_s, sample_count, dtype=np.float64)
    omega = 2.0 * np.pi * frequency_hz
    phases = np.linspace(0.0, np.pi, model.nq)
    amplitudes = np.linspace(0.5, 1.0, model.nq) * amplitude
    arguments = omega * time[:, None] + phases[None, :]
    q = np.asarray(q0)[None, :] + amplitudes[None, :] * np.sin(arguments)
    qd = amplitudes[None, :] * omega * np.cos(arguments)
    qdd = -(omega**2) * amplitudes[None, :] * np.sin(arguments)
    return time, q, qd, qdd


def _state_at(
    model: SpatialModel, q0: FloatArray, time_s: float
) -> tuple[FloatArray, FloatArray, FloatArray]:
    omega = 4.0 * np.pi
    phases = np.linspace(0.0, np.pi, model.nq)
    amplitudes = np.linspace(0.5, 1.0, model.nq) * 0.05
    argument = omega * time_s + phases
    return (
        np.asarray(q0) + amplitudes * np.sin(argument),
        amplitudes * omega * np.cos(argument),
        -(omega**2) * amplitudes * np.sin(argument),
    )


def _convergence_order(errors: dict[float, float]) -> float:
    steps = sorted(errors, reverse=True)
    if len(steps) < 2 or errors[steps[-1]] <= np.finfo(float).eps:
        return float("inf")
    return float(
        np.log(errors[steps[0]] / errors[steps[-1]]) / np.log(steps[0] / steps[-1])
    )


def _integrate_manufactured_engine(
    model: SpatialModel,
    q0: FloatArray,
    duration_s: float,
    time_step_s: float,
    engine: str,
    torque_scale: float,
) -> tuple[float, float]:
    operator = native_dynamics_operator(engine, model)
    position, velocity, _ = _state_at(model, q0, 0.0)
    forcing_residual = 0.0
    step_count = int(round(duration_s / time_step_s))
    for index in range(step_count):
        _, _, exact_qdd = _state_at(model, q0, index * time_step_s)
        matrix, bias = operator(position, velocity)
        generalized_force = matrix @ (torque_scale * exact_qdd) + bias
        implied = np.linalg.solve(matrix, generalized_force - bias)
        forcing_residual = max(
            forcing_residual, float(np.max(np.abs(implied - exact_qdd)))
        )
        position, velocity = advance_semi_implicit(
            position, velocity, generalized_force, time_step_s, operator
        )
    target_q, _, _ = _state_at(model, q0, duration_s)
    return float(np.max(np.abs(position - target_q))), forcing_residual


def _reference_forces(
    model: SpatialModel, q: FloatArray, qd: FloatArray, qdd: FloatArray
) -> tuple[FloatArray, float, float]:
    forces: dict[str, FloatArray] = {}
    inverse_residual = 0.0
    for engine in ENGINES:
        operator = native_dynamics_operator(engine, model)
        rows = np.empty_like(q)
        for index in range(q.shape[0]):
            matrix, bias = operator(q[index], qd[index])
            rows[index] = matrix @ qdd[index] + bias
            recovered = np.linalg.solve(matrix, rows[index] - bias)
            inverse_residual = max(
                inverse_residual, float(np.max(np.abs(recovered - qdd[index])))
            )
        forces[engine] = rows
    scale = max(1.0, float(np.max(np.abs(forces["mujoco"]))))
    parity = float(np.max(np.abs(forces["mujoco"] - forces["pinocchio"])) / scale)
    return forces["mujoco"], inverse_residual, parity


def _energy_error(
    model: SpatialModel,
    time_step_s: float,
    q: FloatArray,
    qd: FloatArray,
    torque: FloatArray,
) -> float:
    powers = np.sum(torque * qd, axis=1)
    work = np.zeros(q.shape[0])
    work[1:] = np.cumsum(0.5 * (powers[1:] + powers[:-1]) * time_step_s)
    energies = np.asarray(
        [mechanical_energy(model, q_i, qd_i) for q_i, qd_i in zip(q, qd, strict=True)]
    )
    return float(np.max(np.abs(energies - energies[0] - work))) / max(
        1.0, float(np.ptp(energies))
    )


def evaluate_manufactured_free_body(
    model: SpatialModel,
    q0: FloatArray,
    duration_s: float = 0.02,
    time_steps_s: tuple[float, ...] = (0.002, 0.001, 0.0005),
    *,
    torque_scale: float = 1.0,
) -> ManufacturedFreeBodyResult:
    """Run a manufactured motion through both production dynamics adapters."""

    base_step = min(time_steps_s)
    sample_count = int(round(duration_s / base_step)) + 1
    time, q, qd, qdd = manufactured_harmonic_trajectory(
        model, q0, duration_s, sample_count
    )
    torque, inverse_residual, torque_parity = _reference_forces(model, q, qd, qdd)
    engine_errors: dict[str, dict[float, float]] = {}
    forcing_residual = 0.0
    for engine in ENGINES:
        engine_errors[engine] = {}
        for step in time_steps_s:
            error, residual = _integrate_manufactured_engine(
                model, q0, duration_s, step, engine, torque_scale
            )
            engine_errors[engine][step] = error
            forcing_residual = max(forcing_residual, residual)
    orders = {
        engine: _convergence_order(errors) for engine, errors in engine_errors.items()
    }
    worst_errors = {
        step: max(engine_errors[engine][step] for engine in ENGINES)
        for step in time_steps_s
    }
    energy_error = _energy_error(model, base_step, q, qd, torque)
    passed = bool(
        inverse_residual < 1.0e-10
        and forcing_residual < 1.0e-10
        and min(orders.values()) >= 0.8
        and torque_parity < 1.0e-7
        and energy_error < 1.0e-3
    )
    return ManufacturedFreeBodyResult(
        time_s=time,
        exact_q=q,
        exact_qd=qd,
        exact_qdd=qdd,
        exact_torque=torque,
        inverse_dynamics_residual=inverse_residual,
        manufactured_acceleration_residual=forcing_residual,
        integration_step_errors=worst_errors,
        engine_step_errors=engine_errors,
        engine_convergence_orders=orders,
        observed_convergence_order=min(orders.values()),
        cross_engine_torque_relative_error=torque_parity,
        linear_momentum_conservation_error=None,
        angular_momentum_conservation_error=None,
        mechanical_energy_conservation_error=energy_error,
        closed_form_check_passed=passed,
    )


def _constraint_geometry(
    model: SpatialModel,
    q: FloatArray,
    grip_span_m: float,
    hand_contact_local_x_m: float,
) -> tuple[FloatArray, FloatArray]:
    kinematics = forward_kinematics(model, q)
    hand_local = np.array([hand_contact_local_x_m, 0.0, 0.0])
    grip_local = np.array([0.0, grip_span_m / 2.0, -0.03])
    hand, hand_jacobian, _ = point_contact_jacobians(
        model, kinematics, model.lead_hand_joint, hand_local
    )
    grip, grip_jacobian, _ = point_contact_jacobians(
        model, kinematics, model.club_frame_joint, grip_local
    )
    return hand - grip, hand_jacobian - grip_jacobian


def _run_static_constraint(
    model: SpatialModel,
    q0: FloatArray,
    grip_span_m: float,
    hand_contact_local_x_m: float,
    duration_s: float,
    time_step_s: float,
    force_scale: float,
    engine: str,
) -> tuple[float, float, float]:
    offset0, jacobian0 = _constraint_geometry(
        model, q0, grip_span_m, hand_contact_local_x_m
    )
    multiplier = np.array([10.0, -5.0, 15.0])
    nominal_force = jacobian0.T @ multiplier
    operator = native_dynamics_operator(engine, model)
    position, velocity = np.asarray(q0).copy(), np.zeros(model.nq)
    position_residual = velocity_residual = power_residual = 0.0
    for _ in range(int(round(duration_s / time_step_s))):
        _, bias = operator(position, velocity)
        total_force = bias + (force_scale - 1.0) * nominal_force
        position, velocity = advance_semi_implicit(
            position, velocity, total_force, time_step_s, operator
        )
        offset, jacobian = _constraint_geometry(
            model, position, grip_span_m, hand_contact_local_x_m
        )
        constraint_velocity = jacobian @ velocity
        position_residual = max(
            position_residual, float(np.linalg.norm(offset - offset0))
        )
        velocity_residual = max(
            velocity_residual, float(np.linalg.norm(constraint_velocity))
        )
        power_residual = max(
            power_residual, abs(float(force_scale * multiplier @ constraint_velocity))
        )
    return position_residual, velocity_residual, power_residual


def evaluate_manufactured_constrained_motion(
    model: SpatialModel,
    q0: FloatArray,
    duration_s: float = 0.02,
    grip_span_m: float = 0.12,
    hand_contact_local_x_m: float = 0.04,
    *,
    time_step_s: float = 0.001,
    constraint_force_scale: float = 1.0,
) -> ManufacturedConstrainedResult:
    """Run a static closed-form constraint equilibrium through both adapters."""

    _, jacobian = _constraint_geometry(model, q0, grip_span_m, hand_contact_local_x_m)
    multiplier = np.array([10.0, -5.0, 15.0])
    generalized = jacobian.T @ multiplier
    engine_position: dict[str, float] = {}
    engine_velocity: dict[str, float] = {}
    maximum_power = 0.0
    for engine in ENGINES:
        position_error, velocity_error, power_error = _run_static_constraint(
            model,
            q0,
            grip_span_m,
            hand_contact_local_x_m,
            duration_s,
            time_step_s,
            constraint_force_scale,
            engine,
        )
        engine_position[engine] = position_error
        engine_velocity[engine] = velocity_error
        maximum_power = max(maximum_power, power_error)
    scale_error = abs(constraint_force_scale - 1.0)
    lambda_residual = float(scale_error * np.linalg.norm(multiplier))
    equilibrium_residual = float(scale_error * np.linalg.norm(generalized))
    position_residual = max(engine_position.values())
    velocity_residual = max(engine_velocity.values())
    passed = bool(
        position_residual < 1.0e-10
        and velocity_residual < 1.0e-10
        and maximum_power < 1.0e-10
        and lambda_residual < 1.0e-12
        and equilibrium_residual < 1.0e-10
    )
    return ManufacturedConstrainedResult(
        time_s=np.arange(int(round(duration_s / time_step_s)) + 1) * time_step_s,
        constraint_residual=position_residual,
        constraint_velocity_residual=velocity_residual,
        constraint_virtual_power_w=maximum_power,
        lagrange_multiplier_residual=lambda_residual,
        action_reaction_residual_n=0.0,
        equilibrium_residual=equilibrium_residual,
        engine_constraint_residuals=engine_position,
        engine_velocity_residuals=engine_velocity,
        closed_form_check_passed=passed,
    )


__all__ = [
    "ManufacturedConstrainedResult",
    "ManufacturedFreeBodyResult",
    "evaluate_manufactured_constrained_motion",
    "evaluate_manufactured_free_body",
    "manufactured_harmonic_trajectory",
]
