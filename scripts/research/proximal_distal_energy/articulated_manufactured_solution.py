"""Independent manufactured-solution controls for the articulated tier (#8752).

The manufactured torque is defined by the repository's analytical
Lagrange--Christoffel formulation, then independently evaluated by MuJoCo
``mj_inverse`` and robotics Pinocchio RNEA. Conservation quantities come from
an unforced, gravity-free rollout; no verification field is a literal zero or
an expression compared with itself.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from scripts.research.proximal_distal_energy.articulated_inertia_cross_engine import (
    build_pinocchio_articulated_model,
    require_robotics_pinocchio,
)
from scripts.research.proximal_distal_energy.spatial_full_body import (
    SpatialModel,
    forward_kinematics,
    lagrange_inverse_dynamics,
    mass_matrix,
    mujoco_inverse_dynamics,
    mujoco_mass_matrix_and_bias,
    point_contact_jacobians,
)

FloatArray = NDArray[np.float64]
_DERIVATIVE_STEP = 1.0e-6
_INVERSE_RELATIVE_TOLERANCE = 5.0e-2
_CONSERVATION_RELATIVE_TOLERANCE = 2.0e-2
_RICHARDSON_ORDER_BOUNDS = (0.9, 1.1)
_PINOCCHIO_INVERSE_CACHE: dict[
    str, Callable[[FloatArray, FloatArray, FloatArray], FloatArray]
] = {}


@dataclass(frozen=True, slots=True)
class ManufacturedFreeBodyResult:
    """Measured verification results for manufactured and free rollouts."""

    time_s: FloatArray
    exact_q: FloatArray
    exact_qd: FloatArray
    exact_qdd: FloatArray
    exact_torque: FloatArray
    inverse_dynamics_residual: float
    lagrange_mujoco_relative_error: float
    lagrange_pinocchio_relative_error: float
    mujoco_pinocchio_relative_error: float
    independent_engine_difference_detected: bool
    integration_step_errors: dict[float, float]
    richardson_orders: tuple[float, ...]
    observed_convergence_order: float
    linear_momentum_conservation_error: float
    angular_momentum_conservation_error: float
    mechanical_energy_conservation_error: float
    closed_form_check_passed: bool


@dataclass(frozen=True, slots=True)
class ManufacturedConstrainedResult:
    """Independent inverse-dynamics and kinematic constraint results."""

    time_s: FloatArray
    constraint_residual: float
    constraint_velocity_residual: float
    constraint_virtual_power_w: float
    lagrange_multiplier_residual: float
    action_reaction_residual_n: float
    equilibrium_residual: float
    independent_engine_difference_detected: bool
    closed_form_check_passed: bool


def _harmonic_state(
    model: SpatialModel,
    q0: FloatArray,
    time_s: FloatArray,
    frequency_hz: float,
    amplitude: float,
    active_indices: tuple[int, ...] | None = None,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    omega = 2.0 * np.pi * frequency_hz
    phases = np.linspace(0.0, np.pi, model.nq)
    amplitudes = np.linspace(0.5, 1.0, model.nq) * amplitude
    if active_indices is not None:
        active = np.zeros(model.nq, dtype=bool)
        active[list(active_indices)] = True
        amplitudes[~active] = 0.0
    argument = omega * time_s[:, None] + phases[None, :]
    q = q0[None, :] + amplitudes[None, :] * np.sin(argument)
    qd = amplitudes[None, :] * omega * np.cos(argument)
    qdd = -(amplitudes[None, :] * omega**2) * np.sin(argument)
    return q, qd, qdd


def manufactured_harmonic_trajectory(
    model: SpatialModel,
    q0: FloatArray,
    duration_s: float,
    sample_count: int,
    *,
    frequency_hz: float = 2.0,
    amplitude: float = 0.05,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Generate a smooth harmonic manufactured state and exact derivatives."""

    time = np.linspace(0.0, duration_s, sample_count, dtype=np.float64)
    q, qd, qdd = _harmonic_state(model, q0, time, frequency_hz, amplitude)
    return time, q, qd, qdd


def _pinocchio_inverse_operator(
    model: SpatialModel,
) -> Callable[[FloatArray, FloatArray, FloatArray], FloatArray]:
    cached = _PINOCCHIO_INVERSE_CACHE.get(model.canonical_hash)
    if cached is not None:
        return cached
    try:
        import pinocchio as pin
    except ImportError as error:  # pragma: no cover - native runtime gate
        raise RuntimeError(
            "robotics Pinocchio >= 2.6 is required; install the 'pin' package"
        ) from error
    require_robotics_pinocchio(pin)
    native = build_pinocchio_articulated_model(pin, model)
    data = native.createData()

    def evaluate(q: FloatArray, qd: FloatArray, qdd: FloatArray) -> FloatArray:
        return np.asarray(pin.rnea(native, data, q, qd, qdd), dtype=float).copy()

    _PINOCCHIO_INVERSE_CACHE[model.canonical_hash] = evaluate
    return evaluate


def _relative_error(left: FloatArray, right: FloatArray) -> float:
    absolute = float(np.max(np.abs(np.asarray(left) - np.asarray(right))))
    scale = max(1.0, float(np.max(np.abs(left))), float(np.max(np.abs(right))))
    return absolute / scale


def _inverse_trajectory(
    model: SpatialModel,
    q: FloatArray,
    qd: FloatArray,
    qdd: FloatArray,
) -> tuple[FloatArray, tuple[float, float, float], bool]:
    pinocchio_inverse = _pinocchio_inverse_operator(model)
    analytical = np.empty_like(q)
    mujoco = np.empty_like(q)
    pinocchio = np.empty_like(q)
    zero = np.zeros(model.nq)
    for index in range(q.shape[0]):
        analytical[index] = lagrange_inverse_dynamics(
            model, q[index], qd[index], qdd[index], zero, _DERIVATIVE_STEP
        )
        mujoco[index] = mujoco_inverse_dynamics(
            model, q[index], qd[index], qdd[index], zero
        )
        pinocchio[index] = pinocchio_inverse(q[index], qd[index], qdd[index])
    errors = (
        _relative_error(analytical, mujoco),
        _relative_error(analytical, pinocchio),
        _relative_error(mujoco, pinocchio),
    )
    differences = np.concatenate(
        ((analytical - mujoco).ravel(), (analytical - pinocchio).ravel())
    )
    return analytical, errors, bool(np.any(differences != 0.0))


def _exact_state_at(
    model: SpatialModel, q0: FloatArray, time_s: float
) -> tuple[FloatArray, FloatArray, FloatArray]:
    q, qd, qdd = _harmonic_state(
        model, q0, np.array([time_s]), frequency_hz=2.0, amplitude=0.05
    )
    return q[0], qd[0], qdd[0]


def _integration_errors(
    model: SpatialModel,
    q0: FloatArray,
    duration_s: float,
    time_steps_s: tuple[float, ...],
) -> dict[float, float]:
    errors: dict[float, float] = {}
    zero = np.zeros(model.nq)
    for dt in time_steps_s:
        steps = int(round(duration_s / dt))
        q_sim, qd_sim, _ = _exact_state_at(model, q0, 0.0)
        for step_index in range(steps):
            q_exact, qd_exact, qdd_exact = _exact_state_at(model, q0, step_index * dt)
            torque = lagrange_inverse_dynamics(
                model, q_exact, qd_exact, qdd_exact, zero, _DERIVATIVE_STEP
            )
            matrix, bias = mujoco_mass_matrix_and_bias(model, q_sim, qd_sim)
            qd_sim += dt * np.linalg.solve(matrix, torque - bias)
            q_sim += dt * qd_sim
        q_final, _, _ = _exact_state_at(model, q0, duration_s)
        errors[dt] = float(np.max(np.abs(q_sim - q_final)))
    return errors


def _richardson_orders(errors: dict[float, float]) -> tuple[float, ...]:
    steps = sorted(errors, reverse=True)
    return tuple(
        float(np.log(errors[coarse] / errors[fine]) / np.log(coarse / fine))
        for coarse, fine in zip(steps[:-1], steps[1:], strict=True)
    )


def _gravity_force(model: SpatialModel, q: FloatArray) -> FloatArray:
    kin = forward_kinematics(model, q)
    gravity = np.zeros(model.nq)
    for index, body in enumerate(model.bodies):
        gravity += body.mass_kg * 9.80665 * kin.body_linear_jacobian[index, 2]
    return gravity


def _system_invariants(
    model: SpatialModel, q: FloatArray, qd: FloatArray
) -> tuple[FloatArray, FloatArray, float]:
    kin = forward_kinematics(model, q)
    linear = np.zeros(3)
    angular = np.zeros(3)
    for index, body in enumerate(model.bodies):
        velocity = kin.body_linear_jacobian[index] @ qd
        omega = kin.body_angular_jacobian[index] @ qd
        momentum = body.mass_kg * velocity
        inertia = 0.4 * body.mass_kg * body.radius_m**2
        linear += momentum
        angular += inertia * omega + np.cross(kin.body_position_m[index], momentum)
    kinetic = 0.5 * float(qd @ mass_matrix(model, q) @ qd)
    return linear, angular, kinetic


def _free_rollout_conservation(
    model: SpatialModel,
    q_initial: FloatArray,
    qd_initial: FloatArray,
    duration_s: float,
    step_s: float,
) -> tuple[float, float, float]:
    q = q_initial.copy()
    qd = qd_initial.copy()
    initial = _system_invariants(model, q, qd)
    maxima = np.zeros(3)
    for _ in range(int(round(duration_s / step_s))):
        matrix, native_bias = mujoco_mass_matrix_and_bias(model, q, qd)
        coriolis = native_bias - _gravity_force(model, q)
        qd += step_s * np.linalg.solve(matrix, -coriolis)
        q += step_s * qd
        current = _system_invariants(model, q, qd)
        maxima[0] = max(maxima[0], np.linalg.norm(current[0] - initial[0]))
        maxima[1] = max(maxima[1], np.linalg.norm(current[1] - initial[1]))
        maxima[2] = max(maxima[2], abs(current[2] - initial[2]))
    scales = np.array(
        [
            max(1.0, float(np.linalg.norm(initial[0]))),
            max(1.0, float(np.linalg.norm(initial[1]))),
            max(1.0, abs(initial[2])),
        ]
    )
    values = maxima / scales
    return float(values[0]), float(values[1]), float(values[2])


def evaluate_manufactured_free_body(
    model: SpatialModel,
    q0: FloatArray,
    duration_s: float = 0.02,
    time_steps_s: tuple[float, ...] = (0.002, 0.001, 0.0005),
) -> ManufacturedFreeBodyResult:
    """Verify three inverse-dynamics paths, convergence, and conservation."""

    if len(time_steps_s) < 3:
        raise ValueError(
            "three or more time steps are required for Richardson analysis"
        )
    sample_count = int(round(duration_s / min(time_steps_s))) + 1
    time, q, qd, qdd = manufactured_harmonic_trajectory(
        model, q0, duration_s, sample_count
    )
    torque, inverse_errors, nonzero = _inverse_trajectory(model, q, qd, qdd)
    step_errors = _integration_errors(model, q0, duration_s, time_steps_s)
    orders = _richardson_orders(step_errors)
    free_velocity = np.zeros(model.nq)
    free_velocity[model.club_dof_indices] = qd[0, model.club_dof_indices]
    conservation = _free_rollout_conservation(
        model, q[0], free_velocity, duration_s, min(time_steps_s)
    )
    inverse_max = max(inverse_errors)
    order_pass = all(
        _RICHARDSON_ORDER_BOUNDS[0] <= value <= _RICHARDSON_ORDER_BOUNDS[1]
        for value in orders
    )
    passed = (
        nonzero
        and inverse_max < _INVERSE_RELATIVE_TOLERANCE
        and order_pass
        and max(conservation) < _CONSERVATION_RELATIVE_TOLERANCE
    )
    return ManufacturedFreeBodyResult(
        time_s=time,
        exact_q=q,
        exact_qd=qd,
        exact_qdd=qdd,
        exact_torque=torque,
        inverse_dynamics_residual=inverse_max,
        lagrange_mujoco_relative_error=inverse_errors[0],
        lagrange_pinocchio_relative_error=inverse_errors[1],
        mujoco_pinocchio_relative_error=inverse_errors[2],
        independent_engine_difference_detected=nonzero,
        integration_step_errors=step_errors,
        richardson_orders=orders,
        observed_convergence_order=float(np.mean(orders)),
        linear_momentum_conservation_error=conservation[0],
        angular_momentum_conservation_error=conservation[1],
        mechanical_energy_conservation_error=conservation[2],
        closed_form_check_passed=passed,
    )


def _constrained_yaw_trajectory(
    model: SpatialModel,
    q0: FloatArray,
    time: FloatArray,
    hand_local: FloatArray,
    grip_local: FloatArray,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Manufacture exact closure by coordinating pelvis yaw and club translation."""

    omega = 2.0 * np.pi * 2.0
    amplitude = 0.01
    yaw = q0[0] + amplitude * np.sin(omega * time)
    yaw_rate = amplitude * omega * np.cos(omega * time)
    yaw_acceleration = -(amplitude * omega**2) * np.sin(omega * time)
    q = np.repeat(q0[None, :], time.size, axis=0)
    qd = np.zeros_like(q)
    qdd = np.zeros_like(q)
    derivative_step = 1.0e-6
    for index in range(time.size):
        q[index, 0] = yaw[index]
        q[index, 14:17] = 0.0
        kin = forward_kinematics(model, q[index])
        p_hand, j_hand, _ = point_contact_jacobians(
            model, kin, model.lead_hand_joint, hand_local
        )
        p_grip_zero, _, _ = point_contact_jacobians(
            model, kin, model.club_frame_joint, grip_local
        )
        q[index, 14:17] = p_hand - p_grip_zero
        qd[index, 0] = yaw_rate[index]
        qd[index, 14:17] = j_hand[:, 0] * yaw_rate[index]
        plus = q[index].copy()
        minus = q[index].copy()
        plus[0] += derivative_step
        minus[0] -= derivative_step
        _, j_plus, _ = point_contact_jacobians(
            model, forward_kinematics(model, plus), model.lead_hand_joint, hand_local
        )
        _, j_minus, _ = point_contact_jacobians(
            model, forward_kinematics(model, minus), model.lead_hand_joint, hand_local
        )
        jacobian_derivative = (j_plus[:, 0] - j_minus[:, 0]) / (2.0 * derivative_step)
        qdd[index, 0] = yaw_acceleration[index]
        qdd[index, 14:17] = (
            j_hand[:, 0] * yaw_acceleration[index]
            + jacobian_derivative * yaw_rate[index] ** 2
        )
    return q, qd, qdd


def evaluate_manufactured_constrained_motion(
    model: SpatialModel,
    q0: FloatArray,
    duration_s: float = 0.02,
    grip_span_m: float = 0.12,
    hand_contact_local_x_m: float = 0.04,
) -> ManufacturedConstrainedResult:
    """Verify a closed-contact trajectory with independent inverse dynamics."""

    time = np.linspace(0.0, duration_s, 21, dtype=np.float64)
    hand_local = np.array([hand_contact_local_x_m, 0.0, 0.0])
    grip_local = np.array([0.0, grip_span_m / 2.0, -0.03])
    q, qd, qdd = _constrained_yaw_trajectory(model, q0, time, hand_local, grip_local)
    pinocchio_inverse = _pinocchio_inverse_operator(model)
    lambda_exact = np.array([10.0, -5.0, 15.0])
    reference_difference: FloatArray | None = None
    position_residuals: list[float] = []
    velocity_residuals: list[float] = []
    virtual_powers: list[float] = []
    equilibrium_residuals: list[float] = []
    lambda_residuals: list[float] = []
    cross_engine_lambda_residuals: list[float] = []
    engine_differences: list[float] = []
    for index in range(time.size):
        kin = forward_kinematics(model, q[index])
        p_hand, j_hand, _ = point_contact_jacobians(
            model, kin, model.lead_hand_joint, hand_local
        )
        p_grip, j_grip, _ = point_contact_jacobians(
            model, kin, model.club_frame_joint, grip_local
        )
        difference = p_hand - p_grip
        if reference_difference is None:
            reference_difference = difference.copy()
        jacobian = j_hand - j_grip
        position_residuals.append(
            float(np.linalg.norm(difference - reference_difference))
        )
        velocity = jacobian @ qd[index]
        velocity_residuals.append(float(np.linalg.norm(velocity)))
        virtual_powers.append(float(abs(lambda_exact @ velocity)))
        external = jacobian.T @ lambda_exact
        analytical = lagrange_inverse_dynamics(
            model, q[index], qd[index], qdd[index], external, _DERIVATIVE_STEP
        )
        mujoco_native = mujoco_inverse_dynamics(
            model, q[index], qd[index], qdd[index], np.zeros(model.nq)
        )
        pinocchio_native = pinocchio_inverse(q[index], qd[index], qdd[index])
        mujoco_required = mujoco_native - external
        pinocchio_required = pinocchio_native - external
        equilibrium_residuals.extend(
            (
                _relative_error(analytical, mujoco_required),
                _relative_error(analytical, pinocchio_required),
            )
        )
        engine_differences.append(_relative_error(mujoco_required, pinocchio_required))
        lambda_mujoco = np.linalg.lstsq(
            jacobian.T, mujoco_native - analytical, rcond=None
        )[0]
        lambda_pinocchio = np.linalg.lstsq(
            jacobian.T, pinocchio_native - analytical, rcond=None
        )[0]
        lambda_scale = max(1.0, float(np.linalg.norm(lambda_exact)))
        lambda_residuals.extend(
            (
                float(np.linalg.norm(lambda_mujoco - lambda_exact)) / lambda_scale,
                float(np.linalg.norm(lambda_pinocchio - lambda_exact)) / lambda_scale,
            )
        )
        cross_engine_lambda_residuals.append(
            float(np.linalg.norm(lambda_mujoco - lambda_pinocchio)) / lambda_scale
        )
    max_equilibrium = max(equilibrium_residuals)
    max_lambda = max(lambda_residuals)
    max_cross_lambda = max(cross_engine_lambda_residuals)
    nonzero = bool(any(value != 0.0 for value in engine_differences))
    passed = (
        nonzero
        and max(position_residuals) < 1.0e-10
        and max(velocity_residuals) < 1.0e-10
        and max(virtual_powers) < 1.0e-9
        and max_equilibrium < _INVERSE_RELATIVE_TOLERANCE
        and max_lambda < _INVERSE_RELATIVE_TOLERANCE
        and max_cross_lambda < 1.0e-8
    )
    return ManufacturedConstrainedResult(
        time_s=time,
        constraint_residual=max(position_residuals),
        constraint_velocity_residual=max(velocity_residuals),
        constraint_virtual_power_w=max(virtual_powers),
        lagrange_multiplier_residual=max_lambda,
        action_reaction_residual_n=max_cross_lambda,
        equilibrium_residual=max_equilibrium,
        independent_engine_difference_detected=nonzero,
        closed_form_check_passed=passed,
    )


__all__ = [
    "ManufacturedConstrainedResult",
    "ManufacturedFreeBodyResult",
    "evaluate_manufactured_constrained_motion",
    "evaluate_manufactured_free_body",
    "manufactured_harmonic_trajectory",
]
