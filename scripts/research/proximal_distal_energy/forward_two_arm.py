"""Forward constrained rollout for the planar two-arm floating-club model.

The existing closed-loop module supplies exact mass, bias, constraint, contact,
and KKT primitives. This module adds forward state evolution with explicit
mass-metric position and velocity projection. Projection corrections and every
solver residual are recorded; no least-squares or singular fallback exists.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from scripts.research.proximal_distal_energy.two_arm_closed_loop import (
    N_CONSTRAINTS,
    N_COORDINATES,
    TwoArmControl,
    TwoArmParams,
    constraint_jacobian,
    constraint_vector,
    mass_matrix,
    solve_constrained_dynamics,
)

FloatArray = npt.NDArray[np.float64]
ControlLaw = Callable[[float, FloatArray, FloatArray], TwoArmControl]


@dataclass(frozen=True, slots=True)
class ForwardTwoArmConfig:
    """Integrator and fail-closed projection settings."""

    duration_s: float
    step_s: float
    start_time_s: float = 0.0
    projection_tolerance_m: float = 1e-10
    velocity_tolerance_m_s: float = 1e-9
    maximum_projection_iterations: int = 12

    def __post_init__(self) -> None:
        for name in (
            "duration_s",
            "step_s",
            "projection_tolerance_m",
            "velocity_tolerance_m_s",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not np.isfinite(self.start_time_s):
            raise ValueError("start_time_s must be finite")
        if self.maximum_projection_iterations < 1:
            raise ValueError("maximum_projection_iterations must be at least one")
        steps = self.duration_s / self.step_s
        if not np.isclose(steps, round(steps), rtol=0.0, atol=1e-10):
            raise ValueError("duration_s must be an integer multiple of step_s")

    @property
    def interval_count(self) -> int:
        """Return the exact number of integration intervals."""
        return int(round(self.duration_s / self.step_s))


@dataclass(frozen=True, slots=True)
class ForwardTwoArmTrace:
    """Forward state, constraint, contact, and energy evidence."""

    time: FloatArray
    q: FloatArray
    qdot: FloatArray
    qddot: FloatArray
    controls: tuple[TwoArmControl, ...]
    multipliers_n: FloatArray
    contact_force_on_club_n: FloatArray
    constraint_rank: npt.NDArray[np.int64]
    position_constraint_norm_m: FloatArray
    velocity_constraint_norm_m_s: FloatArray
    kkt_residual_norm: FloatArray
    acceleration_constraint_residual_norm: FloatArray
    mechanical_energy_j: FloatArray
    projection_correction_norm_m: FloatArray
    projection_energy_change_j: FloatArray
    model_tier: str = "forward_two_arm_floating_club_closed_loop"
    counterfactual_kind: str = "forward_commanded"
    branch_source_index: int | None = None

    @property
    def maximum_projection_correction_m(self) -> float:
        """Return the largest mass-metric position correction."""
        return float(np.max(self.projection_correction_norm_m))


def constant_control(control: TwoArmControl) -> ControlLaw:
    """Return a state-independent control law."""

    def law(_time_s: float, _q: FloatArray, _qdot: FloatArray) -> TwoArmControl:
        return control

    return law


def potential_energy(q: object, params: TwoArmParams) -> float:
    """Return gravitational potential energy for both arms and the club."""
    state = _state("q", q)
    energy = params.club_mass_kg * params.gravity_m_s2 * state[5]
    for shoulder_index, elbow_index in ((0, 1), (2, 3)):
        shoulder = state[shoulder_index]
        forearm = shoulder + state[elbow_index]
        upper_y = -0.5 * params.upper_length_m * np.cos(shoulder)
        forearm_y = -params.upper_length_m * np.cos(shoulder) - (
            0.5 * params.forearm_length_m * np.cos(forearm)
        )
        energy += params.gravity_m_s2 * (
            params.upper_mass_kg * upper_y + params.forearm_mass_kg * forearm_y
        )
    return float(energy)


def mechanical_energy(q: object, qdot: object, params: TwoArmParams) -> float:
    """Return kinetic plus gravitational potential energy."""
    state = _state("q", q)
    velocity = _state("qdot", qdot)
    kinetic = 0.5 * float(velocity @ mass_matrix(state, params) @ velocity)
    return kinetic + potential_energy(state, params)


def _state(name: str, value: object) -> FloatArray:
    array = np.asarray(value, dtype=float)
    if array.shape != (N_COORDINATES,) or not np.all(np.isfinite(array)):
        raise ValueError(
            f"{name} must have shape ({N_COORDINATES},) with finite values"
        )
    return array.copy()


def _mass_metric_constraint_correction(
    q: FloatArray,
    residual: FloatArray,
    params: TwoArmParams,
) -> FloatArray:
    matrix = mass_matrix(q, params)
    jacobian = constraint_jacobian(q, params)
    inverse_jacobian = np.linalg.solve(matrix, jacobian.T)
    schur = jacobian @ inverse_jacobian
    if np.linalg.matrix_rank(schur, tol=params.rank_tolerance) != N_CONSTRAINTS:
        raise ValueError("constraint projection is singular; no fallback is allowed")
    return inverse_jacobian @ np.linalg.solve(schur, residual)


def _project_configuration(
    q: FloatArray,
    params: TwoArmParams,
    config: ForwardTwoArmConfig,
) -> tuple[FloatArray, float]:
    projected = q.copy()
    total_correction = np.zeros_like(projected)
    for _iteration in range(config.maximum_projection_iterations):
        residual = constraint_vector(projected, params)
        if np.linalg.norm(residual) <= config.projection_tolerance_m:
            return projected, float(np.linalg.norm(total_correction))
        correction = _mass_metric_constraint_correction(projected, residual, params)
        projected -= correction
        total_correction -= correction
    residual_norm = float(np.linalg.norm(constraint_vector(projected, params)))
    raise ValueError(
        "position projection failed to converge: "
        f"{residual_norm:.3e} m after {config.maximum_projection_iterations} iterations"
    )


def _project_velocity(
    q: FloatArray,
    qdot: FloatArray,
    params: TwoArmParams,
    config: ForwardTwoArmConfig,
) -> FloatArray:
    jacobian = constraint_jacobian(q, params)
    residual = jacobian @ qdot
    if np.linalg.norm(residual) <= config.velocity_tolerance_m_s:
        return qdot.copy()
    projected = qdot - _mass_metric_constraint_correction(q, residual, params)
    projected_residual = float(np.linalg.norm(jacobian @ projected))
    if projected_residual > config.velocity_tolerance_m_s:
        raise ValueError(
            f"velocity projection failed to converge: {projected_residual:.3e} m/s"
        )
    return projected


def rollout_forward_two_arm(
    q0: object,
    qdot0: object,
    control_law: ControlLaw,
    params: TwoArmParams,
    config: ForwardTwoArmConfig,
    *,
    counterfactual_kind: str = "forward_commanded",
    branch_source_index: int | None = None,
) -> ForwardTwoArmTrace:
    """Integrate constrained two-hand dynamics with recorded projections."""
    q_initial, initial_correction = _project_configuration(
        _state("q0", q0), params, config
    )
    velocity_initial = _project_velocity(
        q_initial, _state("qdot0", qdot0), params, config
    )
    intervals = config.interval_count
    samples = intervals + 1
    time = config.start_time_s + np.arange(samples, dtype=float) * config.step_s
    q = np.empty((samples, N_COORDINATES))
    qdot = np.empty_like(q)
    correction_norm = np.zeros(samples)
    projection_energy_change = np.zeros(samples)
    q[0], qdot[0] = q_initial, velocity_initial
    correction_norm[0] = initial_correction

    for index in range(intervals):
        control = control_law(float(time[index]), q[index].copy(), qdot[index].copy())
        solution = solve_constrained_dynamics(q[index], qdot[index], control, params)
        half_velocity = qdot[index] + 0.5 * config.step_s * solution.qddot
        trial_position = q[index] + config.step_s * half_velocity
        energy_before_position_projection = mechanical_energy(
            trial_position, half_velocity, params
        )
        q[index + 1], correction_norm[index + 1] = _project_configuration(
            trial_position, params, config
        )
        projected_half_velocity = _project_velocity(
            q[index + 1], half_velocity, params, config
        )
        position_projection_energy = (
            mechanical_energy(q[index + 1], projected_half_velocity, params)
            - energy_before_position_projection
        )
        next_control = control_law(
            float(time[index + 1]),
            q[index + 1].copy(),
            projected_half_velocity.copy(),
        )
        next_solution = solve_constrained_dynamics(
            q[index + 1], projected_half_velocity, next_control, params
        )
        trial_velocity = (
            projected_half_velocity + 0.5 * config.step_s * next_solution.qddot
        )
        energy_before_velocity_projection = mechanical_energy(
            q[index + 1], trial_velocity, params
        )
        qdot[index + 1] = _project_velocity(
            q[index + 1], trial_velocity, params, config
        )
        projection_energy_change[index + 1] = (
            mechanical_energy(q[index + 1], qdot[index + 1], params)
            - energy_before_velocity_projection
            + position_projection_energy
        )

    controls = tuple(
        control_law(float(sample_time), state.copy(), velocity.copy())
        for sample_time, state, velocity in zip(time, q, qdot, strict=True)
    )
    qddot = np.empty_like(q)
    multipliers = np.empty((samples, N_CONSTRAINTS))
    contacts = np.empty((samples, 2, 2))
    ranks = np.empty(samples, dtype=np.int64)
    kkt = np.empty(samples)
    acceleration_residual = np.empty(samples)
    position_residual = np.empty(samples)
    velocity_residual = np.empty(samples)
    energy = np.empty(samples)
    for index, (state, velocity, control) in enumerate(
        zip(q, qdot, controls, strict=True)
    ):
        solution = solve_constrained_dynamics(state, velocity, control, params)
        qddot[index] = solution.qddot
        multipliers[index] = solution.multipliers_n
        contacts[index] = solution.contact_force_on_club_n
        ranks[index] = solution.constraint_rank
        kkt[index] = solution.kkt_residual_norm
        acceleration_residual[index] = solution.acceleration_constraint_residual_norm
        position_residual[index] = np.linalg.norm(constraint_vector(state, params))
        velocity_residual[index] = np.linalg.norm(
            constraint_jacobian(state, params) @ velocity
        )
        energy[index] = mechanical_energy(state, velocity, params)
    return ForwardTwoArmTrace(
        time=time,
        q=q,
        qdot=qdot,
        qddot=qddot,
        controls=controls,
        multipliers_n=multipliers,
        contact_force_on_club_n=contacts,
        constraint_rank=ranks,
        position_constraint_norm_m=position_residual,
        velocity_constraint_norm_m_s=velocity_residual,
        kkt_residual_norm=kkt,
        acceleration_constraint_residual_norm=acceleration_residual,
        mechanical_energy_j=energy,
        projection_correction_norm_m=correction_norm,
        projection_energy_change_j=projection_energy_change,
        counterfactual_kind=counterfactual_kind,
        branch_source_index=branch_source_index,
    )


def branch_zero_command(
    source: ForwardTwoArmTrace,
    *,
    cut_index: int,
    horizon_s: float,
    params: TwoArmParams,
) -> ForwardTwoArmTrace:
    """Branch a forward zero-command trajectory from an achieved source state."""
    if cut_index < 0 or cut_index >= source.time.size:
        raise IndexError("cut_index must identify a source trajectory sample")
    if source.time.size < 2:
        raise ValueError("source trajectory must contain at least two samples")
    step_s = float(source.time[1] - source.time[0])
    config = ForwardTwoArmConfig(
        duration_s=horizon_s,
        step_s=step_s,
        start_time_s=float(source.time[cut_index]),
    )
    return rollout_forward_two_arm(
        source.q[cut_index],
        source.qdot[cut_index],
        constant_control(TwoArmControl.zero()),
        params,
        config,
        counterfactual_kind="forward_branched_zero_command",
        branch_source_index=cut_index,
    )


__all__ = [
    "ForwardTwoArmConfig",
    "ForwardTwoArmTrace",
    "branch_zero_command",
    "constant_control",
    "mechanical_energy",
    "potential_energy",
    "rollout_forward_two_arm",
]
