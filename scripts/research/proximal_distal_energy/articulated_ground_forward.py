"""Forward integration of distributed grip, passive shaft, and finite ground."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from scripts.research.proximal_distal_energy.articulated_distributed_grip import (
    DistributedGripConfig,
    distributed_reference_lengths,
)
from scripts.research.proximal_distal_energy.articulated_forward_integration import (
    native_dynamics_operator,
)
from scripts.research.proximal_distal_energy.articulated_ground import (
    ArticulatedGroundConfig,
    ArticulatedGroundProperties,
    augmented_ground_mass_matrix,
    build_articulated_ground,
    evaluate_ground_coupled_grip,
    evaluate_ground_wrench,
    ground_extra_potential_gradient,
    ground_mass_increment_coriolis,
    ground_state_energy,
)
from scripts.research.proximal_distal_energy.articulated_shaft import (
    ArticulatedShaftConfig,
    ArticulatedShaftProperties,
    augmented_mass_matrix,
    build_articulated_shaft,
    extra_potential_gradient,
    mass_increment_coriolis,
)
from scripts.research.proximal_distal_energy.articulated_shaft_forward import (
    ShaftForwardConfig,
    ShaftIntegrationCase,
    integrate_articulated_shaft,
)
from scripts.research.proximal_distal_energy.spatial_full_body import SpatialModel

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class GroundForwardConfig:
    """Horizon, refinement, and work-energy controls."""

    duration_s: float = 0.05
    time_steps_s: tuple[float, ...] = (0.00025, 0.000125)
    normalized_energy_residual_tolerance: float = 5.0e-2

    def __post_init__(self) -> None:
        if not np.isfinite(self.duration_s) or self.duration_s <= 0.0:
            raise ValueError("duration_s must be finite and positive")
        steps = np.asarray(self.time_steps_s, dtype=float)
        if (
            steps.ndim != 1
            or steps.size < 2
            or np.any(~np.isfinite(steps))
            or np.any(steps <= 0.0)
            or np.any(np.diff(steps) >= 0.0)
            or not np.allclose(
                self.duration_s / steps, np.rint(self.duration_s / steps)
            )
        ):
            raise ValueError(
                "time_steps_s must be decreasing positive divisors of duration_s"
            )
        if (
            not np.isfinite(self.normalized_energy_residual_tolerance)
            or self.normalized_energy_residual_tolerance <= 0.0
        ):
            raise ValueError("normalized energy tolerance must be finite and positive")


@dataclass(frozen=True, slots=True)
class GroundIntegrationCase:
    """Common rigid state and declared passive shaft/base activations."""

    q: FloatArray
    qd: FloatArray
    grip_span_m: float
    hand_contact_local_x_m: float
    time_step_s: float
    initial_club_displacement_m: float
    initial_club_velocity_m_s: float
    initial_base_displacement: tuple[float, float, float]
    initial_base_velocity: tuple[float, float, float]
    engine: str
    grip: DistributedGripConfig
    shaft: ArticulatedShaftConfig
    ground: ArticulatedGroundConfig


@dataclass(frozen=True, slots=True)
class BaseEquilibrium:
    """Conditional fixed-posture balance of ground, grip, and gravity."""

    base_coordinates: tuple[float, float, float]
    residual_generalized_force: tuple[float, float, float]
    residual_norm: float
    iteration_count: int
    active_station_count: int
    maximum_station_force_n: float


def _validate(
    model: SpatialModel, case: GroundIntegrationCase, config: GroundForwardConfig
) -> int:
    if np.asarray(case.q).shape != (model.nq,) or np.asarray(case.qd).shape != (
        model.nq,
    ):
        raise ValueError("q and qd must match the articulated model dimension")
    scalars = (
        case.grip_span_m,
        case.hand_contact_local_x_m,
        case.time_step_s,
        case.initial_club_displacement_m,
        case.initial_club_velocity_m_s,
        *case.initial_base_displacement,
        *case.initial_base_velocity,
    )
    if any(not np.isfinite(value) for value in scalars):
        raise ValueError("integration values must be finite")
    if min(case.grip_span_m, case.hand_contact_local_x_m, case.time_step_s) <= 0.0:
        raise ValueError("contact geometry and time step must be positive")
    if len(case.initial_base_displacement) != 3 or len(case.initial_base_velocity) != 3:
        raise ValueError(
            "initial base displacement and velocity must have length three"
        )
    steps = int(round(config.duration_s / case.time_step_s))
    if not np.isclose(steps * case.time_step_s, config.duration_s):
        raise ValueError("time_step_s must divide duration_s")
    return steps


def _active_base(
    values: tuple[float, float, float], properties: ArticulatedGroundProperties
) -> FloatArray:
    full = np.asarray(values, dtype=float)
    active = np.asarray(properties.active_full_indices, dtype=int)
    inactive = np.delete(full, active)
    if np.any(np.abs(inactive) > 0.0):
        raise ValueError("inactive base coordinates must have zero initial state")
    return full[active]


def _full_elastic(
    eta: FloatArray, properties: ArticulatedShaftProperties
) -> FloatArray:
    full = np.zeros(3)
    full[np.asarray(properties.active_full_indices, dtype=int)] = eta
    return full


def solve_conditional_base_equilibrium(
    model: SpatialModel,
    q: FloatArray,
    *,
    grip_span_m: float,
    hand_contact_local_x_m: float,
    grip_config: DistributedGripConfig,
    shaft_config: ArticulatedShaftConfig = ArticulatedShaftConfig(),
    ground_config: ArticulatedGroundConfig = ArticulatedGroundConfig(),
    tolerance_n: float = 1.0e-6,
    maximum_iterations: int = 40,
) -> BaseEquilibrium:
    """Solve the base balance with posture and club held at the authority state."""

    if ground_config.activation != "coupled":
        raise ValueError("conditional equilibrium requires coupled ground activation")
    if not np.isfinite(tolerance_n) or tolerance_n <= 0.0:
        raise ValueError("tolerance_n must be finite and positive")
    if not isinstance(maximum_iterations, int) or maximum_iterations < 1:
        raise ValueError("maximum_iterations must be a positive integer")
    q = np.asarray(q, dtype=float)
    shaft = build_articulated_shaft(model, shaft_config)
    ground = build_articulated_ground(ground_config)
    reference = distributed_reference_lengths(
        model,
        q,
        grip_span_m=grip_span_m,
        hand_contact_local_x_m=hand_contact_local_x_m,
        config=grip_config,
    )
    offset = model.nq + shaft.coordinate_count

    def residual(base: FloatArray) -> tuple[FloatArray, Any]:
        contact = evaluate_ground_coupled_grip(
            model,
            q,
            np.zeros(model.nq),
            np.zeros(shaft.coordinate_count),
            base,
            np.zeros(ground.coordinate_count),
            shaft,
            ground,
            grip_span_m=grip_span_m,
            hand_contact_local_x_m=hand_contact_local_x_m,
            reference_lengths_m=reference,
            config=grip_config,
        )
        wrench = evaluate_ground_wrench(base, np.zeros(ground.coordinate_count), ground)
        gravity = ground_extra_potential_gradient(model, q, base, shaft, ground)[
            offset:
        ]
        value = (
            wrench.generalized_force
            + contact.generalized_contact_force[-ground.coordinate_count :]
            - gravity
        )
        return value, contact

    gravity_at_zero = ground_extra_potential_gradient(
        model, q, np.zeros(ground.coordinate_count), shaft, ground
    )[offset:]
    base = -np.linalg.solve(ground.stiffness, gravity_at_zero)
    lower = np.array(
        [
            -ground_config.translation_limit_m,
            -ground_config.translation_limit_m,
            -ground_config.rotation_limit_rad,
        ]
    )
    upper = -lower
    base = np.clip(base, lower, upper)
    contact: Any = None
    for iteration in range(1, maximum_iterations + 1):
        value, contact = residual(base)
        norm = float(np.linalg.norm(value))
        if norm <= tolerance_n:
            return BaseEquilibrium(
                base_coordinates=tuple(float(item) for item in base),  # type: ignore[arg-type]
                residual_generalized_force=tuple(float(item) for item in value),  # type: ignore[arg-type]
                residual_norm=norm,
                iteration_count=iteration,
                active_station_count=int(contact.active_station_count),
                maximum_station_force_n=float(contact.maximum_station_force_n),
            )
        jacobian = np.empty((ground.coordinate_count, ground.coordinate_count))
        step = ground_config.derivative_step
        for index in range(ground.coordinate_count):
            delta = np.zeros(ground.coordinate_count)
            delta[index] = step
            jacobian[:, index] = (
                residual(np.clip(base + delta, lower, upper))[0]
                - residual(np.clip(base - delta, lower, upper))[0]
            ) / (2.0 * step)
        update = np.linalg.solve(jacobian, -value)
        scale = 1.0
        accepted = False
        while scale >= 1.0 / 1024.0:
            candidate = np.clip(base + scale * update, lower, upper)
            if np.linalg.norm(residual(candidate)[0]) < norm:
                base = candidate
                accepted = True
                break
            scale *= 0.5
        if not accepted:
            raise RuntimeError("conditional base equilibrium line search stalled")
    raise RuntimeError("conditional base equilibrium did not converge")


def _fixed_trace(
    model: SpatialModel, case: GroundIntegrationCase, config: GroundForwardConfig
) -> dict[str, NDArray[Any] | str | int | float]:
    if any(case.initial_base_displacement) or any(case.initial_base_velocity):
        raise ValueError("fixed base requires zero initial base state")
    shaft_case = ShaftIntegrationCase(
        q=case.q,
        qd=case.qd,
        grip_span_m=case.grip_span_m,
        hand_contact_local_x_m=case.hand_contact_local_x_m,
        time_step_s=case.time_step_s,
        initial_club_displacement_m=case.initial_club_displacement_m,
        initial_club_velocity_m_s=case.initial_club_velocity_m_s,
        engine=case.engine,
        grip=case.grip,
        shaft=case.shaft,
    )
    result = integrate_articulated_shaft(
        model,
        shaft_case,
        ShaftForwardConfig(
            duration_s=config.duration_s,
            time_steps_s=config.time_steps_s,
            normalized_energy_residual_tolerance=(
                config.normalized_energy_residual_tolerance
            ),
        ),
    )
    samples = np.asarray(result["time_s"]).size
    result.update(
        {
            "base_coordinates": np.empty((samples, 0)),
            "base_velocities": np.empty((samples, 0)),
            "ground_active_labels": np.asarray((), dtype=str),
            "ground_force_n": np.zeros((samples, 3)),
            "ground_intrinsic_free_moment_nm": np.zeros(samples),
            "ground_transported_moment_nm": np.zeros(samples),
            "ground_strain_energy_j": np.zeros(samples),
            "ground_damping_power_w": np.zeros(samples),
            "ground_power_residual_w": np.zeros(samples),
            "base_translation_m": np.zeros((samples, 2)),
            "base_pitch_rad": np.zeros(samples),
        }
    )
    return result


def integrate_articulated_ground(
    model: SpatialModel,
    case: GroundIntegrationCase,
    config: GroundForwardConfig = GroundForwardConfig(),
) -> dict[str, NDArray[Any] | str | int | float]:
    """Advance one common rigid/shaft/base trajectory with full ledgers."""

    step_count = _validate(model, case, config)
    shaft = build_articulated_shaft(model, case.shaft)
    ground = build_articulated_ground(case.ground)
    if ground.coordinate_count == 0:
        return _fixed_trace(model, case, config)
    reference = distributed_reference_lengths(
        model,
        case.q,
        grip_span_m=case.grip_span_m,
        hand_contact_local_x_m=case.hand_contact_local_x_m,
        config=case.grip,
    )
    q, qd = (
        np.asarray(case.q, dtype=float).copy(),
        np.asarray(case.qd, dtype=float).copy(),
    )
    q[14] += case.initial_club_displacement_m
    qd[14] += case.initial_club_velocity_m_s
    eta = np.zeros(shaft.coordinate_count)
    eta_dot = np.zeros(shaft.coordinate_count)
    base = _active_base(case.initial_base_displacement, ground)
    base_velocity = _active_base(case.initial_base_velocity, ground)
    samples = step_count + 1
    arrays: dict[str, FloatArray] = {
        "q": np.empty((samples, model.nq)),
        "qd": np.empty((samples, model.nq)),
        "elastic_coordinates": np.empty((samples, shaft.coordinate_count)),
        "elastic_velocities": np.empty((samples, shaft.coordinate_count)),
        "base_coordinates": np.empty((samples, ground.coordinate_count)),
        "base_velocities": np.empty((samples, ground.coordinate_count)),
        "maximum_station_force_n": np.empty(samples),
        "active_station_count": np.empty(samples),
        "force_couple_vector_nm": np.empty((samples, 3)),
        "grip_strain_energy_j": np.empty(samples),
        "grip_dissipation_power_w": np.empty(samples),
        "virtual_power_residual_w": np.empty(samples),
        "shaft_strain_energy_j": np.empty(samples),
        "shaft_damping_power_w": np.empty(samples),
        "shaft_power_residual_w": np.empty(samples),
        "ground_force_n": np.empty((samples, 3)),
        "ground_intrinsic_free_moment_nm": np.empty(samples),
        "ground_transported_moment_nm": np.empty(samples),
        "ground_strain_energy_j": np.empty(samples),
        "ground_damping_power_w": np.empty(samples),
        "ground_power_residual_w": np.empty(samples),
        "total_mechanical_energy_j": np.empty(samples),
        "tip_bending_m": np.empty((samples, 2)),
        "twist_angle_rad": np.empty(samples),
        "base_translation_m": np.empty((samples, 2)),
        "base_pitch_rad": np.empty(samples),
    }
    ground_intrinsic_free_moment = arrays["ground_intrinsic_free_moment_nm"]
    operator = native_dynamics_operator(case.engine, model)
    for index in range(samples):
        contact = evaluate_ground_coupled_grip(
            model,
            q,
            qd,
            eta_dot,
            base,
            base_velocity,
            shaft,
            ground,
            grip_span_m=case.grip_span_m,
            hand_contact_local_x_m=case.hand_contact_local_x_m,
            reference_lengths_m=reference,
            config=case.grip,
        )
        wrench = evaluate_ground_wrench(base, base_velocity, ground)
        energy = ground_state_energy(
            model, q, qd, eta, eta_dot, base, base_velocity, shaft, ground
        )
        elastic_force = -shaft.elastic_stiffness @ eta - shaft.elastic_damping @ eta_dot
        shaft_storage_power = float(shaft.elastic_stiffness @ eta @ eta_dot)
        shaft_damping_power = -float(eta_dot @ shaft.elastic_damping @ eta_dot)
        full_eta = _full_elastic(eta, shaft)
        full_base = np.zeros(3)
        full_base[np.asarray(ground.active_full_indices, dtype=int)] = base
        arrays["q"][index], arrays["qd"][index] = q, qd
        arrays["elastic_coordinates"][index] = eta
        arrays["elastic_velocities"][index] = eta_dot
        arrays["base_coordinates"][index] = base
        arrays["base_velocities"][index] = base_velocity
        arrays["maximum_station_force_n"][index] = contact.maximum_station_force_n
        arrays["active_station_count"][index] = contact.active_station_count
        arrays["force_couple_vector_nm"][index] = contact.force_couple_vector_nm
        arrays["grip_strain_energy_j"][index] = contact.strain_energy_j
        arrays["grip_dissipation_power_w"][index] = contact.dissipation_power_w
        arrays["virtual_power_residual_w"][index] = contact.virtual_power_residual_w
        arrays["shaft_strain_energy_j"][index] = energy.shaft_energy.elastic_strain_j
        arrays["shaft_damping_power_w"][index] = shaft_damping_power
        arrays["shaft_power_residual_w"][index] = abs(
            float(elastic_force @ eta_dot) + shaft_storage_power - shaft_damping_power
        )
        arrays["ground_force_n"][index] = wrench.force_n
        ground_intrinsic_free_moment[index] = wrench.intrinsic_free_moment_nm
        arrays["ground_transported_moment_nm"][index] = wrench.transported_moment_nm
        arrays["ground_strain_energy_j"][index] = wrench.strain_energy_j
        arrays["ground_damping_power_w"][index] = wrench.damping_power_w
        arrays["ground_power_residual_w"][index] = abs(wrench.power_residual_w)
        arrays["total_mechanical_energy_j"][index] = energy.total_mechanical_j
        arrays["tip_bending_m"][index] = full_eta[:2]
        arrays["twist_angle_rad"][index] = full_eta[2]
        arrays["base_translation_m"][index] = full_base[:2]
        arrays["base_pitch_rad"][index] = full_base[2]
        if index == step_count:
            continue
        rigid_mass, rigid_bias = operator(q, qd)
        shaft_mass = augmented_mass_matrix(model, q, rigid_mass, shaft)
        matrix = augmented_ground_mass_matrix(model, q, shaft_mass, shaft, ground)
        rhs = np.concatenate(
            (
                contact.generalized_contact_force[: model.nq] - rigid_bias,
                elastic_force,
                wrench.generalized_force,
            )
        )
        shaft_bias = mass_increment_coriolis(model, q, qd, eta_dot, shaft)
        shaft_gravity = extra_potential_gradient(model, q, eta, shaft)
        rhs[: shaft_bias.size] -= shaft_bias + shaft_gravity
        rhs -= ground_mass_increment_coriolis(
            model, q, qd, eta_dot, base_velocity, shaft, ground
        )
        rhs -= ground_extra_potential_gradient(model, q, base, shaft, ground)
        # The augmented contact evaluator supplies equal-and-opposite base reaction.
        rhs[-ground.coordinate_count :] += contact.generalized_contact_force[
            -ground.coordinate_count :
        ]
        acceleration = np.linalg.solve(matrix, rhs)
        qd += case.time_step_s * acceleration[: model.nq]
        eta_dot += (
            case.time_step_s
            * acceleration[model.nq : model.nq + shaft.coordinate_count]
        )
        base_velocity += case.time_step_s * acceleration[-ground.coordinate_count :]
        q += case.time_step_s * qd
        eta += case.time_step_s * eta_dot
        base += case.time_step_s * base_velocity
        full_eta = _full_elastic(eta, shaft)
        full_base = np.zeros(3)
        full_base[np.asarray(ground.active_full_indices, dtype=int)] = base
        if (
            np.linalg.norm(full_eta[:2]) / case.shaft.shaft_length_m
            > case.shaft.small_deflection_limit
            or abs(full_eta[2]) > case.shaft.twist_limit_rad
        ):
            raise RuntimeError("linear shaft domain exceeded during ground integration")
        if (
            np.linalg.norm(full_base[:2]) > case.ground.translation_limit_m
            or abs(full_base[2]) > case.ground.rotation_limit_rad
        ):
            raise RuntimeError("finite base domain exceeded during integration")
    dissipation = (
        arrays["grip_dissipation_power_w"]
        + arrays["shaft_damping_power_w"]
        + arrays["ground_damping_power_w"]
    )
    cumulative = np.zeros(samples)
    cumulative[1:] = np.cumsum(
        0.5 * (dissipation[1:] + dissipation[:-1]) * case.time_step_s
    )
    total = arrays["total_mechanical_energy_j"] + arrays["grip_strain_energy_j"]
    active = arrays["active_station_count"] > 0
    transitions = np.zeros(samples, dtype=bool)
    transitions[1:] = active[1:] != active[:-1]
    result: dict[str, NDArray[Any] | str | int | float] = {
        "time_s": np.arange(samples) * case.time_step_s,
        **arrays,
        "active_labels": np.asarray(shaft.active_labels),
        "ground_active_labels": np.asarray(ground.active_labels),
        "active_set_transition": transitions,
        "small_deflection_ratio": np.linalg.norm(arrays["tip_bending_m"], axis=1)
        / case.shaft.shaft_length_m,
        "total_energy_j": total,
        "cumulative_dissipation_j": cumulative,
        "work_energy_residual_j": total - total[0] - cumulative,
    }
    return result


__all__ = [
    "BaseEquilibrium",
    "GroundForwardConfig",
    "GroundIntegrationCase",
    "integrate_articulated_ground",
    "solve_conditional_base_equilibrium",
]
