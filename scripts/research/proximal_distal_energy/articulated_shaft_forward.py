"""Bounded articulated forwarding with distributed grip and elastic shaft."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from scripts.research.proximal_distal_energy.articulated_distributed_forward import (
    DistributedForwardConfig,
    DistributedIntegrationCase,
    integrate_distributed_grip,
)
from scripts.research.proximal_distal_energy.articulated_distributed_grip import (
    DistributedGripConfig,
    distributed_reference_lengths,
    evaluate_distributed_grip,
)
from scripts.research.proximal_distal_energy.articulated_forward_integration import (
    native_dynamics_operator,
)
from scripts.research.proximal_distal_energy.articulated_shaft import (
    ArticulatedShaftConfig,
    ArticulatedShaftProperties,
    augmented_mass_matrix,
    build_articulated_shaft,
    extra_potential_gradient,
    mass_increment_coriolis,
    shaft_state_energy,
)
from scripts.research.proximal_distal_energy.spatial_full_body import SpatialModel

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class ShaftForwardConfig:
    """Horizon and numerical controls for the shaft-coupled tier."""

    duration_s: float = 0.05
    time_steps_s: tuple[float, ...] = (0.00025, 0.000125)
    virtual_power_tolerance_w: float = 1.0e-10
    normalized_energy_residual_tolerance: float = 5.0e-2
    trajectory_relative_tolerance: float = 1.0e-7
    refinement_ratio_limit: float = 0.9

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
        for name in (
            "virtual_power_tolerance_w",
            "normalized_energy_residual_tolerance",
            "trajectory_relative_tolerance",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not 0.0 < self.refinement_ratio_limit < 1.0:
            raise ValueError("refinement_ratio_limit must lie in (0, 1)")


@dataclass(frozen=True, slots=True)
class ShaftIntegrationCase:
    """Rigid initial state, distributed contact, and shaft activation."""

    q: FloatArray
    qd: FloatArray
    grip_span_m: float
    hand_contact_local_x_m: float
    time_step_s: float
    initial_club_displacement_m: float
    initial_club_velocity_m_s: float
    engine: str
    grip: DistributedGripConfig
    shaft: ArticulatedShaftConfig


@dataclass(slots=True)
class _Buffers:
    q: FloatArray
    qd: FloatArray
    eta: FloatArray
    eta_dot: FloatArray
    force: FloatArray
    active_count: NDArray[np.int_]
    couple: FloatArray
    grip_strain: FloatArray
    grip_dissipation: FloatArray
    virtual_power: FloatArray
    shaft_strain: FloatArray
    shaft_damping: FloatArray
    shaft_power_residual: FloatArray
    mechanical: FloatArray
    bending: FloatArray
    twist: FloatArray


def _validate(
    model: SpatialModel, case: ShaftIntegrationCase, config: ShaftForwardConfig
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
    )
    if any(not np.isfinite(value) for value in scalars):
        raise ValueError("integration scalars must be finite")
    if min(case.grip_span_m, case.hand_contact_local_x_m, case.time_step_s) <= 0.0:
        raise ValueError("contact geometry and time step must be positive")
    steps = int(round(config.duration_s / case.time_step_s))
    if not np.isclose(steps * case.time_step_s, config.duration_s):
        raise ValueError("time_step_s must divide duration_s")
    return steps


def _buffers(samples: int, nq: int, ne: int) -> _Buffers:
    return _Buffers(
        q=np.empty((samples, nq)),
        qd=np.empty((samples, nq)),
        eta=np.empty((samples, ne)),
        eta_dot=np.empty((samples, ne)),
        force=np.empty(samples),
        active_count=np.empty(samples, dtype=int),
        couple=np.empty((samples, 3)),
        grip_strain=np.empty(samples),
        grip_dissipation=np.empty(samples),
        virtual_power=np.empty(samples),
        shaft_strain=np.empty(samples),
        shaft_damping=np.empty(samples),
        shaft_power_residual=np.empty(samples),
        mechanical=np.empty(samples),
        bending=np.empty((samples, 2)),
        twist=np.empty(samples),
    )


def _full_elastic(
    eta: FloatArray, properties: ArticulatedShaftProperties
) -> FloatArray:
    full = np.zeros(3)
    full[np.asarray(properties.active_full_indices, dtype=int)] = eta
    return full


def _record(
    buffers: _Buffers,
    index: int,
    model: SpatialModel,
    q: FloatArray,
    qd: FloatArray,
    eta: FloatArray,
    eta_dot: FloatArray,
    contact: Any,
    properties: ArticulatedShaftProperties,
) -> None:
    energy = shaft_state_energy(model, q, qd, eta, eta_dot, properties)
    elastic_force = (
        -properties.elastic_stiffness @ eta - properties.elastic_damping @ eta_dot
    )
    storage_power = float(properties.elastic_stiffness @ eta @ eta_dot)
    damping_power = -float(eta_dot @ properties.elastic_damping @ eta_dot)
    full = _full_elastic(eta, properties)
    buffers.q[index], buffers.qd[index] = q, qd
    buffers.eta[index], buffers.eta_dot[index] = eta, eta_dot
    buffers.force[index] = contact.maximum_station_force_n
    buffers.active_count[index] = contact.active_station_count
    buffers.couple[index] = contact.force_couple_vector_nm
    buffers.grip_strain[index] = contact.strain_energy_j
    buffers.grip_dissipation[index] = contact.dissipation_power_w
    buffers.virtual_power[index] = contact.virtual_power_residual_w
    buffers.shaft_strain[index] = energy.elastic_strain_j
    buffers.shaft_damping[index] = damping_power
    buffers.shaft_power_residual[index] = abs(
        float(elastic_force @ eta_dot) + storage_power - damping_power
    )
    buffers.mechanical[index] = energy.total_mechanical_j
    buffers.bending[index] = full[:2]
    buffers.twist[index] = full[2]


def _result(
    buffers: _Buffers,
    case: ShaftIntegrationCase,
    properties: ArticulatedShaftProperties,
) -> dict[str, NDArray[Any] | str | int | float]:
    dissipation = buffers.grip_dissipation + buffers.shaft_damping
    cumulative = np.zeros(dissipation.size)
    cumulative[1:] = np.cumsum(
        0.5 * (dissipation[1:] + dissipation[:-1]) * case.time_step_s
    )
    total = buffers.mechanical + buffers.grip_strain
    bending_norm = np.linalg.norm(buffers.bending, axis=1)
    active = buffers.active_count > 0
    transitions = np.zeros(active.size, dtype=bool)
    transitions[1:] = active[1:] != active[:-1]
    return {
        "time_s": np.arange(total.size) * case.time_step_s,
        "q": buffers.q,
        "qd": buffers.qd,
        "elastic_coordinates": buffers.eta,
        "elastic_velocities": buffers.eta_dot,
        "active_labels": np.asarray(properties.active_labels),
        "maximum_station_force_n": buffers.force,
        "active_station_count": buffers.active_count,
        "active_set_transition": transitions,
        "force_couple_vector_nm": buffers.couple,
        "grip_strain_energy_j": buffers.grip_strain,
        "grip_dissipation_power_w": buffers.grip_dissipation,
        "virtual_power_residual_w": buffers.virtual_power,
        "shaft_strain_energy_j": buffers.shaft_strain,
        "shaft_damping_power_w": buffers.shaft_damping,
        "shaft_power_residual_w": buffers.shaft_power_residual,
        "tip_bending_m": buffers.bending,
        "twist_angle_rad": buffers.twist,
        "small_deflection_ratio": bending_norm / case.shaft.shaft_length_m,
        "total_mechanical_energy_j": buffers.mechanical,
        "total_energy_j": total,
        "cumulative_dissipation_j": cumulative,
        "work_energy_residual_j": total - total[0] - cumulative,
    }


def _rigid_trace(
    model: SpatialModel, case: ShaftIntegrationCase, config: ShaftForwardConfig
) -> dict[str, NDArray[Any] | str | int | float]:
    distributed = DistributedIntegrationCase(
        q=case.q,
        qd=case.qd,
        grip_span_m=case.grip_span_m,
        hand_contact_local_x_m=case.hand_contact_local_x_m,
        time_step_s=case.time_step_s,
        initial_club_displacement_m=case.initial_club_displacement_m,
        initial_club_velocity_m_s=case.initial_club_velocity_m_s,
        engine=case.engine,
        grip=case.grip,
    )
    result = integrate_distributed_grip(
        model,
        distributed,
        DistributedForwardConfig(
            duration_s=config.duration_s,
            time_steps_s=config.time_steps_s,
        ),
    )
    samples = np.asarray(result["time_s"]).size
    result["elastic_coordinates"] = np.empty((samples, 0))
    result["elastic_velocities"] = np.empty((samples, 0))
    result["active_labels"] = np.asarray((), dtype=str)
    result["grip_strain_energy_j"] = result.pop("strain_energy_j")
    result["grip_dissipation_power_w"] = result.pop("dissipation_power_w")
    result["shaft_strain_energy_j"] = np.zeros(samples)
    result["shaft_damping_power_w"] = np.zeros(samples)
    result["shaft_power_residual_w"] = np.zeros(samples)
    result["tip_bending_m"] = np.zeros((samples, 2))
    result["twist_angle_rad"] = np.zeros(samples)
    result["small_deflection_ratio"] = np.zeros(samples)
    result["total_mechanical_energy_j"] = result.pop("mechanical_energy_j")
    return result


def integrate_articulated_shaft(
    model: SpatialModel,
    case: ShaftIntegrationCase,
    config: ShaftForwardConfig = ShaftForwardConfig(),
) -> dict[str, NDArray[Any] | str | int | float]:
    """Advance one native rigid operator plus shared passive elastic states."""

    step_count = _validate(model, case, config)
    properties = build_articulated_shaft(model, case.shaft)
    if properties.coordinate_count == 0:
        return _rigid_trace(model, case, config)
    reference = distributed_reference_lengths(
        model,
        case.q,
        grip_span_m=case.grip_span_m,
        hand_contact_local_x_m=case.hand_contact_local_x_m,
        config=case.grip,
    )
    q, qd = np.asarray(case.q).copy(), np.asarray(case.qd).copy()
    q[14] += case.initial_club_displacement_m
    qd[14] += case.initial_club_velocity_m_s
    eta = np.zeros(properties.coordinate_count)
    eta_dot = np.zeros(properties.coordinate_count)
    operator = native_dynamics_operator(case.engine, model)
    buffers = _buffers(step_count + 1, model.nq, properties.coordinate_count)
    for index in range(step_count + 1):
        contact = evaluate_distributed_grip(
            model,
            q,
            qd,
            grip_span_m=case.grip_span_m,
            hand_contact_local_x_m=case.hand_contact_local_x_m,
            reference_lengths_m=reference,
            config=case.grip,
        )
        _record(buffers, index, model, q, qd, eta, eta_dot, contact, properties)
        if index == step_count:
            continue
        rigid_mass, rigid_bias = operator(q, qd)
        matrix = augmented_mass_matrix(model, q, rigid_mass, properties)
        rhs = np.concatenate(
            (
                contact.generalized_contact_force - rigid_bias,
                -properties.elastic_stiffness @ eta
                - properties.elastic_damping @ eta_dot,
            )
        )
        rhs -= mass_increment_coriolis(model, q, qd, eta_dot, properties)
        rhs -= extra_potential_gradient(model, q, eta, properties)
        acceleration = np.linalg.solve(matrix, rhs)
        qd = qd + case.time_step_s * acceleration[: model.nq]
        eta_dot = eta_dot + case.time_step_s * acceleration[model.nq :]
        q = q + case.time_step_s * qd
        eta = eta + case.time_step_s * eta_dot
        full_eta = _full_elastic(eta, properties)
        if (
            np.linalg.norm(full_eta[:2]) / case.shaft.shaft_length_m
            > case.shaft.small_deflection_limit
            or abs(full_eta[2]) > case.shaft.twist_limit_rad
        ):
            raise RuntimeError("linear shaft domain exceeded during integration")
    return _result(buffers, case, properties)


__all__ = [
    "ShaftForwardConfig",
    "ShaftIntegrationCase",
    "integrate_articulated_shaft",
]
