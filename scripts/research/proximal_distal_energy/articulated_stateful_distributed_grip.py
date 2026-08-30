"""Stateful tangential-compliance adapter for the distributed grip model."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import cast

import numpy as np
from numpy.typing import NDArray

from scripts.research.proximal_distal_energy.articulated_distributed_grip import (
    DistributedGripConfig,
    DistributedGripSnapshot,
    distributed_contact_kinematics,
    evaluate_distributed_grip,
)
from scripts.research.proximal_distal_energy.articulated_stateful_friction import (
    StatefulFrictionConfig,
    TangentialState,
    advance_stateful_friction,
)
from scripts.research.proximal_distal_energy.spatial_full_body import SpatialModel

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class StatefulDistributedGripInput:
    """Immutable geometry and constitutive inputs for one contact increment."""

    grip_span_m: float
    hand_contact_local_x_m: float
    reference_lengths_m: FloatArray
    grip_config: DistributedGripConfig
    friction_config: StatefulFrictionConfig
    time_step_s: float


@dataclass(frozen=True, slots=True)
class StatefulDistributedGripStep:
    """Combined normal/tangential loads and complete retained state ledger."""

    generalized_contact_force: FloatArray
    normal_generalized_contact_force: FloatArray
    tangential_generalized_contact_force: FloatArray
    force_on_club_n: FloatArray
    normal_force_on_club_n: FloatArray
    tangential_force_on_club_n: FloatArray
    active_station: NDArray[np.bool_]
    station_signed_gap_m: FloatArray
    elastic_displacement_m: FloatArray
    regimes: NDArray[np.str_]
    friction_limit_n: FloatArray
    yield_margin_n: FloatArray
    plastic_slip_increment_m: FloatArray
    elastic_energy_change_j: FloatArray
    frictional_dissipation_j: FloatArray
    release_dissipation_j: FloatArray
    constitutive_work_j: FloatArray
    normal_strain_energy_j: float
    normal_dissipation_power_w: float
    normal_power_w: float
    virtual_power_residual_w: float
    static_stick_modeled: bool = True
    human_or_anatomical_inference: bool = False


@dataclass(slots=True)
class _StationBuffers:
    tangential_force: FloatArray
    next_state: FloatArray
    regimes: NDArray[np.str_]
    friction_limit: FloatArray
    yield_margin: FloatArray
    plastic_slip: FloatArray
    energy_change: FloatArray
    friction_dissipation: FloatArray
    release_dissipation: FloatArray
    constitutive_work: FloatArray
    tangent_generalized: FloatArray
    physical_power: float = 0.0


def _buffers(model: SpatialModel, grip: DistributedGripConfig) -> _StationBuffers:
    station_shape = (2, grip.station_count_per_hand)
    point_shape = (*station_shape, 3)
    return _StationBuffers(
        tangential_force=np.zeros(point_shape),
        next_state=np.zeros(point_shape),
        regimes=np.empty(station_shape, dtype="U16"),
        friction_limit=np.zeros(station_shape),
        yield_margin=np.zeros(station_shape),
        plastic_slip=np.zeros(station_shape),
        energy_change=np.zeros(station_shape),
        friction_dissipation=np.zeros(station_shape),
        release_dissipation=np.zeros(station_shape),
        constitutive_work=np.zeros(station_shape),
        tangent_generalized=np.zeros(model.nq),
    )


def _validate_inputs(
    model: SpatialModel,
    q: FloatArray,
    qd: FloatArray,
    elastic_displacement_m: FloatArray,
    inputs: StatefulDistributedGripInput,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    if not isinstance(model, SpatialModel):
        raise TypeError("model must be a SpatialModel")
    if not isinstance(inputs, StatefulDistributedGripInput):
        raise TypeError("inputs must be a StatefulDistributedGripInput")
    if not isinstance(inputs.grip_config, DistributedGripConfig):
        raise TypeError("grip_config must be a DistributedGripConfig")
    if not isinstance(inputs.friction_config, StatefulFrictionConfig):
        raise TypeError("friction_config must be a StatefulFrictionConfig")
    if not np.isfinite(inputs.time_step_s) or inputs.time_step_s <= 0.0:
        raise ValueError("time_step_s must be finite and positive")
    position = np.asarray(q, dtype=np.float64)
    velocity = np.asarray(qd, dtype=np.float64)
    if position.shape != (model.nq,) or velocity.shape != (model.nq,):
        raise ValueError("q and qd must match the articulated model dimension")
    expected = (2, inputs.grip_config.station_count_per_hand, 3)
    previous = np.asarray(elastic_displacement_m, dtype=np.float64)
    if previous.shape != expected or not np.all(np.isfinite(previous)):
        raise ValueError(f"elastic_displacement_m must have finite shape {expected}")
    return position, velocity, previous


def _normal_snapshot(
    model: SpatialModel,
    position: FloatArray,
    velocity: FloatArray,
    inputs: StatefulDistributedGripInput,
) -> DistributedGripSnapshot:
    normal = evaluate_distributed_grip(
        model,
        position,
        velocity,
        grip_span_m=inputs.grip_span_m,
        hand_contact_local_x_m=inputs.hand_contact_local_x_m,
        reference_lengths_m=inputs.reference_lengths_m,
        config=replace(inputs.grip_config, friction_coefficient=0.0),
    )
    if normal.normal_force_on_club_n is None or normal.station_signed_gap_m is None:
        raise ValueError("distributed normal snapshot omitted required station fields")
    return normal


def _advance_stations(
    velocity: FloatArray,
    previous: FloatArray,
    kinematics: tuple[FloatArray, FloatArray, FloatArray, FloatArray],
    normal: DistributedGripSnapshot,
    inputs: StatefulDistributedGripInput,
    buf: _StationBuffers,
) -> None:
    hand, hand_jac, grip, grip_jac = kinematics
    stiffness = inputs.friction_config.tangential_stiffness_n_m
    assert normal.normal_force_on_club_n is not None
    for hand_index in range(2):
        for station_index in range(inputs.grip_config.station_count_per_hand):
            key = (hand_index, station_index)
            f_normal = normal.normal_force_on_club_n[key]
            displacement = hand[key] - grip[key]
            distance = float(np.linalg.norm(displacement))
            direction = displacement / distance if distance > 0.0 else np.zeros(3)
            relative_velocity = (hand_jac[key] - grip_jac[key]) @ velocity
            tangential_velocity = relative_velocity - direction * float(
                direction @ relative_velocity
            )
            prior = previous[key]
            active = bool(normal.active_station[key])
            projected = (
                prior - direction * float(direction @ prior) if active else prior
            )
            projection_release = float(
                0.5 * stiffness * ((prior @ prior) - (projected @ projected))
            )
            projected[np.abs(projected) < 1.0e-18] = 0.0
            step = advance_stateful_friction(
                TangentialState(projected),
                tangential_displacement_increment_m=(
                    tangential_velocity * inputs.time_step_s if active else np.zeros(3)
                ),
                normal_load_n=float(np.linalg.norm(f_normal)) if active else 0.0,
                active=active,
                config=inputs.friction_config,
            )
            before = float(0.5 * stiffness * (prior @ prior))
            change = step.elastic_energy_after_j - before
            release = step.release_dissipation_j + projection_release
            if not np.isclose(
                step.constitutive_work_j,
                change + step.frictional_dissipation_j + release,
                rtol=1.0e-10,
                atol=1.0e-14,
            ):
                raise ValueError("stateful station energy ledger did not close")
            force = step.force_on_club_n
            buf.tangential_force[key] = force
            buf.next_state[key] = step.state.elastic_displacement_m
            buf.regimes[key] = step.regime.value
            buf.friction_limit[key] = step.friction_limit_n
            buf.yield_margin[key] = step.yield_margin_n
            buf.plastic_slip[key] = step.plastic_slip_increment_m
            buf.energy_change[key] = change
            buf.friction_dissipation[key] = step.frictional_dissipation_j
            buf.release_dissipation[key] = release
            buf.constitutive_work[key] = step.constitutive_work_j
            buf.tangent_generalized += grip_jac[key].T @ force - hand_jac[key].T @ force
            buf.physical_power += float(force @ grip_jac[key] @ velocity)
            buf.physical_power -= float(force @ hand_jac[key] @ velocity)


def evaluate_stateful_distributed_grip(
    model: SpatialModel,
    q: FloatArray,
    qd: FloatArray,
    elastic_displacement_m: FloatArray,
    *,
    inputs: StatefulDistributedGripInput,
) -> StatefulDistributedGripStep:
    """Advance every station's tangential state at one retained configuration."""

    position, velocity, previous = _validate_inputs(
        model, q, qd, elastic_displacement_m, inputs
    )
    normal = _normal_snapshot(model, position, velocity, inputs)
    kinematics = distributed_contact_kinematics(
        model,
        position,
        grip_span_m=inputs.grip_span_m,
        hand_contact_local_x_m=inputs.hand_contact_local_x_m,
        config=inputs.grip_config,
    )
    buf = _buffers(model, inputs.grip_config)
    _advance_stations(velocity, previous, kinematics, normal, inputs, buf)
    normal_force = cast(FloatArray, normal.normal_force_on_club_n)
    station_gap = cast(FloatArray, normal.station_signed_gap_m)
    generalized = normal.generalized_contact_force + buf.tangent_generalized
    total_force = normal_force + buf.tangential_force
    virtual_residual = (
        abs(float(buf.tangent_generalized @ velocity) - buf.physical_power)
        + normal.virtual_power_residual_w
    )
    return StatefulDistributedGripStep(
        generalized_contact_force=generalized,
        normal_generalized_contact_force=normal.generalized_contact_force,
        tangential_generalized_contact_force=buf.tangent_generalized,
        force_on_club_n=total_force,
        normal_force_on_club_n=normal_force,
        tangential_force_on_club_n=buf.tangential_force,
        active_station=normal.active_station,
        station_signed_gap_m=station_gap,
        elastic_displacement_m=buf.next_state,
        regimes=buf.regimes,
        friction_limit_n=buf.friction_limit,
        yield_margin_n=buf.yield_margin,
        plastic_slip_increment_m=buf.plastic_slip,
        elastic_energy_change_j=buf.energy_change,
        frictional_dissipation_j=buf.friction_dissipation,
        release_dissipation_j=buf.release_dissipation,
        constitutive_work_j=buf.constitutive_work,
        normal_strain_energy_j=normal.strain_energy_j,
        normal_dissipation_power_w=normal.normal_dissipation_power_w,
        normal_power_w=normal.normal_power_w,
        virtual_power_residual_w=virtual_residual,
    )


__all__ = [
    "StatefulDistributedGripInput",
    "StatefulDistributedGripStep",
    "evaluate_stateful_distributed_grip",
]
