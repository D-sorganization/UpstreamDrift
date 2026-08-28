"""Stateful tangential-compliance adapter for the distributed grip model."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray

from scripts.research.proximal_distal_energy.articulated_distributed_grip import (
    DistributedGripConfig,
    distributed_contact_kinematics,
    evaluate_distributed_grip,
)
from scripts.research.proximal_distal_energy.articulated_stateful_friction import (
    StatefulFrictionConfig,
    TangentialRegime,
    TangentialState,
    advance_stateful_friction,
)
from scripts.research.proximal_distal_energy.spatial_full_body import SpatialModel

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class StatefulDistributedGripStep:
    """Combined normal/tangential loads and complete retained state ledger."""

    generalized_contact_force: FloatArray
    force_on_club_n: FloatArray
    normal_force_on_club_n: FloatArray
    tangential_force_on_club_n: FloatArray
    elastic_displacement_m: FloatArray
    regimes: NDArray[np.str_]
    friction_limit_n: FloatArray
    yield_margin_n: FloatArray
    plastic_slip_increment_m: FloatArray
    elastic_energy_change_j: FloatArray
    frictional_dissipation_j: FloatArray
    release_dissipation_j: FloatArray
    constitutive_work_j: FloatArray
    virtual_power_residual_w: float
    static_stick_modeled: bool = True
    human_or_anatomical_inference: bool = False


def _validate_state(
    elastic_displacement_m: FloatArray, shape: tuple[int, int, int]
) -> FloatArray:
    state = np.asarray(elastic_displacement_m, dtype=np.float64)
    if state.shape != shape or not np.all(np.isfinite(state)):
        raise ValueError(f"elastic_displacement_m must have finite shape {shape}")
    return state


def evaluate_stateful_distributed_grip(
    model: SpatialModel,
    q: FloatArray,
    qd: FloatArray,
    *,
    grip_span_m: float,
    hand_contact_local_x_m: float,
    reference_lengths_m: FloatArray,
    grip_config: DistributedGripConfig,
    friction_config: StatefulFrictionConfig,
    elastic_displacement_m: FloatArray,
    time_step_s: float,
) -> StatefulDistributedGripStep:
    """Advance every station's tangential state at one retained configuration.

    Retained elastic displacement is projected into the current tangent plane.
    The lost projection energy is explicit release dissipation, preventing a
    changing normal direction from silently creating or deleting stored energy.
    """

    if not isinstance(model, SpatialModel):
        raise TypeError("model must be a SpatialModel")
    if not isinstance(grip_config, DistributedGripConfig):
        raise TypeError("grip_config must be a DistributedGripConfig")
    if not isinstance(friction_config, StatefulFrictionConfig):
        raise TypeError("friction_config must be a StatefulFrictionConfig")
    if not np.isfinite(time_step_s) or time_step_s <= 0.0:
        raise ValueError("time_step_s must be finite and positive")
    position = np.asarray(q, dtype=np.float64)
    velocity = np.asarray(qd, dtype=np.float64)
    if position.shape != (model.nq,) or velocity.shape != (model.nq,):
        raise ValueError("q and qd must match the articulated model dimension")
    station_shape = (2, grip_config.station_count_per_hand)
    point_shape = (*station_shape, 3)
    previous = _validate_state(elastic_displacement_m, point_shape)
    normal_config = replace(grip_config, friction_coefficient=0.0)
    normal = evaluate_distributed_grip(
        model,
        position,
        velocity,
        grip_span_m=grip_span_m,
        hand_contact_local_x_m=hand_contact_local_x_m,
        reference_lengths_m=reference_lengths_m,
        config=normal_config,
    )
    if normal.normal_force_on_club_n is None:
        raise ValueError("distributed normal snapshot omitted station forces")
    hand, hand_jac, grip, grip_jac = distributed_contact_kinematics(
        model,
        position,
        grip_span_m=grip_span_m,
        hand_contact_local_x_m=hand_contact_local_x_m,
        config=grip_config,
    )
    tangential_force = np.zeros(point_shape)
    next_state = np.zeros(point_shape)
    regimes = np.empty(station_shape, dtype="U16")
    friction_limit = np.zeros(station_shape)
    yield_margin = np.zeros(station_shape)
    plastic_slip = np.zeros(station_shape)
    energy_change = np.zeros(station_shape)
    friction_dissipation = np.zeros(station_shape)
    release_dissipation = np.zeros(station_shape)
    constitutive_work = np.zeros(station_shape)
    tangent_generalized = np.zeros(model.nq)
    physical_power = 0.0
    stiffness = friction_config.tangential_stiffness_n_m
    for hand_index in range(2):
        for station_index in range(grip_config.station_count_per_hand):
            f_normal = normal.normal_force_on_club_n[hand_index, station_index]
            normal_load = float(np.linalg.norm(f_normal))
            active = bool(normal.active_station[hand_index, station_index])
            displacement = (
                hand[hand_index, station_index] - grip[hand_index, station_index]
            )
            distance = float(np.linalg.norm(displacement))
            normal_direction = (
                displacement / distance if distance > 0.0 else np.zeros(3)
            )
            v_hand = hand_jac[hand_index, station_index] @ velocity
            v_grip = grip_jac[hand_index, station_index] @ velocity
            relative_velocity = v_hand - v_grip
            tangential_velocity = relative_velocity - normal_direction * float(
                normal_direction @ relative_velocity
            )
            prior = previous[hand_index, station_index]
            projection_release = 0.0
            if active:
                projected = prior - normal_direction * float(normal_direction @ prior)
                projection_release = float(
                    0.5 * stiffness * ((prior @ prior) - (projected @ projected))
                )
                projected[np.abs(projected) < 1.0e-18] = 0.0
                state = TangentialState(projected)
                increment = tangential_velocity * time_step_s
                retained_load = normal_load
            else:
                state = TangentialState(prior)
                increment = np.zeros(3)
                retained_load = 0.0
            step = advance_stateful_friction(
                state,
                tangential_displacement_increment_m=increment,
                normal_load_n=retained_load,
                active=active,
                config=friction_config,
            )
            actual_before = float(0.5 * stiffness * (prior @ prior))
            actual_change = step.elastic_energy_after_j - actual_before
            release = step.release_dissipation_j + projection_release
            if not np.isclose(
                step.constitutive_work_j,
                actual_change + step.frictional_dissipation_j + release,
                rtol=1.0e-10,
                atol=1.0e-14,
            ):
                raise ValueError("stateful station energy ledger did not close")
            force = step.force_on_club_n
            tangential_force[hand_index, station_index] = force
            next_state[hand_index, station_index] = step.state.elastic_displacement_m
            regimes[hand_index, station_index] = step.regime.value
            friction_limit[hand_index, station_index] = step.friction_limit_n
            yield_margin[hand_index, station_index] = step.yield_margin_n
            plastic_slip[hand_index, station_index] = step.plastic_slip_increment_m
            energy_change[hand_index, station_index] = actual_change
            friction_dissipation[hand_index, station_index] = (
                step.frictional_dissipation_j
            )
            release_dissipation[hand_index, station_index] = release
            constitutive_work[hand_index, station_index] = step.constitutive_work_j
            tangent_generalized += (
                grip_jac[hand_index, station_index].T @ force
                - hand_jac[hand_index, station_index].T @ force
            )
            physical_power += float(force @ v_grip - force @ v_hand)
    generalized = normal.generalized_contact_force + tangent_generalized
    total_force = normal.normal_force_on_club_n + tangential_force
    virtual_residual = (
        abs(float(tangent_generalized @ velocity) - physical_power)
        + normal.virtual_power_residual_w
    )
    return StatefulDistributedGripStep(
        generalized_contact_force=generalized,
        force_on_club_n=total_force,
        normal_force_on_club_n=normal.normal_force_on_club_n,
        tangential_force_on_club_n=tangential_force,
        elastic_displacement_m=next_state,
        regimes=regimes,
        friction_limit_n=friction_limit,
        yield_margin_n=yield_margin,
        plastic_slip_increment_m=plastic_slip,
        elastic_energy_change_j=energy_change,
        frictional_dissipation_j=friction_dissipation,
        release_dissipation_j=release_dissipation,
        constitutive_work_j=constitutive_work,
        virtual_power_residual_w=virtual_residual,
    )


__all__ = ["StatefulDistributedGripStep", "evaluate_stateful_distributed_grip"]
