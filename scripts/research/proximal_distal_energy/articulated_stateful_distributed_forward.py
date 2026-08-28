"""Timestamp-explicit forward integration for the stateful grip countermodel.

The implementation is an engineering comparator. It does not represent finger
anatomy, neural control, intent, or human evidence. Node states and interval
constitutive responses are deliberately separate so reviewers can identify the
operator split and quantify its numerical work defect under refinement.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from numpy.typing import NDArray

from scripts.research.proximal_distal_energy.articulated_distributed_grip import (
    DistributedGripConfig,
    distributed_reference_lengths,
    evaluate_distributed_grip,
)
from scripts.research.proximal_distal_energy.articulated_forward_contract import (
    mechanical_energy,
)
from scripts.research.proximal_distal_energy.articulated_forward_integration import (
    native_dynamics_operator,
)
from scripts.research.proximal_distal_energy.articulated_stateful_distributed_grip import (
    StatefulDistributedGripInput,
    evaluate_stateful_distributed_grip,
)
from scripts.research.proximal_distal_energy.articulated_stateful_friction import (
    StatefulFrictionConfig,
)
from scripts.research.proximal_distal_energy.spatial_full_body import SpatialModel

FloatArray = NDArray[np.float64]

STATEFUL_OPERATOR_SPLIT = (
    "left_node_kinematics__end_increment_return_map__semi_implicit_mechanics"
)


@dataclass(frozen=True, slots=True)
class StatefulDistributedForwardConfig:
    """Bounded horizon and decreasing refinement steps for this integrator."""

    duration_s: float = 0.005
    time_steps_s: tuple[float, ...] = (0.001, 0.0005, 0.00025)
    virtual_power_tolerance_w: float = 1.0e-10

    def __post_init__(self) -> None:
        if not np.isfinite(self.duration_s) or self.duration_s <= 0.0:
            raise ValueError("duration_s must be finite and positive")
        if (
            not np.isfinite(self.virtual_power_tolerance_w)
            or self.virtual_power_tolerance_w <= 0.0
        ):
            raise ValueError("virtual_power_tolerance_w must be finite and positive")
        steps = np.asarray(self.time_steps_s, dtype=float)
        valid = (
            steps.ndim == 1
            and steps.size >= 2
            and np.all(np.isfinite(steps))
            and np.all(steps > 0.0)
            and np.all(np.diff(steps) < 0.0)
            and np.allclose(self.duration_s / steps, np.rint(self.duration_s / steps))
        )
        if not valid:
            raise ValueError(
                "time_steps_s must be decreasing positive divisors of duration_s"
            )


@dataclass(frozen=True, slots=True)
class StatefulDistributedIntegrationCase:
    """Initial conditions and constitutive choices for one forward trace."""

    q: FloatArray
    qd: FloatArray
    grip_span_m: float
    hand_contact_local_x_m: float
    time_step_s: float
    initial_club_displacement_m: float
    initial_club_velocity_m_s: float
    engine: str
    grip: DistributedGripConfig
    friction: StatefulFrictionConfig
    initial_elastic_displacement_m: FloatArray
    initial_state_velocity_factor: float = 1.0


def _validate_case(
    model: SpatialModel,
    case: StatefulDistributedIntegrationCase,
    config: StatefulDistributedForwardConfig,
) -> tuple[int, FloatArray]:
    if not isinstance(model, SpatialModel):
        raise TypeError("model must be a SpatialModel")
    if not isinstance(case, StatefulDistributedIntegrationCase):
        raise TypeError("case must be a StatefulDistributedIntegrationCase")
    if not isinstance(config, StatefulDistributedForwardConfig):
        raise TypeError("config must be a StatefulDistributedForwardConfig")
    if not isinstance(case.grip, DistributedGripConfig):
        raise TypeError("grip must be a DistributedGripConfig")
    if not isinstance(case.friction, StatefulFrictionConfig):
        raise TypeError("friction must be a StatefulFrictionConfig")
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
        case.initial_state_velocity_factor,
    )
    if any(not np.isfinite(value) for value in scalars):
        raise ValueError("integration scalars must be finite")
    if min(case.grip_span_m, case.hand_contact_local_x_m, case.time_step_s) <= 0.0:
        raise ValueError("contact geometry and time_step_s must be positive")
    step_count = int(round(config.duration_s / case.time_step_s))
    if not np.isclose(step_count * case.time_step_s, config.duration_s):
        raise ValueError("time_step_s must divide the configured duration")
    expected = (2, case.grip.station_count_per_hand, 3)
    state = np.asarray(case.initial_elastic_displacement_m, dtype=np.float64)
    if state.shape != expected or not np.all(np.isfinite(state)):
        raise ValueError(
            f"initial_elastic_displacement_m must have finite shape {expected}"
        )
    return step_count, state.copy()


def _normal_strain_energy(
    model: SpatialModel,
    position: FloatArray,
    velocity: FloatArray,
    case: StatefulDistributedIntegrationCase,
    references: FloatArray,
) -> float:
    snapshot = evaluate_distributed_grip(
        model,
        position,
        velocity,
        grip_span_m=case.grip_span_m,
        hand_contact_local_x_m=case.hand_contact_local_x_m,
        reference_lengths_m=references,
        config=replace(case.grip, friction_coefficient=0.0),
    )
    return snapshot.strain_energy_j


def _tangential_strain_energy(state: FloatArray, stiffness: float) -> float:
    return float(0.5 * stiffness * np.sum(state * state))


def integrate_stateful_distributed_grip(
    model: SpatialModel,
    case: StatefulDistributedIntegrationCase,
    config: StatefulDistributedForwardConfig = StatefulDistributedForwardConfig(),
) -> dict[str, Any]:
    """Advance one trace under an explicitly declared first-order split.

    At interval ``n``, kinematics are sampled at node ``n``. The tangential
    return map advances its retained state across that interval. Its
    end-of-increment force and the left-node normal force then drive a
    semi-implicit Euler mechanical update. Consequently the constitutive ledger
    is exact, while the coupling-work and total-energy residuals are numerical
    diagnostics that must be assessed under time-step refinement.
    """

    step_count, elastic_state = _validate_case(model, case, config)
    references = distributed_reference_lengths(
        model,
        case.q,
        grip_span_m=case.grip_span_m,
        hand_contact_local_x_m=case.hand_contact_local_x_m,
        config=case.grip,
    )
    position = np.asarray(case.q, dtype=np.float64).copy()
    velocity = (
        np.asarray(case.qd, dtype=np.float64).copy()
        * case.initial_state_velocity_factor
    )
    position[14] += case.initial_club_displacement_m
    velocity[14] += case.initial_club_velocity_m_s
    operator = native_dynamics_operator(case.engine, model)
    station_shape = (2, case.grip.station_count_per_hand)
    state_shape = (*station_shape, 3)

    node_q = np.empty((step_count + 1, model.nq))
    node_qd = np.empty_like(node_q)
    node_state = np.empty((step_count + 1, *state_shape))
    node_mechanical = np.empty(step_count + 1)
    node_normal_strain = np.empty(step_count + 1)
    node_tangent_strain = np.empty(step_count + 1)
    interval_generalized = np.empty((step_count, model.nq))
    interval_normal_generalized = np.empty_like(interval_generalized)
    interval_tangent_generalized = np.empty_like(interval_generalized)
    interval_regime = np.empty((step_count, *station_shape), dtype="U16")
    interval_active = np.empty((step_count, *station_shape), dtype=bool)
    interval_gap = np.empty((step_count, *station_shape))
    interval_before = np.empty((step_count, *state_shape))
    interval_after = np.empty_like(interval_before)
    interval_energy_change = np.empty((step_count, *station_shape))
    interval_friction_diss = np.empty_like(interval_energy_change)
    interval_release_diss = np.empty_like(interval_energy_change)
    interval_work = np.empty_like(interval_energy_change)
    interval_normal_diss = np.empty(step_count)
    interval_virtual_residual = np.empty(step_count)
    interval_coupling_residual = np.empty(step_count)

    def record_node(index: int) -> None:
        node_q[index] = position
        node_qd[index] = velocity
        node_state[index] = elastic_state
        node_mechanical[index] = mechanical_energy(model, position, velocity)
        node_normal_strain[index] = _normal_strain_energy(
            model, position, velocity, case, references
        )
        node_tangent_strain[index] = _tangential_strain_energy(
            elastic_state, case.friction.tangential_stiffness_n_m
        )

    record_node(0)
    for index in range(step_count):
        before = elastic_state.copy()
        response = evaluate_stateful_distributed_grip(
            model,
            position,
            velocity,
            elastic_state,
            inputs=StatefulDistributedGripInput(
                grip_span_m=case.grip_span_m,
                hand_contact_local_x_m=case.hand_contact_local_x_m,
                reference_lengths_m=references,
                grip_config=case.grip,
                friction_config=case.friction,
                time_step_s=case.time_step_s,
            ),
        )
        if response.virtual_power_residual_w > config.virtual_power_tolerance_w:
            raise ValueError("stateful contact virtual-power gate failed")
        interval_generalized[index] = response.generalized_contact_force
        interval_normal_generalized[index] = response.normal_generalized_contact_force
        interval_tangent_generalized[index] = (
            response.tangential_generalized_contact_force
        )
        interval_regime[index] = response.regimes
        interval_active[index] = response.active_station
        interval_gap[index] = response.station_signed_gap_m
        interval_before[index] = before
        interval_after[index] = response.elastic_displacement_m
        interval_energy_change[index] = response.elastic_energy_change_j
        interval_friction_diss[index] = response.frictional_dissipation_j
        interval_release_diss[index] = response.release_dissipation_j
        interval_work[index] = response.constitutive_work_j
        interval_normal_diss[index] = max(
            0.0, -response.normal_dissipation_power_w * case.time_step_s
        )
        interval_virtual_residual[index] = response.virtual_power_residual_w
        interval_coupling_residual[index] = float(
            response.tangential_generalized_contact_force @ velocity * case.time_step_s
            + np.sum(response.constitutive_work_j)
        )

        matrix, bias = operator(position, velocity)
        acceleration = np.linalg.solve(
            matrix, response.generalized_contact_force - bias
        )
        velocity = velocity + case.time_step_s * acceleration
        position = position + case.time_step_s * velocity
        elastic_state = response.elastic_displacement_m.copy()
        record_node(index + 1)

    total_stored = node_normal_strain + node_tangent_strain
    total_energy = node_mechanical + total_stored
    interval_total_diss = (
        interval_normal_diss
        + np.sum(interval_friction_diss, axis=(1, 2))
        + np.sum(interval_release_diss, axis=(1, 2))
    )
    cumulative_diss = np.concatenate(([0.0], np.cumsum(interval_total_diss)))
    return {
        "operator_split": STATEFUL_OPERATOR_SPLIT,
        "force_timestamp": "interval_end_state_at_left_node_kinematics",
        "mechanical_step": "semi_implicit_euler",
        "node_time_s": np.arange(step_count + 1) * case.time_step_s,
        "interval_time_start_s": np.arange(step_count) * case.time_step_s,
        "node_q": node_q,
        "node_qd": node_qd,
        "node_elastic_displacement_m": node_state,
        "node_mechanical_energy_j": node_mechanical,
        "node_normal_strain_energy_j": node_normal_strain,
        "node_tangential_strain_energy_j": node_tangent_strain,
        "node_total_stored_energy_j": total_stored,
        "node_total_energy_j": total_energy,
        "interval_generalized_contact_force": interval_generalized,
        "interval_normal_generalized_contact_force": interval_normal_generalized,
        "interval_tangential_generalized_contact_force": interval_tangent_generalized,
        "interval_regime": interval_regime,
        "interval_active_station": interval_active,
        "interval_station_signed_gap_m": interval_gap,
        "interval_elastic_displacement_before_m": interval_before,
        "interval_elastic_displacement_after_m": interval_after,
        "interval_tangential_elastic_energy_change_j": interval_energy_change,
        "interval_frictional_dissipation_j": interval_friction_diss,
        "interval_release_dissipation_j": interval_release_diss,
        "interval_constitutive_work_j": interval_work,
        "interval_normal_dissipation_j": interval_normal_diss,
        "interval_total_dissipation_j": interval_total_diss,
        "interval_virtual_power_residual_w": interval_virtual_residual,
        "interval_tangential_coupling_work_residual_j": interval_coupling_residual,
        "cumulative_dissipation_j": cumulative_diss,
        "passive_energy_balance_residual_j": (
            total_energy - total_energy[0] + cumulative_diss
        ),
        "static_stick_modeled": True,
        "human_or_anatomical_inference": False,
    }


__all__ = [
    "STATEFUL_OPERATOR_SPLIT",
    "StatefulDistributedForwardConfig",
    "StatefulDistributedIntegrationCase",
    "integrate_stateful_distributed_grip",
]
