"""Event-location adapter for existing distributed-grip trajectories."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from scripts.research.proximal_distal_energy.articulated_contact_events import (
    ContactEventRecord,
    EventAlignedStateTrace,
    align_state_trace_to_events,
    locate_contact_events,
)
from scripts.research.proximal_distal_energy.articulated_forward_attribution import (
    ForwardAttribution,
    differentiate_mass_along_velocity,
    integrate_forward_attribution,
)
from scripts.research.proximal_distal_energy.articulated_distributed_forward import (
    DistributedForwardConfig,
    DistributedIntegrationCase,
    integrate_distributed_grip,
)
from scripts.research.proximal_distal_energy.articulated_distributed_grip import (
    distributed_reference_lengths,
    distributed_signed_gaps,
    evaluate_distributed_grip,
)
from scripts.research.proximal_distal_energy.articulated_forward_integration import (
    native_dynamics_operator,
)
from scripts.research.proximal_distal_energy.spatial_full_body import SpatialModel

CONTRIBUTION_NAMES = ("configuration", "velocity", "contact", "active")


@dataclass(frozen=True, slots=True)
class DistributedTrajectoryAttributionEvidence:
    """Event-aligned distributed trace and descriptive attribution."""

    trace: dict[str, Any]
    events: tuple[ContactEventRecord, ...]
    aligned: EventAlignedStateTrace
    mass_matrices: np.ndarray
    mass_matrix_rates: np.ndarray
    generalized_forces: np.ndarray
    pointwise_force_closure_residual: np.ndarray
    attribution: ForwardAttribution


def locate_distributed_trace_events(
    *,
    model: SpatialModel,
    case: DistributedIntegrationCase,
    trace: Mapping[str, Any],
    gap_tolerance_m: float = 1.0e-10,
    time_tolerance_s: float = 1.0e-12,
) -> tuple[ContactEventRecord, ...]:
    """Locate opening/reattachment roots on a retained distributed trace."""

    if not isinstance(model, SpatialModel):
        raise TypeError("model must be a SpatialModel")
    if not isinstance(case, DistributedIntegrationCase):
        raise TypeError("case must be a DistributedIntegrationCase")
    required = (
        "time_s",
        "q",
        "qd",
        "station_signed_gap_m",
        "station_active",
    )
    missing = tuple(name for name in required if name not in trace)
    if missing:
        raise ValueError(f"trace is missing event fields: {', '.join(missing)}")
    reference = distributed_reference_lengths(
        model,
        case.q,
        grip_span_m=case.grip_span_m,
        hand_contact_local_x_m=case.hand_contact_local_x_m,
        config=case.grip,
    )

    def evaluate_gap(position: np.ndarray) -> np.ndarray:
        return distributed_signed_gaps(
            model,
            position,
            grip_span_m=case.grip_span_m,
            hand_contact_local_x_m=case.hand_contact_local_x_m,
            reference_lengths_m=reference,
            config=case.grip,
        )

    return locate_contact_events(
        time_s=np.asarray(trace["time_s"], dtype=np.float64),
        positions=np.asarray(trace["q"], dtype=np.float64),
        velocities=np.asarray(trace["qd"], dtype=np.float64),
        station_signed_gap_m=np.asarray(
            trace["station_signed_gap_m"], dtype=np.float64
        ),
        station_active=np.asarray(trace["station_active"], dtype=bool),
        gap_evaluator=evaluate_gap,
        gap_tolerance_m=gap_tolerance_m,
        time_tolerance_s=time_tolerance_s,
    )


def attribute_distributed_contact_trajectory(
    *,
    model: SpatialModel,
    case: DistributedIntegrationCase,
    config: DistributedForwardConfig,
) -> DistributedTrajectoryAttributionEvidence:
    """Integrate distributed force contributions on an event-aligned replay.

    Opening and reattachment under the compliant tension law have zero
    discrete impulse and work. They are still explicit quadrature boundaries.
    The attribution follows one realized trace and is not a causal ablation.
    """

    if not isinstance(model, SpatialModel):
        raise TypeError("model must be a SpatialModel")
    if not isinstance(case, DistributedIntegrationCase):
        raise TypeError("case must be a DistributedIntegrationCase")
    if not isinstance(config, DistributedForwardConfig):
        raise TypeError("config must be a DistributedForwardConfig")
    raw_trace = integrate_distributed_grip(model, case, config)
    trace = dict(raw_trace)
    events = locate_distributed_trace_events(model=model, case=case, trace=trace)
    aligned = align_state_trace_to_events(
        time_s=np.asarray(trace["time_s"], dtype=np.float64),
        positions=np.asarray(trace["q"], dtype=np.float64),
        velocities=np.asarray(trace["qd"], dtype=np.float64),
        events=events,
    )
    reference = distributed_reference_lengths(
        model,
        case.q,
        grip_span_m=case.grip_span_m,
        hand_contact_local_x_m=case.hand_contact_local_x_m,
        config=case.grip,
    )
    operator = native_dynamics_operator(case.engine, model)
    sample_count = aligned.time_s.size
    masses = np.empty((sample_count, model.nq, model.nq))
    forces = np.empty((sample_count, len(CONTRIBUTION_NAMES), model.nq))
    closure = np.empty(sample_count)
    for index, (position, velocity) in enumerate(
        zip(aligned.positions, aligned.velocities, strict=True)
    ):
        matrix, bias = operator(position, velocity)
        zero_velocity_matrix, static_bias = operator(position, np.zeros(model.nq))
        if not np.allclose(matrix, zero_velocity_matrix, rtol=1.0e-10, atol=1.0e-12):
            raise ValueError("native mass matrix changed with generalized velocity")
        snapshot = evaluate_distributed_grip(
            model,
            position,
            velocity,
            grip_span_m=case.grip_span_m,
            hand_contact_local_x_m=case.hand_contact_local_x_m,
            reference_lengths_m=reference,
            config=case.grip,
        )
        contribution_forces = np.stack(
            (
                -static_bias,
                -(bias - static_bias),
                snapshot.generalized_contact_force,
                np.zeros(model.nq),
            )
        )
        masses[index] = matrix
        forces[index] = contribution_forces
        closure[index] = np.linalg.norm(
            np.sum(contribution_forces, axis=0)
            - (snapshot.generalized_contact_force - bias)
        )

    def evaluate_mass(position: np.ndarray) -> np.ndarray:
        return operator(position, np.zeros(model.nq))[0]

    mass_rates = differentiate_mass_along_velocity(
        positions=aligned.positions,
        velocities=aligned.velocities,
        mass_evaluator=evaluate_mass,
    )
    event_count = aligned.event_record_offsets.size
    attribution = integrate_forward_attribution(
        time_s=aligned.time_s,
        mass_matrices=masses,
        mass_matrix_rates=mass_rates,
        velocities=aligned.velocities,
        generalized_forces=forces,
        contribution_names=CONTRIBUTION_NAMES,
        segment_ids=aligned.segment_ids,
        event_impulses=np.zeros((event_count, model.nq)),
        event_work_j=np.zeros(event_count),
    )
    return DistributedTrajectoryAttributionEvidence(
        trace=trace,
        events=events,
        aligned=aligned,
        mass_matrices=masses,
        mass_matrix_rates=mass_rates,
        generalized_forces=forces,
        pointwise_force_closure_residual=closure,
        attribution=attribution,
    )


__all__ = [
    "CONTRIBUTION_NAMES",
    "DistributedTrajectoryAttributionEvidence",
    "attribute_distributed_contact_trajectory",
    "locate_distributed_trace_events",
]
