"""Replay rigid articulated contact traces for forward attribution (#9153)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from scripts.research.proximal_distal_energy.articulated_contact_projection import (
    ArticulatedContactProjectionConfig,
    evaluate_contact_projection,
)
from scripts.research.proximal_distal_energy.articulated_forward_attribution import (
    ForwardAttribution,
    differentiate_mass_along_velocity,
    integrate_forward_attribution,
)
from scripts.research.proximal_distal_energy.articulated_forward_contract import (
    ArticulatedForwardContactConfig,
)
from scripts.research.proximal_distal_energy.articulated_forward_integration import (
    ForwardIntegrationCase,
    integrate_articulated_contact,
    native_dynamics_operator,
)
from scripts.research.proximal_distal_energy.spatial_full_body import SpatialModel

FloatArray = NDArray[np.float64]
CONTRIBUTION_NAMES = ("configuration", "velocity", "contact", "active")


@dataclass(frozen=True, slots=True)
class RigidTrajectoryAttributionEvidence:
    """Trace replay and its descriptive same-trajectory attribution."""

    trace: dict[str, Any]
    mass_matrices: FloatArray
    mass_matrix_rates: FloatArray
    generalized_forces: FloatArray
    pointwise_force_closure_residual: FloatArray
    attribution: ForwardAttribution


def _validate_contact_contract(
    case: ForwardIntegrationCase,
    contact: ArticulatedContactProjectionConfig,
) -> None:
    if not isinstance(case, ForwardIntegrationCase):
        raise TypeError("case must be a ForwardIntegrationCase")
    if not isinstance(contact, ArticulatedContactProjectionConfig):
        raise TypeError("contact must be an ArticulatedContactProjectionConfig")
    if not np.isclose(contact.contact_stiffness, case.contact_stiffness):
        raise ValueError("contact stiffness must match the forward case")
    if not np.isclose(contact.contact_damping, case.contact_damping):
        raise ValueError("contact damping must match the forward case")


def attribute_rigid_contact_trajectory(
    model: SpatialModel,
    case: ForwardIntegrationCase,
    config: ArticulatedForwardContactConfig,
    contact: ArticulatedContactProjectionConfig,
) -> RigidTrajectoryAttributionEvidence:
    """Replay one qualified trace and integrate named force contributions.

    The returned contributions are evaluated along the same realized
    trajectory. They are descriptive and must not be reported as the outcome
    of a forward ablation whose state history would diverge.
    """

    if not isinstance(model, SpatialModel):
        raise TypeError("model must be a SpatialModel")
    if not isinstance(config, ArticulatedForwardContactConfig):
        raise TypeError("config must be an ArticulatedForwardContactConfig")
    _validate_contact_contract(case, contact)
    trace = integrate_articulated_contact(model, case, config)
    time = np.asarray(trace["time_s"], dtype=np.float64)
    positions = np.asarray(trace["q"], dtype=np.float64)
    velocities = np.asarray(trace["qd"], dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != model.nq:
        raise ValueError("trace q must have shape (samples, model.nq)")
    if velocities.shape != positions.shape:
        raise ValueError("trace qd must have the same shape as trace q")
    if time.ndim != 1 or time.size != positions.shape[0]:
        raise ValueError("trace time_s must contain one value per state sample")
    if not (
        np.all(np.isfinite(time))
        and np.all(np.isfinite(positions))
        and np.all(np.isfinite(velocities))
    ):
        raise ValueError("trace time_s, q, and qd must be finite")
    sample_count = time.size
    masses = np.empty((sample_count, model.nq, model.nq))
    forces = np.empty((sample_count, len(CONTRIBUTION_NAMES), model.nq))
    closure = np.empty(sample_count)
    operator = native_dynamics_operator(case.engine, model)
    for index in range(sample_count):
        position = np.asarray(positions[index], dtype=np.float64)
        velocity = np.asarray(velocities[index], dtype=np.float64)
        matrix, bias = operator(position, velocity)
        zero_velocity_matrix, static_bias = operator(position, np.zeros(model.nq))
        if not np.allclose(matrix, zero_velocity_matrix, rtol=1.0e-10, atol=1.0e-12):
            raise ValueError("native mass matrix changed with generalized velocity")
        snapshot = evaluate_contact_projection(
            model,
            position,
            velocity,
            grip_span_m=case.grip_span_m,
            hand_contact_local_x_m=case.hand_contact_local_x_m,
            perturb_contact=False,
            config=contact,
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
    segments = np.zeros(sample_count, dtype=np.int64)

    def evaluate_mass(position: FloatArray) -> FloatArray:
        return operator(position, np.zeros(model.nq))[0]

    mass_rates = differentiate_mass_along_velocity(
        positions=positions,
        velocities=velocities,
        mass_evaluator=evaluate_mass,
    )
    attribution = integrate_forward_attribution(
        time_s=time,
        mass_matrices=masses,
        mass_matrix_rates=mass_rates,
        velocities=velocities,
        generalized_forces=forces,
        contribution_names=CONTRIBUTION_NAMES,
        segment_ids=segments,
        event_impulses=np.empty((0, model.nq)),
        event_work_j=np.empty(0),
    )
    return RigidTrajectoryAttributionEvidence(
        trace=dict(trace),
        mass_matrices=masses,
        mass_matrix_rates=mass_rates,
        generalized_forces=forces,
        pointwise_force_closure_residual=closure,
        attribution=attribution,
    )


__all__ = [
    "CONTRIBUTION_NAMES",
    "RigidTrajectoryAttributionEvidence",
    "attribute_rigid_contact_trajectory",
]
