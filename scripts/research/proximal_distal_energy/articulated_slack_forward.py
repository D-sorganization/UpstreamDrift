"""Bounded forward integration for typed articulated slack attachments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from scripts.research.proximal_distal_energy.articulated_forward_contract import (
    mechanical_energy,
)
from scripts.research.proximal_distal_energy.articulated_forward_integration import (
    native_dynamics_operator,
)
from scripts.research.proximal_distal_energy.articulated_slack_contact import (
    AttachmentLawConfig,
    SlackProjectionSnapshot,
    evaluate_slack_projection,
)
from scripts.research.proximal_distal_energy.spatial_full_body import SpatialModel

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class ArticulatedSlackForwardConfig:
    """Horizon, refinement schedule, and numerical tolerances."""

    duration_s: float = 0.005
    time_steps_s: tuple[float, ...] = (0.001, 0.0005, 0.00025)
    virtual_power_tolerance_w: float = 1.0e-10
    normalized_energy_residual_tolerance: float = 3.0e-2
    trajectory_relative_tolerance: float = 1.0e-7
    refinement_ratio_limit: float = 0.8

    def __post_init__(self) -> None:
        positive = (
            "duration_s",
            "virtual_power_tolerance_w",
            "normalized_energy_residual_tolerance",
            "trajectory_relative_tolerance",
            "refinement_ratio_limit",
        )
        for name in positive:
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
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
                "time_steps_s must be finite, positive, strictly decreasing, and divide duration_s"
            )
        if not 0.0 < self.refinement_ratio_limit < 1.0:
            raise ValueError("refinement_ratio_limit must lie in (0, 1)")


@dataclass(frozen=True, slots=True)
class SlackIntegrationCase:
    """All varying inputs for one typed-slack trajectory."""

    q: FloatArray
    qd: FloatArray
    grip_span_m: float
    hand_contact_local_x_m: float
    time_step_s: float
    initial_club_displacement_m: float
    initial_club_velocity_m_s: float
    engine: str
    law: AttachmentLawConfig


@dataclass(slots=True)
class _SlackTraceBuffers:
    q: FloatArray
    qd: FloatArray
    force: FloatArray
    separation: FloatArray
    active_count: NDArray[np.int_]
    virtual_power: FloatArray
    storage_power: FloatArray
    dissipation: FloatArray
    strain: FloatArray
    mechanical: FloatArray


def _validate_case(
    model: SpatialModel,
    case: SlackIntegrationCase,
    config: ArticulatedSlackForwardConfig,
) -> int:
    if not isinstance(case, SlackIntegrationCase):
        raise TypeError("case must be a SlackIntegrationCase")
    if not isinstance(config, ArticulatedSlackForwardConfig):
        raise TypeError("config must be an ArticulatedSlackForwardConfig")
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
    if case.grip_span_m <= 0.0 or case.hand_contact_local_x_m <= 0.0:
        raise ValueError("contact geometry must be positive")
    if case.time_step_s <= 0.0:
        raise ValueError("time_step_s must be positive")
    step_count = int(round(config.duration_s / case.time_step_s))
    if not np.isclose(step_count * case.time_step_s, config.duration_s):
        raise ValueError("time_step_s must divide the configured duration")
    return step_count


def _buffers(sample_count: int, nq: int) -> _SlackTraceBuffers:
    states = np.empty((sample_count, nq))
    return _SlackTraceBuffers(
        q=states,
        qd=np.empty_like(states),
        force=np.empty(sample_count),
        separation=np.empty(sample_count),
        active_count=np.empty(sample_count, dtype=int),
        virtual_power=np.empty(sample_count),
        storage_power=np.empty(sample_count),
        dissipation=np.empty(sample_count),
        strain=np.empty(sample_count),
        mechanical=np.empty(sample_count),
    )


def _record(
    buffers: _SlackTraceBuffers,
    index: int,
    position: FloatArray,
    velocity: FloatArray,
    snapshot: SlackProjectionSnapshot,
    model: SpatialModel,
) -> None:
    buffers.q[index], buffers.qd[index] = position, velocity
    buffers.force[index] = snapshot.maximum_contact_force_n
    buffers.separation[index] = snapshot.maximum_attachment_separation_m
    buffers.active_count[index] = snapshot.active_interface_count
    buffers.virtual_power[index] = snapshot.virtual_power_residual_w
    buffers.storage_power[index] = snapshot.storage_power_w
    buffers.dissipation[index] = snapshot.dissipation_power_w
    buffers.strain[index] = snapshot.strain_energy_j
    buffers.mechanical[index] = mechanical_energy(model, position, velocity)


def _result(
    buffers: _SlackTraceBuffers, case: SlackIntegrationCase
) -> dict[str, NDArray[Any]]:
    cumulative = np.zeros(buffers.dissipation.size)
    cumulative[1:] = np.cumsum(
        0.5 * (buffers.dissipation[1:] + buffers.dissipation[:-1]) * case.time_step_s
    )
    total = buffers.mechanical + buffers.strain
    active = buffers.active_count > 0
    transitions = np.zeros(active.size, dtype=bool)
    transitions[1:] = active[1:] != active[:-1]
    return {
        "time_s": np.arange(active.size) * case.time_step_s,
        "q": buffers.q,
        "qd": buffers.qd,
        "maximum_contact_force_n": buffers.force,
        "maximum_attachment_separation_m": buffers.separation,
        "active_interface_count": buffers.active_count,
        "active_set_transition": transitions,
        "virtual_power_residual_w": buffers.virtual_power,
        "storage_power_w": buffers.storage_power,
        "dissipation_power_w": buffers.dissipation,
        "strain_energy_j": buffers.strain,
        "mechanical_energy_j": buffers.mechanical,
        "total_energy_j": total,
        "cumulative_dissipation_j": cumulative,
        "work_energy_residual_j": total - total[0] - cumulative,
    }


def integrate_articulated_slack(
    model: SpatialModel,
    case: SlackIntegrationCase,
    config: ArticulatedSlackForwardConfig = ArticulatedSlackForwardConfig(),
) -> dict[str, NDArray[Any]]:
    """Advance one typed passive attachment law through a bounded horizon."""

    step_count = _validate_case(model, case, config)
    position, velocity = np.asarray(case.q).copy(), np.asarray(case.qd).copy()
    position[14] += case.initial_club_displacement_m
    velocity[14] += case.initial_club_velocity_m_s
    operator = native_dynamics_operator(case.engine, model)
    buffers = _buffers(step_count + 1, model.nq)
    for index in range(step_count + 1):
        snapshot = evaluate_slack_projection(
            model,
            position,
            velocity,
            grip_span_m=case.grip_span_m,
            hand_contact_local_x_m=case.hand_contact_local_x_m,
            law=case.law,
        )
        _record(buffers, index, position, velocity, snapshot, model)
        if index < step_count:
            matrix, bias = operator(position, velocity)
            acceleration = np.linalg.solve(
                matrix, snapshot.generalized_contact_force - bias
            )
            velocity = velocity + case.time_step_s * acceleration
            position = position + case.time_step_s * velocity
    return _result(buffers, case)


__all__ = [
    "ArticulatedSlackForwardConfig",
    "SlackIntegrationCase",
    "integrate_articulated_slack",
]
