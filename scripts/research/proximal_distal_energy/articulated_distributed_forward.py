"""Bounded forward integration for distributed articulated grip fibers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from scripts.research.proximal_distal_energy.articulated_distributed_grip import (
    DistributedGripConfig,
    DistributedGripSnapshot,
    distributed_reference_lengths,
    evaluate_distributed_grip,
)
from scripts.research.proximal_distal_energy.articulated_forward_contract import (
    mechanical_energy,
)
from scripts.research.proximal_distal_energy.articulated_forward_integration import (
    native_dynamics_operator,
)
from scripts.research.proximal_distal_energy.spatial_full_body import SpatialModel

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class DistributedForwardConfig:
    """Horizon and numerical tolerances for distributed-grip forwarding."""

    duration_s: float = 0.05
    time_steps_s: tuple[float, ...] = (0.001, 0.0005)
    virtual_power_tolerance_w: float = 1.0e-10
    normalized_energy_residual_tolerance: float = 5.0e-2
    trajectory_relative_tolerance: float = 1.0e-7
    refinement_ratio_limit: float = 0.9

    def __post_init__(self) -> None:
        for name in (
            "duration_s",
            "virtual_power_tolerance_w",
            "normalized_energy_residual_tolerance",
            "trajectory_relative_tolerance",
            "refinement_ratio_limit",
        ):
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
                "time_steps_s must be decreasing positive divisors of duration_s"
            )
        if not 0.0 < self.refinement_ratio_limit < 1.0:
            raise ValueError("refinement_ratio_limit must lie in (0, 1)")


@dataclass(frozen=True, slots=True)
class DistributedIntegrationCase:
    """All varying inputs for one distributed-grip trajectory."""

    q: FloatArray
    qd: FloatArray
    grip_span_m: float
    hand_contact_local_x_m: float
    time_step_s: float
    initial_club_displacement_m: float
    initial_club_velocity_m_s: float
    engine: str
    grip: DistributedGripConfig


@dataclass(slots=True)
class _Buffers:
    q: FloatArray
    qd: FloatArray
    force: FloatArray
    extension: FloatArray
    active_count: NDArray[np.int_]
    couple: FloatArray
    concentration: FloatArray
    coincident_couple: FloatArray
    reversal_residual: FloatArray
    virtual_power: FloatArray
    dissipation: FloatArray
    strain: FloatArray
    mechanical: FloatArray


def _validate_case(
    model: SpatialModel,
    case: DistributedIntegrationCase,
    config: DistributedForwardConfig,
) -> int:
    if not isinstance(case, DistributedIntegrationCase):
        raise TypeError("case must be a DistributedIntegrationCase")
    if not isinstance(config, DistributedForwardConfig):
        raise TypeError("config must be a DistributedForwardConfig")
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
        raise ValueError("contact geometry and time_step_s must be positive")
    steps = int(round(config.duration_s / case.time_step_s))
    if not np.isclose(steps * case.time_step_s, config.duration_s):
        raise ValueError("time_step_s must divide the configured duration")
    return steps


def _buffers(sample_count: int, nq: int) -> _Buffers:
    states = np.empty((sample_count, nq))
    return _Buffers(
        q=states,
        qd=np.empty_like(states),
        force=np.empty(sample_count),
        extension=np.empty(sample_count),
        active_count=np.empty(sample_count, dtype=int),
        couple=np.empty((sample_count, 3)),
        concentration=np.empty(sample_count),
        coincident_couple=np.empty(sample_count),
        reversal_residual=np.empty(sample_count),
        virtual_power=np.empty(sample_count),
        dissipation=np.empty(sample_count),
        strain=np.empty(sample_count),
        mechanical=np.empty(sample_count),
    )


def _record(
    buffers: _Buffers,
    index: int,
    position: FloatArray,
    velocity: FloatArray,
    snapshot: DistributedGripSnapshot,
    model: SpatialModel,
) -> None:
    buffers.q[index], buffers.qd[index] = position, velocity
    buffers.force[index] = snapshot.maximum_station_force_n
    buffers.extension[index] = snapshot.maximum_extension_m
    buffers.active_count[index] = snapshot.active_station_count
    buffers.couple[index] = snapshot.force_couple_vector_nm
    buffers.concentration[index] = snapshot.load_concentration
    buffers.coincident_couple[index] = snapshot.coincident_couple_residual_nm
    buffers.reversal_residual[index] = snapshot.reversed_couple_sign_residual_nm
    buffers.virtual_power[index] = snapshot.virtual_power_residual_w
    buffers.dissipation[index] = snapshot.dissipation_power_w
    buffers.strain[index] = snapshot.strain_energy_j
    buffers.mechanical[index] = mechanical_energy(model, position, velocity)


def _result(
    buffers: _Buffers, case: DistributedIntegrationCase
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
        "maximum_station_force_n": buffers.force,
        "maximum_extension_m": buffers.extension,
        "active_station_count": buffers.active_count,
        "active_set_transition": transitions,
        "force_couple_vector_nm": buffers.couple,
        "station_load_concentration": buffers.concentration,
        "coincident_couple_residual_nm": buffers.coincident_couple,
        "reversed_couple_sign_residual_nm": buffers.reversal_residual,
        "virtual_power_residual_w": buffers.virtual_power,
        "dissipation_power_w": buffers.dissipation,
        "strain_energy_j": buffers.strain,
        "mechanical_energy_j": buffers.mechanical,
        "total_energy_j": total,
        "cumulative_dissipation_j": cumulative,
        "work_energy_residual_j": total - total[0] - cumulative,
    }


def integrate_distributed_grip(
    model: SpatialModel,
    case: DistributedIntegrationCase,
    config: DistributedForwardConfig = DistributedForwardConfig(),
) -> dict[str, NDArray[Any]]:
    """Advance one state-registered passive distributed grip."""

    step_count = _validate_case(model, case, config)
    reference = distributed_reference_lengths(
        model,
        case.q,
        grip_span_m=case.grip_span_m,
        hand_contact_local_x_m=case.hand_contact_local_x_m,
        config=case.grip,
    )
    position, velocity = np.asarray(case.q).copy(), np.asarray(case.qd).copy()
    position[14] += case.initial_club_displacement_m
    velocity[14] += case.initial_club_velocity_m_s
    operator = native_dynamics_operator(case.engine, model)
    buffers = _buffers(step_count + 1, model.nq)
    for index in range(step_count + 1):
        snapshot = evaluate_distributed_grip(
            model,
            position,
            velocity,
            grip_span_m=case.grip_span_m,
            hand_contact_local_x_m=case.hand_contact_local_x_m,
            reference_lengths_m=reference,
            config=case.grip,
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
    "DistributedForwardConfig",
    "DistributedIntegrationCase",
    "integrate_distributed_grip",
]
