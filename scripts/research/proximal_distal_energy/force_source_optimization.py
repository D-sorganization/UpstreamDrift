"""Registered comparison of Coriolis hand-path impulse and speed optima.

This study scores achieved, impact-qualified trajectories.  It does not treat
the coordinate-dependent Christoffel split as a biological force estimate.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from math import isfinite
from typing import Any

import numpy as np

from scripts.research.proximal_distal_energy.run_experiments import rollout_program
from scripts.research.proximal_distal_energy.swing_model import (
    PlanarInertials,
    find_impact,
)
from scripts.research.proximal_distal_energy.torque_programs import (
    restrain_then_drive_program,
)
from src.shared.python.biomechanics.force_source_attribution import (
    attribute_double_pendulum_trajectory,
)
from src.shared.python.simulation_backends import GolfModelParams


@dataclass(frozen=True)
class ForceSourceCandidate:
    """One bounded open-loop torque program in the registered search."""

    shoulder_torque_nm: float
    wrist_drive_nm: float
    wrist_restrain_nm: float
    onset_s: float

    def __post_init__(self) -> None:
        values = asdict(self)
        if not all(isfinite(value) for value in values.values()):
            raise ValueError("candidate values must be finite")
        if self.shoulder_torque_nm <= 0.0:
            raise ValueError("shoulder_torque_nm must be positive")
        if self.wrist_drive_nm < 0.0 or self.wrist_restrain_nm < 0.0:
            raise ValueError("wrist torques must be non-negative magnitudes")
        if self.onset_s < 0.0:
            raise ValueError("onset_s must be non-negative")


@dataclass(frozen=True)
class ForceSourceOutcome:
    """Impact qualification and source metrics for one candidate."""

    candidate: ForceSourceCandidate
    status: str
    impact_time_s: float | None = None
    clubhead_speed_m_s: float | None = None
    coriolis_tangent_impulse_n_s: float | None = None
    coriolis_absolute_tangent_impulse_n_s: float | None = None
    coriolis_work_j: float | None = None
    squared_speed_tangent_impulse_n_s: float | None = None
    squared_speed_work_j: float | None = None
    gravity_tangent_impulse_n_s: float | None = None
    gravity_work_j: float | None = None
    applied_tangent_impulse_n_s: float | None = None
    applied_work_j: float | None = None
    total_tangent_impulse_n_s: float | None = None
    tangent_valid_fraction: float | None = None
    mapping_status: str | None = None
    maximum_mapping_residual_nm: float | None = None


def _through_impact(
    time_s: np.ndarray,
    q: np.ndarray,
    velocity: np.ndarray,
    controls_nm: np.ndarray,
    impact_time_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return histories ending at one shared, linearly interpolated event."""
    before = time_s < impact_time_s
    if not np.any(before):
        raise ValueError("impact must occur after the initial sample")
    if np.isclose(time_s, impact_time_s, atol=1e-12, rtol=0.0).any():
        through = time_s <= impact_time_s + 1e-12
        return time_s[through], q[through], velocity[through], controls_nm[through]

    def interpolate(history: np.ndarray) -> np.ndarray:
        endpoint = np.array(
            [
                np.interp(impact_time_s, time_s, history[:, column])
                for column in range(2)
            ]
        )
        return np.vstack((history[before], endpoint))

    return (
        np.concatenate((time_s[before], [impact_time_s])),
        interpolate(q),
        interpolate(velocity),
        interpolate(controls_nm),
    )


def _mapping_residual_maximum(attribution: Any) -> float:
    return max(
        float(np.max(np.linalg.norm(history.mapping_residual_nm, axis=1)))
        for history in attribution.components.values()
    )


def evaluate_candidate(
    params: GolfModelParams, candidate: ForceSourceCandidate
) -> ForceSourceOutcome:
    """Roll out and score one candidate only through qualified impact."""
    if not isinstance(params, GolfModelParams):
        raise TypeError("params must be a GolfModelParams instance")
    program = restrain_then_drive_program(
        candidate.shoulder_torque_nm,
        candidate.wrist_drive_nm,
        candidate.wrist_restrain_nm,
        candidate.onset_s,
    )
    time_s, q, velocity, controls_nm = rollout_program(params, program)
    impact = find_impact(time_s, q, velocity, PlanarInertials.from_params(params))
    if impact is None:
        return ForceSourceOutcome(candidate=candidate, status="unqualified_impact")
    histories = _through_impact(time_s, q, velocity, controls_nm, impact[0])
    attribution = attribute_double_pendulum_trajectory(params, *histories)
    metrics = attribution.metrics
    coriolis = metrics["coriolis"]
    total_tangent_values = tuple(
        metric.signed_tangent_impulse_n_s for metric in metrics.values()
    )
    total_tangent = (
        None
        if any(value is None for value in total_tangent_values)
        else float(sum(value for value in total_tangent_values if value is not None))
    )
    total_duration = coriolis.tangent_total_duration_s
    valid_fraction = (
        None
        if total_duration <= 0.0
        else coriolis.tangent_valid_duration_s / total_duration
    )
    return ForceSourceOutcome(
        candidate=candidate,
        status="qualified_impact",
        impact_time_s=impact[0],
        clubhead_speed_m_s=impact[1],
        coriolis_tangent_impulse_n_s=coriolis.signed_tangent_impulse_n_s,
        coriolis_absolute_tangent_impulse_n_s=coriolis.absolute_tangent_impulse_n_s,
        coriolis_work_j=coriolis.generalized_work_j,
        squared_speed_tangent_impulse_n_s=metrics[
            "squared_speed"
        ].signed_tangent_impulse_n_s,
        squared_speed_work_j=metrics["squared_speed"].generalized_work_j,
        gravity_tangent_impulse_n_s=metrics["gravity"].signed_tangent_impulse_n_s,
        gravity_work_j=metrics["gravity"].generalized_work_j,
        applied_tangent_impulse_n_s=metrics["applied"].signed_tangent_impulse_n_s,
        applied_work_j=metrics["applied"].generalized_work_j,
        total_tangent_impulse_n_s=total_tangent,
        tangent_valid_fraction=valid_fraction,
        mapping_status="rank_deficient_force_only",
        maximum_mapping_residual_nm=_mapping_residual_maximum(attribution),
    )


def _qualified(
    outcomes: Iterable[ForceSourceOutcome],
) -> tuple[ForceSourceOutcome, ...]:
    return tuple(
        outcome
        for outcome in outcomes
        if outcome.status == "qualified_impact"
        and outcome.coriolis_tangent_impulse_n_s is not None
        and outcome.clubhead_speed_m_s is not None
    )


def _summary_row(outcome: ForceSourceOutcome) -> dict[str, float]:
    row = asdict(outcome.candidate)
    row.update(
        {
            "coriolis_tangent_impulse_n_s": float(outcome.coriolis_tangent_impulse_n_s),
            "coriolis_absolute_tangent_impulse_n_s": float(
                outcome.coriolis_absolute_tangent_impulse_n_s
            ),
            "clubhead_speed_m_s": float(outcome.clubhead_speed_m_s),
        }
    )
    return row


def summarize_optimization(
    outcomes: Iterable[ForceSourceOutcome],
) -> dict[str, Any]:
    """Keep the maximum Coriolis-impulse and speed estimands distinct."""
    materialized = tuple(outcomes)
    qualified = _qualified(materialized)
    if not qualified:
        raise ValueError("at least one qualified outcome is required")
    absolute_impulse_best = max(
        qualified, key=lambda item: item.coriolis_absolute_tangent_impulse_n_s
    )
    signed_impulse_best = max(
        qualified, key=lambda item: item.coriolis_tangent_impulse_n_s
    )
    speed_best = max(qualified, key=lambda item: item.clubhead_speed_m_s)
    return {
        "maximum_coriolis_impulse": _summary_row(absolute_impulse_best),
        "maximum_signed_coriolis_impulse": _summary_row(signed_impulse_best),
        "maximum_clubhead_speed": _summary_row(speed_best),
        "same_candidate": absolute_impulse_best.candidate == speed_best.candidate,
        "candidate_count": len(materialized),
        "qualified_count": len(qualified),
    }


__all__ = [
    "ForceSourceCandidate",
    "ForceSourceOutcome",
    "evaluate_candidate",
    "summarize_optimization",
]
