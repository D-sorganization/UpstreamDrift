"""Contracts shared by bounded articulated forward-contact studies."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from scripts.research.proximal_distal_energy.spatial_full_body import (
    SpatialModel,
    forward_kinematics,
    mass_matrix,
)

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class ArticulatedForwardContactConfig:
    """Preoutcome horizon, cohort, perturbation, and acceptance gates."""

    duration_s: float = 0.005
    time_steps_s: tuple[float, ...] = (0.001, 0.0005, 0.00025)
    case_indices: tuple[int, ...] = (0, 4, 8, 9, 13, 17)
    sample_indices: tuple[int, ...] = (0, 6, 12)
    contact_stiffness: float = 1800.0
    contact_damping: float = 18.0
    initial_club_displacement_m: float = 1.0e-3
    initial_club_velocity_m_s: float = 5.0e-2
    retention_threshold_m: float = 1.0e-2
    virtual_power_tolerance_w: float = 1.0e-10
    positive_dissipation_tolerance_w: float = 1.0e-12
    trajectory_relative_tolerance: float = 1.0e-7
    normalized_energy_residual_tolerance: float = 2.0e-2
    refinement_ratio_limit: float = 0.8

    def __post_init__(self) -> None:
        positive = (
            "duration_s",
            "contact_stiffness",
            "initial_club_displacement_m",
            "initial_club_velocity_m_s",
            "retention_threshold_m",
            "virtual_power_tolerance_w",
            "positive_dissipation_tolerance_w",
            "trajectory_relative_tolerance",
            "normalized_energy_residual_tolerance",
            "refinement_ratio_limit",
        )
        for name in positive:
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        steps = np.asarray(self.time_steps_s, dtype=float)
        valid_steps = (
            steps.ndim == 1
            and steps.size >= 2
            and np.all(np.isfinite(steps))
            and np.all(steps > 0.0)
            and np.all(np.diff(steps) < 0.0)
            and np.allclose(self.duration_s / steps, np.rint(self.duration_s / steps))
        )
        if not valid_steps:
            raise ValueError(
                "time_steps_s must be finite, positive, strictly decreasing, and divide duration_s"
            )
        if not np.isfinite(self.contact_damping) or self.contact_damping < 0.0:
            raise ValueError("contact_damping must be finite and nonnegative")
        if not 0.0 < self.refinement_ratio_limit < 1.0:
            raise ValueError("refinement_ratio_limit must lie in (0, 1)")
        self._validate_indices("case_indices", self.case_indices, 18)
        self._validate_indices("sample_indices", self.sample_indices, 13)

    @staticmethod
    def _validate_indices(name: str, values: tuple[int, ...], upper: int) -> None:
        valid = (
            bool(values)
            and len(set(values)) == len(values)
            and all(isinstance(value, int) and 0 <= value < upper for value in values)
        )
        if not valid:
            raise ValueError(f"{name} must contain unique in-range integers")


@dataclass(frozen=True, slots=True)
class ForwardVariant:
    """One factor at a time from the nominal attachment perturbation."""

    name: str
    stiffness_factor: float = 1.0
    damping_factor: float = 1.0
    displacement_factor: float = 1.0
    velocity_factor: float = 1.0


def registered_variants() -> tuple[ForwardVariant, ...]:
    """Return nominal, null, reversal, and one-factor adverse branches."""

    return (
        ForwardVariant("nominal"),
        ForwardVariant("stiffness_low", stiffness_factor=0.5),
        ForwardVariant("stiffness_high", stiffness_factor=2.0),
        ForwardVariant("damping_low", damping_factor=0.5),
        ForwardVariant("damping_high", damping_factor=2.0),
        ForwardVariant("velocity_reversed", velocity_factor=-1.0),
        ForwardVariant("zero_preload", displacement_factor=0.0, velocity_factor=0.0),
    )


def mechanical_energy(model: SpatialModel, q: FloatArray, qd: FloatArray) -> float:
    """Return articulated kinetic plus gravitational potential energy."""

    position = np.asarray(q, dtype=float)
    velocity = np.asarray(qd, dtype=float)
    if position.shape != (model.nq,) or velocity.shape != (model.nq,):
        raise ValueError("q and qd must match the articulated model dimension")
    kinetic = 0.5 * float(velocity @ mass_matrix(model, position) @ velocity)
    kinematics = forward_kinematics(model, position)
    potential = sum(
        body.mass_kg * 9.80665 * kinematics.body_position_m[index, 2]
        for index, body in enumerate(model.bodies)
    )
    return kinetic + float(potential)


__all__ = [
    "ArticulatedForwardContactConfig",
    "ForwardVariant",
    "mechanical_energy",
    "registered_variants",
]
