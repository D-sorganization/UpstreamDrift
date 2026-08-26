"""Exact position-closed planar orbits and scaled constraint-rank margins."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import numpy.typing as npt

from scripts.research.proximal_distal_energy.constraint_internal_force_diagnostics import (
    PlanarClosedLoopAudit,
    audit_scaled_planar_closure_jacobian,
)
from scripts.research.proximal_distal_energy.mechanism_ladder import (
    closed_loop_grip_jacobian,
)

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class PlanarClosedGeometry:
    """Lengths defining a same-origin planar two-arm/grip triangle."""

    lead_arm_length_m: float
    trail_arm_length_m: float
    grip_separation_m: float


@dataclass(frozen=True, slots=True)
class PlanarCoordinateScale:
    """Positive scales mapping normalized coordinates to physical increments."""

    angular_coordinate_scale_rad: float
    translation_coordinate_scale_m: float


@dataclass(frozen=True, slots=True)
class FeasibleClosedLoopConfiguration:
    """One exact planar position-closure configuration."""

    lead_angle_rad: float
    trail_angle_rad: float
    grip_angle_rad: float
    grip_center_xy_m: tuple[float, float]
    triangle_sine_margin: float
    lower_degeneracy_distance_m: float
    upper_degeneracy_distance_m: float


@dataclass(frozen=True, slots=True)
class ClosedLoopOrbitAudit:
    """Scaled rank and closure ranges across both exact assembly branches."""

    sample_count: int
    scale: PlanarCoordinateScale
    relative_tolerance: float
    minimum_rank: int
    maximum_rank: int
    minimum_nullity: int
    maximum_nullity: int
    minimum_smallest_scaled_singular_value_m: float
    maximum_smallest_scaled_singular_value_m: float
    minimum_scaled_condition_number: float
    maximum_scaled_condition_number: float
    reference_scaled_singular_values_m: tuple[float, ...]
    maximum_scaled_singular_value_spread_m: float
    maximum_scaled_nullspace_residual_m: float
    maximum_closure_residual_m: float
    triangle_sine_margin: float
    lower_degeneracy_distance_m: float
    upper_degeneracy_distance_m: float


@dataclass(frozen=True, slots=True)
class TriangleDegeneracyAudit:
    """Exact lower- and upper-span triangle-degeneracy controls."""

    lower_geometry: PlanarClosedGeometry
    upper_geometry: PlanarClosedGeometry
    lower: PlanarClosedLoopAudit
    upper: PlanarClosedLoopAudit
    lower_position_closure_residual_m: float
    upper_position_closure_residual_m: float


def _positive_finite(value: float, *, name: str) -> float:
    scalar = float(value)
    if not math.isfinite(scalar) or scalar <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return scalar


def _validated_geometry(
    geometry: PlanarClosedGeometry,
) -> tuple[float, float, float, float, float]:
    lead = _positive_finite(geometry.lead_arm_length_m, name="lead_arm_length_m")
    trail = _positive_finite(geometry.trail_arm_length_m, name="trail_arm_length_m")
    span = _positive_finite(geometry.grip_separation_m, name="grip_separation_m")
    lower = abs(lead - trail)
    upper = lead + trail
    if span < lower or span > upper:
        raise ValueError("grip separation violates the triangle inequality")
    return lead, trail, span, lower, upper


def _arm_endpoint(angle_rad: float, length_m: float) -> FloatArray:
    return np.array(
        (length_m * math.sin(angle_rad), -length_m * math.cos(angle_rad)),
        dtype=float,
    )


def feasible_closed_loop_configuration(
    geometry: PlanarClosedGeometry,
    *,
    phase_rad: float,
    branch: int,
) -> FeasibleClosedLoopConfiguration:
    """Construct one exact same-origin triangle on a declared assembly branch."""

    lead, trail, span, lower, upper = _validated_geometry(geometry)
    phase = float(phase_rad)
    if not math.isfinite(phase):
        raise ValueError("phase_rad must be finite")
    if branch not in (-1, 1):
        raise ValueError("branch must be -1 or 1")
    cosine = (lead**2 + trail**2 - span**2) / (2.0 * lead * trail)
    relative_angle = branch * math.acos(float(np.clip(cosine, -1.0, 1.0)))
    lead_angle = phase
    trail_angle = phase + relative_angle
    lead_point = _arm_endpoint(lead_angle, lead)
    trail_point = _arm_endpoint(trail_angle, trail)
    grip_vector = trail_point - lead_point
    grip_angle = math.atan2(float(grip_vector[1]), float(grip_vector[0]))
    center = 0.5 * (lead_point + trail_point)
    return FeasibleClosedLoopConfiguration(
        lead_angle_rad=lead_angle,
        trail_angle_rad=trail_angle,
        grip_angle_rad=grip_angle,
        grip_center_xy_m=(float(center[0]), float(center[1])),
        triangle_sine_margin=abs(math.sin(relative_angle)),
        lower_degeneracy_distance_m=span - lower,
        upper_degeneracy_distance_m=upper - span,
    )


def planar_closure_residual(
    configuration: FeasibleClosedLoopConfiguration,
    geometry: PlanarClosedGeometry,
) -> FloatArray:
    """Return lead and trail contact-position residuals in length units."""

    lead, trail, span, _, _ = _validated_geometry(geometry)
    center = np.asarray(configuration.grip_center_xy_m, dtype=float)
    grip_axis = np.array(
        (math.cos(configuration.grip_angle_rad), math.sin(configuration.grip_angle_rad))
    )
    lead_target = center - 0.5 * span * grip_axis
    trail_target = center + 0.5 * span * grip_axis
    return np.concatenate(
        (
            _arm_endpoint(configuration.lead_angle_rad, lead) - lead_target,
            _arm_endpoint(configuration.trail_angle_rad, trail) - trail_target,
        )
    )


def audit_feasible_configuration(
    configuration: FeasibleClosedLoopConfiguration,
    geometry: PlanarClosedGeometry,
    scale: PlanarCoordinateScale,
    *,
    relative_tolerance: float,
) -> PlanarClosedLoopAudit:
    """Audit one exact-closure configuration under declared scales."""

    lead, trail, span, _, _ = _validated_geometry(geometry)
    jacobian = closed_loop_grip_jacobian(
        lead_angle_rad=configuration.lead_angle_rad,
        trail_angle_rad=configuration.trail_angle_rad,
        grip_angle_rad=configuration.grip_angle_rad,
        lead_arm_length_m=lead,
        trail_arm_length_m=trail,
        grip_separation_m=span,
    )
    return audit_scaled_planar_closure_jacobian(
        jacobian,
        (
            scale.angular_coordinate_scale_rad,
            scale.translation_coordinate_scale_m,
        ),
        relative_tolerance=relative_tolerance,
    )


def _orbit_samples(
    geometry: PlanarClosedGeometry,
    *,
    phase_sample_count: int,
) -> list[FeasibleClosedLoopConfiguration]:
    if phase_sample_count < 3:
        raise ValueError("phase_sample_count must be at least three")
    phases = np.linspace(-math.pi, math.pi, phase_sample_count, endpoint=False)
    return [
        feasible_closed_loop_configuration(
            geometry,
            phase_rad=float(phase),
            branch=branch,
        )
        for branch in (-1, 1)
        for phase in phases
    ]


def audit_closed_loop_orbit(
    geometry: PlanarClosedGeometry,
    scale: PlanarCoordinateScale,
    *,
    phase_sample_count: int,
    relative_tolerance: float,
) -> ClosedLoopOrbitAudit:
    """Audit both exact assembly branches over a full phase orbit."""

    configurations = _orbit_samples(geometry, phase_sample_count=phase_sample_count)
    audits = [
        audit_feasible_configuration(
            configuration,
            geometry,
            scale,
            relative_tolerance=relative_tolerance,
        )
        for configuration in configurations
    ]
    closure_residuals = [
        float(np.max(np.abs(planar_closure_residual(configuration, geometry))))
        for configuration in configurations
    ]
    first = configurations[0]
    smallest = [audit.smallest_scaled_singular_value_m for audit in audits]
    conditions = [audit.scaled_condition_number for audit in audits]
    spectra = np.asarray([audit.scaled_singular_values_m for audit in audits])
    return ClosedLoopOrbitAudit(
        sample_count=len(audits),
        scale=scale,
        relative_tolerance=float(relative_tolerance),
        minimum_rank=min(audit.rank for audit in audits),
        maximum_rank=max(audit.rank for audit in audits),
        minimum_nullity=min(audit.nullity for audit in audits),
        maximum_nullity=max(audit.nullity for audit in audits),
        minimum_smallest_scaled_singular_value_m=min(smallest),
        maximum_smallest_scaled_singular_value_m=max(smallest),
        minimum_scaled_condition_number=min(conditions),
        maximum_scaled_condition_number=max(conditions),
        reference_scaled_singular_values_m=tuple(float(value) for value in spectra[0]),
        maximum_scaled_singular_value_spread_m=float(np.max(np.ptp(spectra, axis=0))),
        maximum_scaled_nullspace_residual_m=max(
            audit.maximum_scaled_nullspace_residual_m for audit in audits
        ),
        maximum_closure_residual_m=max(closure_residuals),
        triangle_sine_margin=first.triangle_sine_margin,
        lower_degeneracy_distance_m=first.lower_degeneracy_distance_m,
        upper_degeneracy_distance_m=first.upper_degeneracy_distance_m,
    )


def _boundary_audit(
    geometry: PlanarClosedGeometry,
    scale: PlanarCoordinateScale,
    *,
    relative_tolerance: float,
) -> tuple[PlanarClosedLoopAudit, float]:
    configuration = feasible_closed_loop_configuration(
        geometry,
        phase_rad=0.0,
        branch=1,
    )
    residual = float(np.max(np.abs(planar_closure_residual(configuration, geometry))))
    audit = audit_feasible_configuration(
        configuration,
        geometry,
        scale,
        relative_tolerance=relative_tolerance,
    )
    return audit, residual


def audit_triangle_degeneracies(
    geometry: PlanarClosedGeometry,
    scale: PlanarCoordinateScale,
    *,
    relative_tolerance: float,
) -> TriangleDegeneracyAudit:
    """Audit both exact collinear triangle boundaries of the declared map."""

    lead, trail, _, lower, upper = _validated_geometry(geometry)
    if lower == 0.0:
        raise ValueError("equal arm lengths make the lower boundary coincident")
    lower_geometry = PlanarClosedGeometry(lead, trail, lower)
    upper_geometry = PlanarClosedGeometry(lead, trail, upper)
    lower_audit, lower_residual = _boundary_audit(
        lower_geometry,
        scale,
        relative_tolerance=relative_tolerance,
    )
    upper_audit, upper_residual = _boundary_audit(
        upper_geometry,
        scale,
        relative_tolerance=relative_tolerance,
    )
    return TriangleDegeneracyAudit(
        lower_geometry=lower_geometry,
        upper_geometry=upper_geometry,
        lower=lower_audit,
        upper=upper_audit,
        lower_position_closure_residual_m=lower_residual,
        upper_position_closure_residual_m=upper_residual,
    )
