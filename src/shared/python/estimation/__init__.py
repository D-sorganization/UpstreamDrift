"""Canonical-core estimation primitives and synthetic-data utilities."""

from __future__ import annotations

from src.shared.python.estimation.identifiability import (
    IdentifiabilityReport,
    ParameterSpec,
    finite_difference_jacobian as identifiability_finite_difference_jacobian,
    probe_identifiability,
    sweep_parameter,
)
from src.shared.python.estimation.residuals import (
    ResidualFunction,
    RneaFunction,
    anthropometric_prior_residual,
    autodiff_jacobian,
    dynamics_residual,
    finite_difference_jacobian,
    project_pinhole,
    reprojection_residual,
    reprojection_residual_from_points,
    residual_jacobian,
    smoothness_residual,
)
from src.shared.python.estimation.synthetic_ground_truth import (
    ForwardModel,
    GroundTruthRigResult,
    NoiseModel,
    ObservationPolicy,
    ProjectionRecord,
    SkeletonRigForwardModel,
    SyntheticCamera,
    SyntheticObservationRig,
    project_world_point,
)

__all__ = [
    "ForwardModel",
    "GroundTruthRigResult",
    "IdentifiabilityReport",
    "NoiseModel",
    "ObservationPolicy",
    "ParameterSpec",
    "ProjectionRecord",
    "ResidualFunction",
    "RneaFunction",
    "SkeletonRigForwardModel",
    "SyntheticCamera",
    "SyntheticObservationRig",
    "anthropometric_prior_residual",
    "autodiff_jacobian",
    "dynamics_residual",
    "finite_difference_jacobian",
    "identifiability_finite_difference_jacobian",
    "probe_identifiability",
    "project_pinhole",
    "project_world_point",
    "reprojection_residual",
    "reprojection_residual_from_points",
    "residual_jacobian",
    "smoothness_residual",
    "sweep_parameter",
]
