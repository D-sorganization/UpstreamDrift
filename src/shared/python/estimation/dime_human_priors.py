"""Hierarchical Human Dimensions and Coupled Range-of-Motion Priors (DIME-13, #11421, #11434).

Provides:
1. Versioned population priors and uncertainty models (ANSUR-II, de Leva 1996).
2. Hierarchical body dimension priors with full covariance structure.
3. Bayesian posterior update combining population priors with sparse/partial subject measurements.
4. Physical dimension bounds and fail-closed height consistency verification.
5. Coupled range-of-motion constraints (e.g. scapulothoracic/glenohumeral shoulder coupling).
6. Physical realizability verification for segment inertia tensors (triangle inequalities).
7. Structured diagnostics reporting prior influence, Mahalanobis distances, and conflict evidence.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from enum import Enum
import logging
import math
from typing import Any, Final

import numpy as np
import numpy.typing as npt

from src.shared.python.contracts import PreconditionError, postcondition, precondition

logger = logging.getLogger(__name__)


class PopulationPriorVersion(str, Enum):
    """Supported population prior tables and empirical data models."""

    ANSUR2_V1 = "ansur2_v1"
    DE_LEVA_1996_V1 = "de_leva_1996_v1"


@dataclass(frozen=True)
class SubjectDimensionMeasurement:
    """A direct measurement of a human subject dimension with declared uncertainty."""

    dimension_name: str
    measured_value: float
    uncertainty: float  # 1-sigma standard deviation in meters

    def __post_init__(self) -> None:
        if not self.dimension_name or not self.dimension_name.strip():
            raise PreconditionError("dimension_name must be a non-empty string.")
        if not math.isfinite(self.measured_value) or self.measured_value <= 0.0:
            raise PreconditionError(
                f"measured_value for {self.dimension_name} must be positive and finite, got {self.measured_value}."
            )
        if not math.isfinite(self.uncertainty) or self.uncertainty <= 0.0:
            raise PreconditionError(
                f"uncertainty for {self.dimension_name} must be positive and finite, got {self.uncertainty}."
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "dimension_name": self.dimension_name,
            "measured_value": self.measured_value,
            "uncertainty": self.uncertainty,
        }


@dataclass(frozen=True)
class PhysicalDimensionBounds:
    """Hard physical/biological bounds for human dimensions."""

    min_height_m: float = 0.50
    max_height_m: float = 2.50
    height_consistency_tolerance: float = (
        0.15  # Max 15% discrepancy between segment sum and height
    )

    def validate_height_consistency(
        self, height_m: float, segments: Mapping[str, float]
    ) -> None:
        """Verify that longitudinal body segments are consistent with total height."""
        if (
            not math.isfinite(height_m)
            or height_m < self.min_height_m
            or height_m > self.max_height_m
        ):
            raise PreconditionError(
                f"Total height {height_m}m is outside valid physical bounds [{self.min_height_m}, {self.max_height_m}]m."
            )

        longitudinal_keys = (
            "head",
            "torso",
            "thigh_left",
            "thigh_right",
            "shank_left",
            "shank_right",
        )
        found_keys = [k for k in longitudinal_keys if k in segments]
        if "torso" in segments and (
            "thigh_left" in segments or "thigh_right" in segments
        ):
            thigh = segments.get("thigh_left", segments.get("thigh_right", 0.0))
            shank = segments.get("shank_left", segments.get("shank_right", 0.0))
            head = segments.get("head", 0.25)
            torso = segments["torso"]
            segment_sum = head + torso + thigh + shank
            discrepancy = abs(segment_sum - height_m) / height_m
            if discrepancy > self.height_consistency_tolerance:
                raise PreconditionError(
                    f"Contradictory measured height: longitudinal segment sum {segment_sum:.2f}m "
                    f"contradicts total measured height {height_m:.2f}m (relative error {discrepancy:.1%} > {self.height_consistency_tolerance:.1%})."
                )


@dataclass(frozen=True)
class RangeOfMotionBound:
    """Uncoupled joint range-of-motion bound in radians."""

    joint_name: str
    lower_limit: float  # radians
    upper_limit: float  # radians
    unit: str = "rad"

    def __post_init__(self) -> None:
        validate_range_of_motion_units(
            self.joint_name, self.lower_limit, self.upper_limit, self.unit
        )


def validate_range_of_motion_units(
    joint_name: str, lower_limit: float, upper_limit: float, unit: str
) -> None:
    """Validate that range-of-motion limits are expressed in radians and are physically valid."""
    if unit.lower() != "rad":
        raise PreconditionError(
            f"Joint {joint_name} limits must be specified in radians, got unit '{unit}'."
        )
    if not math.isfinite(lower_limit) or not math.isfinite(upper_limit):
        raise PreconditionError(f"Joint {joint_name} limits must be finite numbers.")
    if lower_limit >= upper_limit:
        raise PreconditionError(
            f"Joint {joint_name} lower limit {lower_limit} must be < upper limit {upper_limit}."
        )
    # Guard against passing degree values (e.g. 90 deg instead of 1.57 rad)
    limit_cutoff = 2.0 * math.pi + 0.1
    if abs(lower_limit) > limit_cutoff or abs(upper_limit) > limit_cutoff:
        raise PreconditionError(
            f"Joint {joint_name} limits [{lower_limit}, {upper_limit}] exceed maximum radian threshold {limit_cutoff:.2f}. "
            "Pass limits in radians, not degrees."
        )


@dataclass(frozen=True)
class CoupledRomEvaluationResult:
    """Result of evaluating a pose against uncoupled and coupled ROM constraints."""

    is_feasible: bool
    violation_amount: float
    violated_couplings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_feasible": self.is_feasible,
            "violation_amount": self.violation_amount,
            "violated_couplings": list(self.violated_couplings),
        }


@dataclass(frozen=True)
class CoupledRangeOfMotionPrior:
    """Enforces coupled physiological limits across multi-joint complexes."""

    max_shoulder_elevation_rad: float = 3.10  # ~177 deg
    min_shoulder_elevation_rad: float = 0.0

    def evaluate_plausibility(
        self, pose: Mapping[str, float]
    ) -> CoupledRomEvaluationResult:
        """Evaluate coupled constraints, notably scapulohumeral rhythm."""
        violations: list[str] = []
        total_violation = 0.0

        elev = pose.get("shoulder_elevation")
        rot = pose.get("shoulder_internal_rotation")

        if elev is not None and rot is not None:
            # Scapulohumeral coupling: high elevation severely restricts internal/external rotation
            # At elevation = 0, rot_max = 1.5 rad (~86 deg)
            # At elevation = 2.5 rad (~143 deg), rot_max = 0.35 rad (~20 deg)
            max_rot_allowed = max(0.30, 1.50 - 0.46 * max(0.0, elev))
            if abs(rot) > max_rot_allowed:
                excess = abs(rot) - max_rot_allowed
                total_violation += excess
                violations.append("shoulder_coupled")

        is_feasible = total_violation <= 1e-6
        return CoupledRomEvaluationResult(
            is_feasible=is_feasible,
            violation_amount=float(total_violation),
            violated_couplings=tuple(violations),
        )


@dataclass(frozen=True)
class InertiaRealizabilityCheck:
    """Diagnostic report on segment inertia physical realizability."""

    is_realizable: bool
    eigenvalues: tuple[float, float, float]
    failure_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_realizable": self.is_realizable,
            "eigenvalues": list(self.eigenvalues),
            "failure_reason": self.failure_reason,
        }


def validate_inertia_realizability(
    mass_kg: float, ixx: float, iyy: float, izz: float
) -> InertiaRealizabilityCheck:
    """Verify positive mass and classical triangle inequalities for principal moments of inertia."""
    if not math.isfinite(mass_kg) or mass_kg <= 0.0:
        return InertiaRealizabilityCheck(
            is_realizable=False,
            eigenvalues=(ixx, iyy, izz),
            failure_reason="non_positive_mass",
        )

    for val, name in ((ixx, "ixx"), (iyy, "iyy"), (izz, "izz")):
        if not math.isfinite(val) or val <= 0.0:
            return InertiaRealizabilityCheck(
                is_realizable=False,
                eigenvalues=(ixx, iyy, izz),
                failure_reason=f"non_positive_inertia_{name}",
            )

    # Triangle inequalities: the sum of any two principal moments must be >= the third
    eps = 1e-9
    if (ixx + iyy < izz - eps) or (ixx + izz < iyy - eps) or (iyy + izz < ixx - eps):
        return InertiaRealizabilityCheck(
            is_realizable=False,
            eigenvalues=(ixx, iyy, izz),
            failure_reason="triangle_inequality_violation",
        )

    return InertiaRealizabilityCheck(
        is_realizable=True,
        eigenvalues=(ixx, iyy, izz),
        failure_reason=None,
    )


@dataclass(frozen=True)
class HierarchicalDimensionPrior:
    """Correlated Gaussian prior over hierarchical human body dimensions."""

    names: tuple[str, ...]
    mean: npt.NDArray[np.float64]
    covariance: npt.NDArray[np.float64]
    version: PopulationPriorVersion = PopulationPriorVersion.ANSUR2_V1

    def __post_init__(self) -> None:
        dim = len(self.names)
        raw_mean = np.asarray(self.mean, dtype=np.float64)
        raw_cov = np.asarray(self.covariance, dtype=np.float64)

        if raw_mean.shape != (dim,):
            raise PreconditionError(
                f"Mean vector shape {raw_mean.shape} must match dimension count {dim}."
            )
        if raw_cov.shape != (dim, dim):
            raise PreconditionError(
                f"Covariance matrix shape {raw_cov.shape} must be ({dim}, {dim})."
            )

        # Symmetry check
        if not np.allclose(raw_cov, raw_cov.T, atol=1e-8):
            raise PreconditionError("Covariance matrix must be symmetric.")

        # Positive semi-definiteness check
        min_eig = float(np.min(np.linalg.eigvalsh(raw_cov)))
        if min_eig < -1e-8:
            raise PreconditionError(
                f"Covariance matrix must be positive semi-definite; min eigenvalue is {min_eig}."
            )

    def get_mean(self, name: str) -> float:
        """Retrieve prior mean for a named dimension."""
        if name not in self.names:
            raise KeyError(f"Dimension '{name}' not found in prior.")
        idx = self.names.index(name)
        return float(self.mean[idx])

    def get_variance(self, name: str) -> float:
        """Retrieve prior variance for a named dimension."""
        if name not in self.names:
            raise KeyError(f"Dimension '{name}' not found in prior.")
        idx = self.names.index(name)
        return float(self.covariance[idx, idx])

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": (
                self.version.value
                if isinstance(self.version, PopulationPriorVersion)
                else str(self.version)
            ),
            "names": list(self.names),
            "mean": self.mean.tolist(),
            "covariance": self.covariance.tolist(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> HierarchicalDimensionPrior:
        names = tuple(str(x) for x in data["names"])
        mean = np.asarray(data["mean"], dtype=np.float64)
        cov = np.asarray(data["covariance"], dtype=np.float64)
        ver_str = data.get("version", PopulationPriorVersion.ANSUR2_V1.value)
        version = (
            PopulationPriorVersion(ver_str)
            if ver_str in PopulationPriorVersion._value2member_map_
            else ver_str
        )
        return cls(names=names, mean=mean, covariance=cov, version=version)

    @classmethod
    def create_default(
        cls, allow_asymmetry: bool = False
    ) -> HierarchicalDimensionPrior:
        """Construct standard human anthropometric prior from de Leva (1996) and ANSUR-II."""
        names: tuple[str, ...]
        if allow_asymmetry:
            names = (
                "height",
                "wingspan",
                "torso",
                "thigh_left",
                "thigh_right",
                "shank_left",
                "shank_right",
            )
            means = [1.75, 1.78, 0.60, 0.42, 0.42, 0.40, 0.40]
            stds = np.array(
                [0.08, 0.09, 0.04, 0.03, 0.03, 0.03, 0.03], dtype=np.float64
            )
            # Latent factors: general body scale + lower extremity length
            w1 = np.array(
                [0.075, 0.082, 0.032, 0.026, 0.026, 0.025, 0.025], dtype=np.float64
            )
            w2 = np.array(
                [0.0, 0.0, -0.010, 0.010, 0.010, 0.010, 0.010], dtype=np.float64
            )
            spec_var = np.maximum(1e-6, stds**2 - w1**2 - w2**2)
            cov = np.outer(w1, w1) + np.outer(w2, w2) + np.diag(spec_var)
        else:
            names = ("height", "wingspan", "torso", "thigh", "shank")
            means = [1.75, 1.78, 0.60, 0.42, 0.40]
            stds = np.array([0.08, 0.09, 0.04, 0.03, 0.03], dtype=np.float64)
            w1 = np.array([0.075, 0.082, 0.032, 0.026, 0.025], dtype=np.float64)
            spec_var = np.maximum(1e-6, stds**2 - w1**2)
            cov = np.outer(w1, w1) + np.diag(spec_var)

        # Symmetrize
        cov = 0.5 * (cov + cov.T)
        return cls(
            names=names,
            mean=np.asarray(means, dtype=np.float64),
            covariance=cov,
            version=PopulationPriorVersion.ANSUR2_V1,
        )


def update_hierarchical_dimension_posterior(
    prior: HierarchicalDimensionPrior,
    measurements: Sequence[SubjectDimensionMeasurement],
) -> HierarchicalDimensionPrior:
    """Perform Bayesian Gaussian posterior update given partial/sparse subject measurements."""
    if not measurements:
        return prior

    dim = len(prior.names)
    mu_prior = prior.mean.copy()
    cov_prior = prior.covariance.copy()

    # Filter measurements that exist in prior
    valid_meas = [m for m in measurements if m.dimension_name in prior.names]
    if not valid_meas:
        return prior

    m_count = len(valid_meas)
    H = np.zeros((m_count, dim), dtype=np.float64)
    y = np.zeros(m_count, dtype=np.float64)
    R_diag = np.zeros(m_count, dtype=np.float64)

    for i, m in enumerate(valid_meas):
        idx = prior.names.index(m.dimension_name)
        H[i, idx] = 1.0
        y[i] = m.measured_value
        R_diag[i] = m.uncertainty**2

    R = np.diag(R_diag)
    # Innovation covariance S = H P H^T + R
    S = H @ cov_prior @ H.T + R

    # Kalman gain K = P H^T S^-1
    K = cov_prior @ H.T @ np.linalg.inv(S)

    # Updated mean: mu_post = mu_prior + K (y - H mu_prior)
    y_pred = H @ mu_prior
    mu_post = mu_prior + K @ (y - y_pred)

    # Joseph form covariance update for numerical stability: (I - KH) P (I - KH)^T + K R K^T
    I_KH = np.eye(dim) - K @ H
    cov_post = I_KH @ cov_prior @ I_KH.T + K @ R @ K.T
    # Symmetrize
    cov_post = 0.5 * (cov_post + cov_post.T)

    return HierarchicalDimensionPrior(
        names=prior.names,
        mean=mu_post,
        covariance=cov_post,
        version=prior.version,
    )


@dataclass(frozen=True)
class CompatibilityReport:
    """Report on compatibility of individual dimensions with correlated population prior."""

    is_acceptable: bool
    mahalanobis_distance: float
    dimensions_evaluated: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_acceptable": self.is_acceptable,
            "mahalanobis_distance": self.mahalanobis_distance,
            "dimensions_evaluated": list(self.dimensions_evaluated),
        }


def evaluate_human_prior_compatibility(
    prior: HierarchicalDimensionPrior,
    max_mahalanobis: float = 4.0,
    **dimensions: float,
) -> CompatibilityReport:
    """Evaluate whether candidate dimensions are acceptable under the correlated prior."""
    norm_map: dict[str, float] = {}
    for k, v in dimensions.items():
        if k in prior.names:
            norm_map[k] = float(v)
        elif k.endswith("_m") and k[:-2] in prior.names:
            norm_map[k[:-2]] = float(v)

    if not norm_map:
        raise PreconditionError(
            "None of the evaluated dimensions exist in the prior model."
        )

    eval_names = list(norm_map.keys())
    indices = [prior.names.index(k) for k in eval_names]
    sub_mean = prior.mean[indices]
    sub_cov = prior.covariance[np.ix_(indices, indices)]
    sub_val = np.array([norm_map[k] for k in eval_names], dtype=np.float64)

    diff = sub_val - sub_mean
    inv_cov = np.linalg.inv(sub_cov)
    dist_sq = float(diff.T @ inv_cov @ diff)
    dist = math.sqrt(max(0.0, dist_sq))

    is_acceptable = dist <= max_mahalanobis
    return CompatibilityReport(
        is_acceptable=is_acceptable,
        mahalanobis_distance=dist,
        dimensions_evaluated=tuple(eval_names),
    )


@dataclass(frozen=True)
class DimeHumanPriorReport:
    """Structured receipt for human prior and range-of-motion constraints."""

    prior_version: str
    dimensions_modeled: tuple[str, ...]
    prior_influence_norm: float
    rom_feasible: bool
    inertia_realizable: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "prior_version": self.prior_version,
            "dimensions_modeled": list(self.dimensions_modeled),
            "prior_influence_norm": self.prior_influence_norm,
            "rom_feasible": self.rom_feasible,
            "inertia_realizable": self.inertia_realizable,
        }
