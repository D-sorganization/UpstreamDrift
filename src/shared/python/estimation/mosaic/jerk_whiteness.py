"""Kinematic jerk prior, whiteness gate and declared discrepancy (MOSAIC-07, #11539).

This module provides:
1. :class:`KinematicJerkPrior`: penalises the third time derivative (jerk) of the
   spline trajectory, regularising acceleration noise without imposing artificial
   effort priors that bias inertia downward.
2. :func:`evaluate_whiteness_gate`: Gate 1 evaluation at :math:`\\gamma = 0`.  Verifies
   that marker residuals behave as white noise (unit RMS, zero time autocorrelation)
   on a sufficiently resolved basis, and flags under-resolved bases before dynamics
   weighting is introduced.
3. :func:`compute_declared_discrepancy` and :func:`validate_declared_discrepancy`:
   Gate 2 declared-discrepancy calculator enforcing
   :math:`\\sigma_\\tau \\ge \\epsilon_a \\cdot M_{\\text{scale}}`.  Below-floor
   declarations raise a precondition contract violation to prevent spurious torque
   amplification.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
import math
from typing import Literal, TypeAlias

import numpy as np
import numpy.typing as npt

from src.shared.python.core.contracts import ensure, require
from src.shared.python.estimation.mosaic.kinematic_basis import BSplineBasis
from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)

FloatArray: TypeAlias = npt.NDArray[np.float64]

__all__ = [
    "DeclaredDiscrepancyCalculator",
    "DiscrepancyGateResult",
    "KinematicJerkPrior",
    "WhitenessGateResult",
    "compute_basis_acceleration_error",
    "compute_declared_discrepancy",
    "evaluate_whiteness_gate",
    "validate_declared_discrepancy",
]


@dataclass(frozen=True)
class KinematicJerkPrior:
    """Quadratic penalty on trajectory jerk :math:`\\dddot{q}(t) = J(t) C`.

    Preconditions:
        basis has ``jerk`` defined (degree >= 3).
        ``weight >= 0.0``.
    """

    basis: BSplineBasis
    weight: float = 1.0

    def __post_init__(self) -> None:
        require(
            self.weight >= 0.0, "jerk prior weight must be non-negative", self.weight
        )
        require(
            self.basis.jerk is not None,
            "basis degree must be >= 3 to define jerk matrix",
            self.basis.degree,
        )

    @property
    def jerk_matrix(self) -> FloatArray:
        assert self.basis.jerk is not None
        return self.basis.jerk

    def compute_residual(self, coefficients: FloatArray) -> FloatArray:
        """Return weighted jerk residual :math:`w_{\\text{jerk}} J C`."""
        require(
            coefficients.shape[0] == self.basis.n_coefficients,
            "coefficients row count must match basis",
            coefficients.shape,
        )
        return self.weight * (self.jerk_matrix @ coefficients)

    def compute_cost(self, coefficients: FloatArray) -> float:
        """Return sum of squared jerk residuals :math:`\\|w_{\\text{jerk}} J C\\|^2`."""
        res = self.compute_residual(coefficients)
        return float(np.sum(res**2))

    def normal_matrix(self) -> FloatArray:
        """Return Gauss-Newton normal matrix block :math:`w_{\\text{jerk}}^2 J^T J`."""
        return (self.weight**2) * (self.jerk_matrix.T @ self.jerk_matrix)

    def gradient(self, coefficients: FloatArray) -> FloatArray:
        """Return gradient :math:`w_{\\text{jerk}}^2 J^T J C`."""
        return self.normal_matrix() @ coefficients


@dataclass(frozen=True)
class WhitenessGateResult:
    """Outcome and underlying metrics of Gate 1 (Kinematic Whiteness)."""

    status: Literal["pass", "fail", "unqualified"]
    passed: bool
    rms: float
    autocorrelation_lag1: float
    durbin_watson: float
    reason: str
    details: dict[str, float] = field(default_factory=dict)


def _validate_whiteness_inputs(
    obs: FloatArray,
    pred: FloatArray,
    sigma: float,
) -> str | None:
    """Validate inputs for whiteness gate; return error string or None if valid."""
    if not (math.isfinite(sigma) and sigma > 0.0):
        return f"sigma must be positive and finite, got {sigma!r}"
    if obs.shape != pred.shape or obs.size == 0:
        return f"shape mismatch or empty input: obs {obs.shape} vs pred {pred.shape}"
    if not (np.all(np.isfinite(obs)) and np.all(np.isfinite(pred))):
        return "non-finite values in observed or predicted data"
    if obs.shape[0] < 4:
        return f"need at least 4 temporal samples, got {obs.shape[0]}"
    return None


def _compute_residual_statistics(
    e: FloatArray,
) -> tuple[float, float, float, dict[str, float]]:
    """Compute whitened RMS, mean lag-1 autocorrelation, and mean Durbin-Watson statistic."""
    n_samples = e.shape[0]
    rms = float(np.sqrt(np.mean(e**2)))
    e_flat_cols = e.reshape(n_samples, -1)
    n_cols = e_flat_cols.shape[1]

    autocorrs: list[float] = []
    dw_stats: list[float] = []
    for col in range(n_cols):
        series = e_flat_cols[:, col]
        diff = series - np.mean(series)
        var = np.sum(diff**2)
        rho1 = float(np.sum(diff[:-1] * diff[1:]) / var) if var > 1e-15 else 0.0
        autocorrs.append(rho1)

        denom = np.sum(series**2)
        dw = float(np.sum(np.diff(series) ** 2) / denom) if denom > 1e-15 else 2.0
        dw_stats.append(dw)

    mean_autocorr = float(np.mean(autocorrs))
    mean_dw = float(np.mean(dw_stats))
    details = {
        "rms": rms,
        "mean_autocorrelation_lag1": mean_autocorr,
        "durbin_watson": mean_dw,
        "max_abs_autocorr": float(np.max(np.abs(autocorrs))),
    }
    return rms, mean_autocorr, mean_dw, details


def evaluate_whiteness_gate(
    observed: FloatArray,
    predicted: FloatArray,
    sigma: float,
    *,
    max_autocorr: float = 0.25,
    min_rms: float = 0.6,
    max_rms: float = 1.6,
    dw_min: float = 1.4,
    dw_max: float = 2.6,
) -> WhitenessGateResult:
    """Evaluate whether kinematic residuals at :math:`\\gamma = 0` behave as white noise.

    Returns:
        WhitenessGateResult with status 'pass', 'fail', or 'unqualified'.
    """
    obs = np.asarray(observed, dtype=np.float64)
    pred = np.asarray(predicted, dtype=np.float64)

    val_err = _validate_whiteness_inputs(obs, pred, sigma)
    if val_err is not None:
        return WhitenessGateResult(
            status="unqualified",
            passed=False,
            rms=float("nan"),
            autocorrelation_lag1=float("nan"),
            durbin_watson=float("nan"),
            reason=val_err,
        )

    e = (obs - pred) / sigma
    rms, mean_autocorr, mean_dw, details = _compute_residual_statistics(e)

    passed_rms = min_rms <= rms <= max_rms
    passed_autocorr = abs(mean_autocorr) <= max_autocorr
    passed_dw = dw_min <= mean_dw <= dw_max

    if passed_rms and passed_autocorr and passed_dw:
        return WhitenessGateResult(
            status="pass",
            passed=True,
            rms=rms,
            autocorrelation_lag1=mean_autocorr,
            durbin_watson=mean_dw,
            reason="Residuals are white noise: RMS near 1.0 and uncorrelated in time.",
            details=details,
        )

    reasons: list[str] = []
    if not passed_rms:
        reasons.append(f"normalized RMS {rms:.3f} outside [{min_rms}, {max_rms}]")
    if not passed_autocorr:
        reasons.append(
            f"lag-1 autocorrelation {mean_autocorr:.3f} exceeds tolerance {max_autocorr}"
        )
    if not passed_dw:
        reasons.append(f"Durbin-Watson {mean_dw:.3f} outside [{dw_min}, {dw_max}]")

    return WhitenessGateResult(
        status="fail",
        passed=False,
        rms=rms,
        autocorrelation_lag1=mean_autocorr,
        durbin_watson=mean_dw,
        reason="Under-resolved basis or non-white residuals: " + "; ".join(reasons),
        details=details,
    )


def compute_declared_discrepancy(
    basis_acceleration_error: float,
    mass_scale: float,
    safety_factor: float = 1.0,
) -> float:
    """Calculate the minimum dynamics discrepancy floor :math:`\\sigma_{\\tau,\\text{min}}`.

    Formula:
        :math:`\\sigma_{\\tau,\\text{min}} = \\epsilon_a \\cdot M_{\\text{scale}} \\cdot s`
    """
    require(
        basis_acceleration_error >= 0.0 and math.isfinite(basis_acceleration_error),
        "basis_acceleration_error must be non-negative and finite",
        basis_acceleration_error,
    )
    require(
        mass_scale > 0.0 and math.isfinite(mass_scale),
        "mass_scale must be positive and finite",
        mass_scale,
    )
    require(
        safety_factor >= 1.0 and math.isfinite(safety_factor),
        "safety_factor must be >= 1.0 and finite",
        safety_factor,
    )
    return float(basis_acceleration_error * mass_scale * safety_factor)


@dataclass(frozen=True)
class DiscrepancyGateResult:
    """Outcome and numbers of Gate 2 (Declared Discrepancy)."""

    status: Literal["pass", "fail", "unqualified"]
    passed: bool
    declared_sigma_tau: float
    min_discrepancy_floor: float
    basis_acceleration_error: float
    mass_scale: float
    reason: str


def validate_declared_discrepancy(
    declared_sigma_tau: float,
    basis_acceleration_error: float,
    mass_scale: float,
    safety_factor: float = 1.0,
) -> DiscrepancyGateResult:
    """Validate that declared discrepancy respects the basis discretisation floor.

    Raises:
        ContractViolationError: if declared_sigma_tau is strictly below the floor.
    """
    floor = compute_declared_discrepancy(
        basis_acceleration_error=basis_acceleration_error,
        mass_scale=mass_scale,
        safety_factor=safety_factor,
    )
    require(
        declared_sigma_tau >= floor,
        f"declared discrepancy {declared_sigma_tau} is below floor {floor} "
        f"(basis acceleration error {basis_acceleration_error} * mass scale {mass_scale})",
    )
    return DiscrepancyGateResult(
        status="pass",
        passed=True,
        declared_sigma_tau=declared_sigma_tau,
        min_discrepancy_floor=floor,
        basis_acceleration_error=basis_acceleration_error,
        mass_scale=mass_scale,
        reason=(
            f"Declared discrepancy {declared_sigma_tau:.4f} >= "
            f"discretisation floor {floor:.4f}."
        ),
    )


def compute_basis_acceleration_error(
    basis: BSplineBasis,
    trajectory_func: Callable[[FloatArray], FloatArray],
    acceleration_func: Callable[[FloatArray], FloatArray],
    times: FloatArray,
) -> float:
    """Measure empirical acceleration error of a basis on a test trajectory."""
    samples = trajectory_func(times)
    true_acc = acceleration_func(times)
    coeffs = basis.fit_least_squares(samples)
    _, _, spline_acc = basis.evaluate(coeffs)
    error = float(np.sqrt(np.mean((spline_acc - true_acc) ** 2)))
    ensure(math.isfinite(error), "acceleration error must be finite")
    return error


class DeclaredDiscrepancyCalculator:
    """Helper class encapsulating discrepancy calculations and policy bounds."""

    def __init__(self, mass_scale: float, safety_factor: float = 1.0) -> None:
        require(mass_scale > 0.0, "mass_scale must be positive", mass_scale)
        self.mass_scale = mass_scale
        self.safety_factor = safety_factor

    def compute_floor(self, basis_acceleration_error: float) -> float:
        return compute_declared_discrepancy(
            basis_acceleration_error, self.mass_scale, self.safety_factor
        )

    def validate(
        self, declared_sigma_tau: float, basis_acceleration_error: float
    ) -> DiscrepancyGateResult:
        return validate_declared_discrepancy(
            declared_sigma_tau,
            basis_acceleration_error,
            self.mass_scale,
            self.safety_factor,
        )
