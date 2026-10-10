"""Descriptive two-dimensional statistics for golf-shot landing endpoints.

Coordinates use ``x`` downrange and ``y`` to the player's right. All error
statistics are relative to the target point ``(target_x_m, 0)``. The principal
axis is an unoriented covariance eigenvector, with angles measured from positive
downrange toward positive right and normalized to ``[-90, 90)`` degrees.

These metrics describe the supplied endpoints. With face variation as the only
uncertain input, the covariance is not calibrated two-dimensional player
precision and must not be presented as such.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import TypeAlias

import numpy as np

Point2D: TypeAlias = tuple[float, float]


def landing_dispersion(
    points: Iterable[tuple[float, float]],
    target_x_m: float,
    *,
    nominal_point: tuple[float, float] | None = None,
) -> dict[str, object]:
    """Summarize aimed landing endpoints relative to a downrange target.

    Args:
        points: At least two finite ``(downrange_x_m, lateral_y_m)`` pairs.
        target_x_m: Target downrange coordinate; target lateral coordinate is 0.
        nominal_point: Optional nominal endpoint used for mean offset and RMSE.

    Returns:
        A JSON-friendly mapping of sample covariance, correlations, empirical
        radial target errors, principal covariance axis, quadrant fractions,
        and optional nominal-centered metrics. Correlations are ``None`` when
        either variable has zero sample variance.

    Raises:
        ValueError: If coordinates, target, or nominal point are invalid.
    """
    if not isinstance(target_x_m, (int, float)) or not math.isfinite(target_x_m):
        raise ValueError("target_x_m must be finite")
    try:
        coordinates = np.asarray(list(points), dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("points must be finite (x, y) pairs") from exc
    if coordinates.ndim != 2 or coordinates.shape[1] != 2 or len(coordinates) < 2:
        raise ValueError("points must contain at least two (x, y) pairs")
    if not np.isfinite(coordinates).all():
        raise ValueError("points must be finite")

    nominal: np.ndarray | None = None
    if nominal_point is not None:
        try:
            nominal = np.asarray(nominal_point, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("nominal_point must be a finite (x, y) pair") from exc
        if nominal.shape != (2,) or not np.isfinite(nominal).all():
            raise ValueError("nominal_point must be a finite (x, y) pair")

    target = np.array((float(target_x_m), 0.0))
    errors = coordinates - target
    downrange_error = errors[:, 0]
    lateral_error = errors[:, 1]
    covariance = np.cov(errors, rowvar=False, ddof=1)
    downrange_variance = float(covariance[0, 0])
    lateral_variance = float(covariance[1, 1])
    cross_covariance = float(covariance[0, 1])

    def correlation(left: np.ndarray, right: np.ndarray) -> float | None:
        left_std = float(np.std(left, ddof=1))
        right_std = float(np.std(right, ddof=1))
        if left_std == 0.0 or right_std == 0.0:
            return None
        return float(np.corrcoef(left, right)[0, 1])

    radial_error = np.hypot(downrange_error, lateral_error)
    radial_carry = np.hypot(coordinates[:, 0], coordinates[:, 1])
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalues = eigenvalues[::-1]
    principal_orientation: float | None = None
    if not np.isclose(eigenvalues[0], eigenvalues[1], rtol=1e-12, atol=1e-15):
        axis = eigenvectors[:, -1]
        principal_orientation = float(
            (
                math.degrees(math.atan2(float(axis[1]), float(axis[0])) + math.pi / 2)
                % 180.0
            )
            - 90.0
        )

    n = len(coordinates)
    quadrants = {
        "left_long": (lateral_error < 0) & (downrange_error > 0),
        "left_short": (lateral_error < 0) & (downrange_error < 0),
        "right_long": (lateral_error > 0) & (downrange_error > 0),
        "right_short": (lateral_error > 0) & (downrange_error < 0),
    }
    conditional = {
        side: (float(np.mean(downrange_error[mask])) if bool(np.any(mask)) else None)
        for side, mask in (
            ("left", lateral_error < 0),
            ("right", lateral_error > 0),
        )
    }

    stats: dict[str, object] = {
        "n": n,
        "coordinate_convention": "x downrange, y right; target=(target_x_m, 0)",
        "covariance_convention": "unbiased sample covariance (ddof=1), m^2",
        "mean_downrange_error_m": float(np.mean(downrange_error)),
        "mean_lateral_error_m": float(np.mean(lateral_error)),
        "downrange_variance_m2": downrange_variance,
        "lateral_variance_m2": lateral_variance,
        "covariance_downrange_lateral_m2": cross_covariance,
        "corr_lateral_downrange_error": correlation(lateral_error, downrange_error),
        "corr_lateral_radial_carry": correlation(lateral_error, radial_carry),
        "median_radial_target_error_m": float(np.median(radial_error)),
        "p95_radial_target_error_m": float(np.quantile(radial_error, 0.95)),
        "principal_axis_eigenvalues_m2": tuple(float(x) for x in eigenvalues),
        "principal_axis_orientation_deg": principal_orientation,
        "principal_axis_angle_convention": (
            "unoriented axis from +downrange toward +right, normalized to [-90, 90)"
        ),
        "quadrant_fractions": {
            name: float(np.count_nonzero(mask) / n) for name, mask in quadrants.items()
        },
        "quadrant_fraction_convention": (
            "strictly left/right and long/short; axis-aligned endpoints enter no "
            "quadrant; fractions use all endpoints as denominator"
        ),
        "conditional_mean_downrange_error_m": conditional,
        "interpretation_limit": (
            "Descriptive endpoint covariance only; face variation alone does not "
            "calibrate two-dimensional player precision."
        ),
    }
    if nominal is not None:
        nominal_errors = coordinates - nominal
        mean_nominal_error = np.mean(nominal_errors, axis=0)
        stats["nominal_mean_offset_m"] = tuple(float(x) for x in mean_nominal_error)
        stats["nominal_rmse_m"] = float(
            np.sqrt(np.mean(np.sum(nominal_errors**2, axis=1)))
        )
    return stats


__all__ = ["Point2D", "landing_dispersion"]
