"""Target-relative shot-dispersion statistics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DispersionResult:
    """Robust center, covariance ellipse, and radial-error summary."""

    sample_count: int
    center_forward: float
    center_lateral: float
    mean_forward: float
    mean_lateral: float
    ellipse_major: float
    ellipse_minor: float
    ellipse_angle_rad: float
    area_95: float
    radial_rmse: float
    radial_p50: float
    radial_p90: float


def analyze_dispersion(
    frame: pd.DataFrame,
    *,
    forward: str = "carry_distance",
    lateral: str = "lateral_carry",
) -> DispersionResult:
    """Compute a 95% covariance ellipse and robust dispersion metrics."""
    missing = {forward, lateral} - set(frame.columns)
    if missing:
        raise ValueError(f"Dispersion columns not present: {sorted(missing)}")
    values = frame[[forward, lateral]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(values) < 3:
        raise ValueError("At least three complete shots are required for dispersion")
    points = values.to_numpy(float)
    robust_center = np.median(points, axis=0)
    mean_center = np.mean(points, axis=0)
    covariance = np.cov(points, rowvar=False, ddof=1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    eigenvectors = eigenvectors[:, order]
    chi_square_95 = 5.991464547
    radii = np.sqrt(eigenvalues * chi_square_95)
    major, minor = 2 * radii
    vector = eigenvectors[:, 0]
    angle = float(np.arctan2(vector[1], vector[0]))
    delta = points - robust_center
    radial = np.hypot(delta[:, 0], delta[:, 1])
    return DispersionResult(
        len(points),
        float(robust_center[0]),
        float(robust_center[1]),
        float(mean_center[0]),
        float(mean_center[1]),
        float(major),
        float(minor),
        angle,
        float(np.pi * radii[0] * radii[1]),
        float(np.sqrt(np.mean(radial**2))),
        float(np.quantile(radial, 0.5)),
        float(np.quantile(radial, 0.9)),
    )
