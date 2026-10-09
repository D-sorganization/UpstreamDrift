"""User-selected Gaussian delivery covariance for paired synthetic shots.

The variables are assumptions about delivery, not measured player data. A
single returned array is reused across patterns to pair shot indices.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

NOISE_FIELDS = (
    "shaft_rotation_deg",
    "path_deg",
    "speed_mps",
    "contact_horizontal_m",
    "contact_vertical_m",
    "shaft_lean_deg",
    "attack_angle_deg",
)


@dataclass(frozen=True)
class NoiseSpec:
    """Standard deviations and an optional PSD correlation in field order."""

    standard_deviations: tuple[float, ...]
    correlation: tuple[tuple[float, ...], ...] | None = None

    def __post_init__(self) -> None:
        n = len(NOISE_FIELDS)
        sd = self.standard_deviations
        if len(sd) != n or any(not math.isfinite(v) or v < 0 for v in sd):
            raise ValueError(
                "standard deviations must be finite, nonnegative, and complete"
            )
        if self.correlation is None:
            return
        corr = np.asarray(self.correlation, dtype=float)
        if (
            corr.shape != (n, n)
            or not np.isfinite(corr).all()
            or not np.allclose(corr, corr.T, atol=1e-12, rtol=0)
            or not np.allclose(np.diag(corr), 1.0, atol=1e-12, rtol=0)
            or np.max(np.abs(corr)) > 1.0 + 1e-12
        ):
            raise ValueError("correlation must be a symmetric unit-diagonal matrix")
        if float(np.linalg.eigvalsh(corr).min()) < -1e-10:
            raise ValueError("correlation must be positive semidefinite")

    def correlation_matrix(self) -> np.ndarray:
        """Return a copy so callers cannot mutate this frozen specification."""
        if self.correlation is None:
            return np.eye(len(NOISE_FIELDS))
        return np.asarray(self.correlation, dtype=float).copy()


def sample_paired_noise(spec: NoiseSpec, *, n_shots: int, seed: int) -> np.ndarray:
    """Sample Gaussian errors; use the returned rows for every shot shape.

    Eigenvalue clipping removes only sub-roundoff negative roots of an already
    validated PSD matrix. Zero-SD variables remain exactly zero.
    """
    if not isinstance(n_shots, int) or isinstance(n_shots, bool) or n_shots < 1:
        raise ValueError("n_shots must be a positive integer")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    eigenvalues, eigenvectors = np.linalg.eigh(spec.correlation_matrix())
    root = eigenvectors @ np.diag(np.sqrt(np.maximum(eigenvalues, 0.0)))
    standard = np.random.default_rng(seed).standard_normal((n_shots, len(NOISE_FIELDS)))
    samples = (standard @ root.T) * np.asarray(spec.standard_deviations)
    if not np.isfinite(samples).all():
        raise RuntimeError("delivery-noise sampling produced nonfinite values")
    return samples
