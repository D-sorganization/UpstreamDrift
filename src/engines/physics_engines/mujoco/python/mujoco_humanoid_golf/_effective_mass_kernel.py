"""Shared effective-mass numerical kernels."""

from __future__ import annotations

import warnings

import numpy as np

from src.shared.python.core.numerical_constants import (
    EPSILON_SINGULARITY_DETECTION,
)


def _as_finite_float_array(name: str, value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def compute_effective_mass_from_solve(
    direction: np.ndarray, jacp: np.ndarray, mass_matrix: np.ndarray
) -> float:
    """Compute scalar effective mass without explicitly forming ``M^-1``."""
    direction = _as_finite_float_array("direction", direction)
    jacp = _as_finite_float_array("jacp", jacp)
    mass_matrix = _as_finite_float_array("mass_matrix", mass_matrix)

    if direction.shape != (3,):
        raise ValueError(f"direction must have shape (3,), got {direction.shape}")
    if jacp.ndim != 2 or jacp.shape[0] != 3:
        raise ValueError(f"jacp must have shape (3, nv), got {jacp.shape}")
    if mass_matrix.ndim != 2 or mass_matrix.shape[0] != mass_matrix.shape[1]:
        raise ValueError(f"mass_matrix must be square, got shape {mass_matrix.shape}")
    if jacp.shape[1] != mass_matrix.shape[0]:
        raise ValueError(
            "jacp velocity dimension must match mass_matrix size: "
            f"{jacp.shape[1]} != {mass_matrix.shape[0]}"
        )

    j_dir = direction @ jacp
    solved_j_dir = np.linalg.solve(mass_matrix, j_dir)
    if not np.all(np.isfinite(solved_j_dir)):
        raise ValueError("mass_matrix solve returned non-finite values")

    denominator = float(j_dir @ solved_j_dir.T + EPSILON_SINGULARITY_DETECTION)

    if abs(denominator) < 1e-8:
        warnings.warn(
            f"Effective mass denominator near zero: {denominator:.2e}. "
            "Robot is at or very close to a kinematic singularity in the "
            f"specified direction {direction}. Effective mass is extremely large.",
            category=UserWarning,
            stacklevel=2,
        )

    m_eff = 1.0 / denominator

    if m_eff < 0:
        raise ValueError(
            f"Computed negative effective mass: {m_eff:.2e} kg. "
            "This indicates a numerical error or modeling issue."
        )

    if not np.isfinite(m_eff):
        warnings.warn(
            f"Effective mass is non-finite: {m_eff}. "
            "Robot is at a kinematic singularity. "
            "Returning large finite value instead.",
            category=UserWarning,
            stacklevel=2,
        )
        m_eff = 1e10

    return float(m_eff)
