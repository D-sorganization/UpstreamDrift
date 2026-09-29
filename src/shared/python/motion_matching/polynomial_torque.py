"""Shared polynomial torque evaluation for motion-matching engines."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from src.shared.python.motion_matching.validate_theta import COEFFS_PER_JOINT

__all__ = [
    "COEFFS_PER_JOINT",
    "POLY_DEGREE",
    "evaluate_polynomial_torque",
]

POLY_DEGREE: int = COEFFS_PER_JOINT - 1


def evaluate_polynomial_torque(
    coeffs: NDArray[np.float64],
    t: float,
) -> NDArray[np.float64]:
    """Evaluate per-joint degree-6 torques with lowest-power-first coeffs.

    ``coeffs[j, k]`` is the coefficient of ``t**k``. For the canonical
    seven-column layout this is ``[A, B, C, D, E, F, G]`` for
    ``A + B*t + ... + G*t**6``.
    """
    coeffs_arr = np.asarray(coeffs, dtype=np.float64)
    if coeffs_arr.ndim != 2:
        msg = f"coeffs must be 2D (n_joints, 7); got ndim={coeffs_arr.ndim}"
        raise ValueError(msg)
    if coeffs_arr.shape[1] != COEFFS_PER_JOINT:
        msg = (
            f"coeffs must have {COEFFS_PER_JOINT} columns "
            f"(degree {POLY_DEGREE}); got shape {coeffs_arr.shape}"
        )
        raise ValueError(msg)
    if not np.isfinite(t):
        raise ValueError(f"t must be finite, got {t!r}")

    out = coeffs_arr[:, POLY_DEGREE].copy()
    for k in range(POLY_DEGREE - 1, -1, -1):
        out = out * t + coeffs_arr[:, k]
    return out
