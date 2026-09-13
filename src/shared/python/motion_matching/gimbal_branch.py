"""Numerical branch bounds for three distinct serial rotation axes."""

import numpy as np


def gimbal_branch_interval(initial_angle: float, margin: float) -> tuple[float, float]:
    """Bound the middle angle away from pi/2+k*pi on its original branch.

    This is a numerical path-search restriction, not a physical joint limit.
    It applies to three distinct serial orthogonal rotation axes, not proper
    Euler sequences with repeated axes. Initial states are never clamped.
    """
    if (
        not np.isfinite(initial_angle)
        or not np.isfinite(margin)
        or not 0 < margin < np.pi / 2
    ):
        raise ValueError("Finite initial angle and margin in (0,pi/2) required")
    branch = np.floor((initial_angle + np.pi / 2) / np.pi)
    lower = float(-np.pi / 2 + branch * np.pi + margin)
    upper = float(np.pi / 2 + branch * np.pi - margin)
    if not lower <= initial_angle <= upper:
        raise ValueError("Original angle is inside the excluded singularity margin")
    return lower, upper
