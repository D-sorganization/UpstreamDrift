"""Shared validation for oriented ellipsoid glyph geometry."""

from __future__ import annotations

import numpy as np

__all__ = ["validate_oriented_ellipsoid_fields"]


def validate_oriented_ellipsoid_fields(
    centre: tuple[float, float, float],
    axes: tuple[tuple[float, float, float], ...],
    radii: tuple[float, float, float],
    rgba: tuple[float, float, float, float],
) -> tuple[
    tuple[float, float, float],
    np.ndarray,
    tuple[float, float, float],
    tuple[float, float, float, float],
]:
    """Validate oriented-ellipsoid geometry shared by solid and wire glyphs."""
    centre_arr = np.asarray(centre, dtype=float)
    if centre_arr.shape != (3,) or not np.isfinite(centre_arr).all():
        raise ValueError(f"centre must be a finite 3-vector, got {centre!r}")
    axes_arr = np.asarray(axes, dtype=float)
    if axes_arr.shape != (3, 3) or not np.isfinite(axes_arr).all():
        raise ValueError("axes must be a finite 3x3 matrix")
    if not np.allclose(axes_arr.T @ axes_arr, np.eye(3), atol=1e-6):
        raise ValueError("axes must have orthonormal columns (within 1e-6)")
    if np.linalg.det(axes_arr) < 0.0:
        axes_arr = axes_arr.copy()
        axes_arr[:, 2] *= -1.0
    radii_arr = np.asarray(radii, dtype=float)
    if (
        radii_arr.shape != (3,)
        or not np.isfinite(radii_arr).all()
        or (radii_arr <= 0).any()
    ):
        raise ValueError(f"radii must be 3 finite positive values, got {radii!r}")
    rgba_arr = np.asarray(rgba, dtype=float)
    if rgba_arr.shape != (4,) or not ((rgba_arr >= 0.0) & (rgba_arr <= 1.0)).all():
        raise ValueError(f"rgba must be 4 values in [0, 1], got {rgba!r}")
    return (
        tuple(float(v) for v in centre_arr),
        axes_arr,
        tuple(float(v) for v in radii_arr),
        tuple(float(v) for v in rgba_arr),
    )
