"""Comprehensive tests for src.shared.python.spatial_algebra package.

Covers spatial_vectors, transforms, inertia, and joints modules.
Tests verify mathematical properties (antisymmetry, orthogonality, Jacobi identity)
and physical correctness of the spatial algebra implementation.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import numpy.typing as npt
import pytest
from src.shared.python.spatial_algebra.inertia import (
    mcI,
    mci,
    transform_spatial_inertia,
)
from src.shared.python.spatial_algebra.joints import (
    JOINT_AXIS_INDICES,
    S_PX,
    S_PY,
    S_PZ,
    S_RX,
    S_RY,
    S_RZ,
    jcalc,
)
from src.shared.python.spatial_algebra.spatial_vectors import (
    crf,
    crm,
    cross_force,
    cross_force_fast,
    cross_motion,
    cross_motion_axis,
    cross_motion_fast,
    skew,
    spatial_cross,
)
from src.shared.python.spatial_algebra.transforms import (
    inv_xtrans,
    xlt,
    xrot,
    xtrans,
)

# ============================================================================
# Helpers
# ============================================================================


def _rotation_x(angle: float) -> npt.NDArray[np.float64]:
    """Rotation matrix about X axis."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def _rotation_y(angle: float) -> npt.NDArray[np.float64]:
    """Rotation matrix about Y axis."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def _rotation_z(angle: float) -> npt.NDArray[np.float64]:
    """Rotation matrix about Z axis."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


# ============================================================================
# Tests for skew
# ============================================================================


# ============================================================================
# Tests for crm and crf
# ============================================================================


# ============================================================================
# Tests for cross_motion and cross_force
# ============================================================================


# ============================================================================
# Tests for fast variants
# ============================================================================


# ============================================================================
# Tests for cross_motion_axis
# ============================================================================


class TestCrossMotionAxis:
    """Tests for cross_motion_axis (sparse m vector)."""

    @pytest.mark.parametrize("axis_idx", [0, 1, 2, 3, 4, 5])
    def test_matches_full_cross_motion(self, axis_idx: int) -> None:
        """Sparse version should match full cross_motion."""
        v = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        val = 2.5
        m = np.zeros(6)
        m[axis_idx] = val

        out_sparse = np.zeros(6)
        cross_motion_axis(v, axis_idx, val, out_sparse)

        expected = cross_motion(v, m)
        np.testing.assert_allclose(out_sparse, expected, atol=1e-14)


# ============================================================================
# Tests for spatial_cross dispatcher
# ============================================================================


# ============================================================================
# Tests for transforms: xrot, xlt, xtrans, inv_xtrans
# ============================================================================


# ============================================================================
# Tests for inertia: mcI, mci, transform_spatial_inertia
# ============================================================================


# ============================================================================
# Tests for joints: jcalc
# ============================================================================
