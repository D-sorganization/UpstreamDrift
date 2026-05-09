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


class TestCrossMotion:
    """Tests for cross_motion spatial operation."""

    def test_matches_crm_matrix(self) -> None:
        """cross_motion(v, m) should equal crm(v) @ m."""
        v = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        m = np.array([6.0, 5.0, 4.0, 3.0, 2.0, 1.0])
        expected = crm(v) @ m
        np.testing.assert_allclose(cross_motion(v, m), expected, atol=1e-14)

    def test_with_output_buffer(self) -> None:
        v = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        m = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
        out = np.zeros(6)
        result = cross_motion(v, m, out=out)
        assert result is out
        # w=(1,0,0) x m=(0,1,0,0,0,0) -> (0,0,1,0,0,0) in angular part
        assert out[2] == pytest.approx(1.0)

    def test_invalid_v_shape(self) -> None:
        with pytest.raises(ValueError):
            cross_motion(np.zeros(3), np.zeros(6))

    def test_invalid_m_shape(self) -> None:
        with pytest.raises(ValueError):
            cross_motion(np.zeros(6), np.zeros(3))


# ============================================================================
# Tests for fast variants
# ============================================================================


# ============================================================================
# Tests for cross_motion_axis
# ============================================================================


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
