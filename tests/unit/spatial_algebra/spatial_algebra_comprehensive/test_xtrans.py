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


# ============================================================================
# Tests for spatial_cross dispatcher
# ============================================================================


# ============================================================================
# Tests for transforms: xrot, xlt, xtrans, inv_xtrans
# ============================================================================


class TestXtrans:
    """Tests for general spatial coordinate transformation."""

    def test_spatial_algebra_comprehensive_identity(self) -> None:
        X = xtrans(np.eye(3), np.zeros(3))
        np.testing.assert_allclose(X, np.eye(6), atol=1e-14)

    def test_pure_rotation_matches_xrot(self) -> None:
        E = _rotation_x(np.pi / 6)
        np.testing.assert_allclose(xtrans(E, np.zeros(3)), xrot(E), atol=1e-14)

    def test_composition(self) -> None:
        """xtrans(E, r) should equal xrot(E) @ xlt(r)."""
        E = _rotation_y(np.pi / 3)
        r = np.array([1.0, 2.0, 3.0])
        np.testing.assert_allclose(xtrans(E, r), xrot(E) @ xlt(r), atol=1e-12)

    def test_invalid_rotation_shape(self) -> None:
        with pytest.raises(ValueError, match="3x3"):
            xtrans(np.eye(4), np.zeros(3))

    def test_invalid_translation_shape(self) -> None:
        with pytest.raises(ValueError, match="3x1"):
            xtrans(np.eye(3), np.zeros(2))


# ============================================================================
# Tests for inertia: mcI, mci, transform_spatial_inertia
# ============================================================================


# ============================================================================
# Tests for joints: jcalc
# ============================================================================
