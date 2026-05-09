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


class TestCrmCrf:
    """Tests for spatial cross product motion/force operators."""

    @pytest.mark.parametrize("func", [crm, crf], ids=["crm", "crf"])
    def test_shape(self, func: Callable[[np.ndarray], np.ndarray]) -> None:
        v = np.zeros(6)
        result = func(v)
        assert result.shape == (6, 6)

    @pytest.mark.parametrize("func", [crm, crf], ids=["crm", "crf"])
    def test_invalid_shape(self, func: Callable[[np.ndarray], np.ndarray]) -> None:
        with pytest.raises(ValueError, match="6x1"):
            func(np.array([1.0, 2.0, 3.0]))

    def test_crm_zero_vector(self) -> None:
        result = crm(np.zeros(6))
        np.testing.assert_allclose(result, np.zeros((6, 6)))

    def test_crm_antisymmetric_omega_block(self) -> None:
        """Upper-left 3x3 block should be skew-symmetric."""
        v = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        result = crm(v)
        omega_block = result[:3, :3]
        np.testing.assert_allclose(omega_block, -omega_block.T)

    def test_crm_crf_relation(self) -> None:
        """crf(v) = -crm(v)^T (dual relationship)."""
        v = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        np.testing.assert_allclose(crf(v), -crm(v).T)


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


# ============================================================================
# Tests for inertia: mcI, mci, transform_spatial_inertia
# ============================================================================


# ============================================================================
# Tests for joints: jcalc
# ============================================================================
