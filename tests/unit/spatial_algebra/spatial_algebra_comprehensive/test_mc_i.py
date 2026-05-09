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


# ============================================================================
# Tests for inertia: mcI, mci, transform_spatial_inertia
# ============================================================================


class TestMcI:
    """Tests for spatial inertia matrix construction."""

    def test_point_mass_at_origin(self) -> None:
        """Point mass at origin should have m*I in lower-right."""
        mass = 2.0
        I_s = mcI(mass, np.zeros(3), np.zeros((3, 3)))
        # Lower-right should be m*I3
        np.testing.assert_allclose(I_s[3:, 3:], mass * np.eye(3))
        # Upper-left should be zero (no rotational inertia)
        np.testing.assert_allclose(I_s[:3, :3], np.zeros((3, 3)), atol=1e-14)

    def test_spatial_algebra_comprehensive_symmetric(self) -> None:
        """Spatial inertia matrix should be symmetric."""
        I_com = np.diag([0.1, 0.2, 0.3])
        I_s = mcI(5.0, np.array([0.1, 0.2, 0.3]), I_com)
        np.testing.assert_allclose(I_s, I_s.T, atol=1e-14)

    def test_positive_semidefinite(self) -> None:
        """Spatial inertia should be positive semi-definite."""
        I_com = np.diag([1.0, 2.0, 3.0])
        I_s = mcI(10.0, np.array([0.5, 0.0, 0.0]), I_com)
        eigenvalues = np.linalg.eigvalsh(I_s)
        assert np.all(eigenvalues >= -1e-10)

    @pytest.mark.parametrize(
        "mass, com, inertia, match",
        [
            (-1.0, np.zeros(3), np.zeros((3, 3)), "positive"),
            (1.0, np.zeros(2), np.zeros((3, 3)), "3x1"),
            (1.0, np.zeros(3), np.zeros((2, 2)), "3x3"),
        ],
        ids=["negative-mass", "invalid-com-shape", "invalid-inertia-shape"],
    )
    def test_invalid_inputs_raise(
        self,
        mass: float,
        com: np.ndarray,
        inertia: np.ndarray,
        match: str,
    ) -> None:
        with pytest.raises(ValueError, match=match):
            mcI(mass, com, inertia)

    def test_alias_mci(self) -> None:
        """mci should produce same result as mcI."""
        mass, com, I_com = 3.0, np.array([0.1, 0.2, 0.3]), np.diag([1.0, 2.0, 3.0])
        np.testing.assert_allclose(mci(mass, com, I_com), mcI(mass, com, I_com))


# ============================================================================
# Tests for joints: jcalc
# ============================================================================
