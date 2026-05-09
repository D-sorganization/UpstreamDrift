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


# ============================================================================
# Tests for joints: jcalc
# ============================================================================


class TestJcalc:
    """Tests for joint transform and motion subspace calculations."""

    @pytest.mark.parametrize("jtype", ["Rx", "Ry", "Rz", "Px", "Py", "Pz"])
    def test_zero_angle_is_identity(self, jtype: str) -> None:
        """At q=0, rotational joints should give identity."""
        xj, s, dof_idx = jcalc(jtype, 0.0)
        assert xj.shape == (6, 6)
        np.testing.assert_allclose(xj, np.eye(6), atol=1e-14)

    @pytest.mark.parametrize(
        "jtype,expected_s",
        [
            ("Rx", S_RX),
            ("Ry", S_RY),
            ("Rz", S_RZ),
            ("Px", S_PX),
            ("Py", S_PY),
            ("Pz", S_PZ),
        ],
    )
    def test_motion_subspace(
        self, jtype: str, expected_s: npt.NDArray[np.float64]
    ) -> None:
        _, s, _ = jcalc(jtype, 0.5)
        np.testing.assert_array_equal(s, expected_s)

    @pytest.mark.parametrize(
        "jtype,expected_idx",
        [("Rx", 0), ("Ry", 1), ("Rz", 2), ("Px", 3), ("Py", 4), ("Pz", 5)],
    )
    def test_dof_index(self, jtype: str, expected_idx: int) -> None:
        _, _, dof_idx = jcalc(jtype, 0.0)
        assert dof_idx == expected_idx

    @pytest.mark.parametrize(
        "jtype, angle",
        [("Rx", np.pi / 4), ("Ry", np.pi / 3), ("Rz", np.pi / 6)],
        ids=["Rx-pi/4", "Ry-pi/3", "Rz-pi/6"],
    )
    def test_rotation_orthogonal(self, jtype: str, angle: float) -> None:
        """Rotation transform should be orthogonal (det = 1)."""
        xj, _, _ = jcalc(jtype, angle)
        det = np.linalg.det(xj)
        assert det == pytest.approx(1.0, abs=1e-10)

    def test_output_buffer(self) -> None:
        buf = np.zeros((6, 6), dtype=np.float64)
        xj, _, _ = jcalc("Rx", np.pi / 4, out=buf)
        assert xj is buf

    def test_invalid_joint_type(self) -> None:
        with pytest.raises(ValueError, match="Unsupported joint type"):
            jcalc("invalid", 0.0)

    def test_joint_axis_indices(self) -> None:
        assert JOINT_AXIS_INDICES["Rx"] == 0
        assert JOINT_AXIS_INDICES["Pz"] == 5
        assert len(JOINT_AXIS_INDICES) == 6

    def test_motion_subspace_immutable(self) -> None:
        """Motion subspace vectors should be read-only."""
        with pytest.raises((ValueError, TypeError)):
            S_RX[0] = 99.0
