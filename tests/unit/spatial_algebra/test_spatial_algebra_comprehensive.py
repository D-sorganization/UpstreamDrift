"""Comprehensive tests for src.shared.python.spatial_algebra package.

Covers spatial_vectors, transforms, inertia, and joints modules.
Tests verify mathematical properties (antisymmetry, orthogonality, Jacobi identity)
and physical correctness of the spatial algebra implementation.
"""

from __future__ import annotations

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


class TestSkew:
    """Tests for skew-symmetric matrix construction."""

    def test_basic(self) -> None:
        v = np.array([1.0, 2.0, 3.0])
        s = skew(v)
        assert s.shape == (3, 3)

    def test_antisymmetric(self) -> None:
        """skew(v) should be antisymmetric: S = -S^T."""
        v = np.array([1.0, 2.0, 3.0])
        s = skew(v)
        np.testing.assert_allclose(s, -s.T)

    def test_cross_product_equivalence(self) -> None:
        """skew(v) @ u should equal np.cross(v, u)."""
        v = np.array([1.0, 2.0, 3.0])
        u = np.array([4.0, 5.0, 6.0])
        np.testing.assert_allclose(skew(v) @ u, np.cross(v, u))

    def test_zero_vector(self) -> None:
        s = skew(np.zeros(3))
        np.testing.assert_allclose(s, np.zeros((3, 3)))

    def test_invalid_shape_raises(self) -> None:
        with pytest.raises(ValueError, match="3x1"):
            skew(np.array([1.0, 2.0]))


# ============================================================================
# Tests for crm and crf
# ============================================================================


class TestCrmCrf:
    """Tests for spatial cross product motion/force operators."""

    @pytest.mark.parametrize("func", [crm, crf], ids=["crm", "crf"])
    def test_shape(self, func: object) -> None:
        v = np.zeros(6)
        result = func(v)
        assert result.shape == (6, 6)

    @pytest.mark.parametrize("func", [crm, crf], ids=["crm", "crf"])
    def test_invalid_shape(self, func: object) -> None:
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


class TestCrossForce:
    """Tests for cross_force spatial operation."""

    def test_matches_crf_matrix(self) -> None:
        """cross_force(v, f) should equal crf(v) @ f."""
        v = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        f = np.array([6.0, 5.0, 4.0, 3.0, 2.0, 1.0])
        expected = crf(v) @ f
        np.testing.assert_allclose(cross_force(v, f), expected, atol=1e-14)

    def test_negative_of_transpose_relation(self) -> None:
        """v x* f = -(v x)^T @ f."""
        v = np.array([1.0, 0.5, -1.0, 2.0, -0.5, 1.5])
        f = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        expected = -crm(v).T @ f
        np.testing.assert_allclose(cross_force(v, f), expected, atol=1e-14)


# ============================================================================
# Tests for fast variants
# ============================================================================


class TestFastVariants:
    """Tests for cross_motion_fast and cross_force_fast."""

    def test_cross_motion_fast_matches(self) -> None:
        v = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        m = np.array([6.0, 5.0, 4.0, 3.0, 2.0, 1.0])
        out_fast = np.zeros(6)
        cross_motion_fast(v, m, out_fast)
        expected = cross_motion(v, m)
        np.testing.assert_allclose(out_fast, expected, atol=1e-14)

    def test_cross_force_fast_matches(self) -> None:
        v = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        f = np.array([6.0, 5.0, 4.0, 3.0, 2.0, 1.0])
        out_fast = np.zeros(6)
        cross_force_fast(v, f, out_fast)
        expected = cross_force(v, f)
        np.testing.assert_allclose(out_fast, expected, atol=1e-14)


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


class TestSpatialCross:
    """Tests for spatial_cross dispatcher."""

    def test_motion_type(self) -> None:
        v = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        u = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
        result = spatial_cross(v, u, cross_type="motion")
        expected = cross_motion(v, u)
        np.testing.assert_allclose(result, expected)

    def test_force_type(self) -> None:
        v = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        f = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
        result = spatial_cross(v, f, cross_type="force")
        expected = cross_force(v, f)
        np.testing.assert_allclose(result, expected)

    def test_invalid_type_raises(self) -> None:
        with pytest.raises(ValueError, match="must be"):
            spatial_cross(np.zeros(6), np.zeros(6), cross_type="invalid")  # type: ignore[arg-type]


# ============================================================================
# Tests for transforms: xrot, xlt, xtrans, inv_xtrans
# ============================================================================


class TestXrot:
    """Tests for pure rotation spatial transform."""

    def test_identity_rotation(self) -> None:
        X = xrot(np.eye(3))
        np.testing.assert_allclose(X, np.eye(6), atol=1e-14)

    def test_block_diagonal(self) -> None:
        """Result should be block diagonal [E 0; 0 E]."""
        E = _rotation_z(np.pi / 4)
        X = xrot(E)
        np.testing.assert_allclose(X[:3, :3], E)
        np.testing.assert_allclose(X[3:, 3:], E)
        np.testing.assert_allclose(X[:3, 3:], np.zeros((3, 3)), atol=1e-14)
        np.testing.assert_allclose(X[3:, :3], np.zeros((3, 3)), atol=1e-14)

    def test_invalid_shape(self) -> None:
        with pytest.raises(ValueError, match="3x3"):
            xrot(np.eye(4))

    def test_invalid_rotation(self) -> None:
        with pytest.raises(ValueError, match="rotation matrix"):
            xrot(np.ones((3, 3)))  # Not orthogonal


class TestXlt:
    """Tests for pure translation spatial transform."""

    def test_zero_translation_is_identity(self) -> None:
        X = xlt(np.zeros(3))
        np.testing.assert_allclose(X, np.eye(6), atol=1e-14)

    def test_shape(self) -> None:
        X = xlt(np.array([1.0, 2.0, 3.0]))
        assert X.shape == (6, 6)

    def test_upper_left_identity(self) -> None:
        """Upper-left 3x3 block should be identity."""
        X = xlt(np.array([1.0, 0.0, 0.0]))
        np.testing.assert_allclose(X[:3, :3], np.eye(3))

    def test_invalid_shape(self) -> None:
        with pytest.raises(ValueError, match="3x1"):
            xlt(np.array([1.0, 2.0]))


class TestXtrans:
    """Tests for general spatial coordinate transformation."""

    def test_identity(self) -> None:
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


class TestInvXtrans:
    """Tests for inverse spatial transform."""

    def test_inverse_identity(self) -> None:
        X_inv = inv_xtrans(np.eye(3), np.zeros(3))
        np.testing.assert_allclose(X_inv, np.eye(6), atol=1e-14)

    def test_inverse_roundtrip(self) -> None:
        """xtrans(E, r) @ inv_xtrans(E, r) should be identity."""
        E = _rotation_z(np.pi / 4)
        r = np.array([1.0, -0.5, 2.0])
        X = xtrans(E, r)
        X_inv = inv_xtrans(E, r)
        np.testing.assert_allclose(X @ X_inv, np.eye(6), atol=1e-12)

    def test_invalid_shapes(self) -> None:
        with pytest.raises(ValueError, match="3x3"):
            inv_xtrans(np.eye(4), np.zeros(3))
        with pytest.raises(ValueError, match="3x1"):
            inv_xtrans(np.eye(3), np.zeros(2))


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

    def test_symmetric(self) -> None:
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


class TestTransformSpatialInertia:
    """Tests for spatial inertia transformation."""

    def test_identity_transform(self) -> None:
        """Identity transform should not change inertia."""
        I_B = mcI(5.0, np.zeros(3), np.diag([1.0, 2.0, 3.0]))
        X = np.eye(6)
        I_A = transform_spatial_inertia(I_B, X)
        np.testing.assert_allclose(I_A, I_B, atol=1e-12)

    def test_symmetry_preserved(self) -> None:
        """Transformed inertia should remain symmetric."""
        I_B = mcI(5.0, np.zeros(3), np.diag([1.0, 2.0, 3.0]))
        E = _rotation_z(np.pi / 4)
        X = xrot(E)
        I_A = transform_spatial_inertia(I_B, X)
        np.testing.assert_allclose(I_A, I_A.T, atol=1e-12)

    def test_invalid_shapes(self) -> None:
        with pytest.raises(ValueError, match="6x6"):
            transform_spatial_inertia(np.zeros((3, 3)), np.eye(6))
        with pytest.raises(ValueError, match="6x6"):
            transform_spatial_inertia(np.eye(6), np.zeros((3, 3)))


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
