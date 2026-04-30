"""Unit tests for spatial_algebra modules (spatial_vectors, inertia, joints)."""

from __future__ import annotations

import numpy as np
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

# ---------------------------------------------------------------------------
# spatial_vectors.py
# ---------------------------------------------------------------------------


class TestSkew:
    """Tests for skew-symmetric matrix construction."""

    def test_skew_antisymmetric(self) -> None:
        v = np.array([1.0, 2.0, 3.0])
        S = skew(v)
        assert np.allclose(S + S.T, 0)

    def test_skew_cross_product(self) -> None:
        """skew(v) @ u == cross(v, u)."""
        v = np.array([1.0, 2.0, 3.0])
        u = np.array([4.0, 5.0, 6.0])
        assert np.allclose(skew(v) @ u, np.cross(v, u))

    def test_skew_shape(self) -> None:
        v = np.array([0.0, 0.0, 1.0])
        assert skew(v).shape == (3, 3)

    def test_skew_invalid_shape(self) -> None:
        with pytest.raises(ValueError):
            skew(np.array([1.0, 2.0]))

    def test_skew_zero_vector(self) -> None:
        S = skew(np.zeros(3))
        assert np.allclose(S, 0)


class TestCrm:
    """Tests for crm: spatial cross product for motion vectors."""

    def test_crm_shape(self) -> None:
        v = np.zeros(6)
        assert crm(v).shape == (6, 6)

    def test_crm_antisymmetric_top_left(self) -> None:
        """Top-left 3x3 block of crm should equal skew of angular part."""
        v = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        C = crm(v)
        S = skew(v[:3])
        assert np.allclose(C[:3, :3], S)

    def test_crm_invalid_shape(self) -> None:
        with pytest.raises(ValueError):
            crm(np.zeros(5))


class TestCrf:
    """Tests for crf: spatial cross product for force vectors (dual)."""

    def test_crf_shape(self) -> None:
        v = np.zeros(6)
        assert crf(v).shape == (6, 6)

    def test_crf_invalid_shape(self) -> None:
        with pytest.raises(ValueError):
            crf(np.zeros(5))

    def test_crf_is_negative_transpose_crm(self) -> None:
        """crf(v) == -crm(v).T for spatial algebra."""
        v = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        assert np.allclose(crf(v), -crm(v).T)


class TestCrossMotion:
    """Tests for cross_motion function."""

    def test_cross_motion_shape(self) -> None:
        v = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        m = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
        result = cross_motion(v, m)
        assert result.shape == (6,)

    def test_cross_motion_equals_crm(self) -> None:
        """cross_motion(v, m) should equal crm(v) @ m."""
        v = np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0])
        m = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
        assert np.allclose(cross_motion(v, m), crm(v) @ m)

    def test_cross_motion_invalid_v(self) -> None:
        with pytest.raises(ValueError):
            cross_motion(np.zeros(5), np.zeros(6))

    def test_cross_motion_invalid_m(self) -> None:
        with pytest.raises(ValueError):
            cross_motion(np.zeros(6), np.zeros(5))

    def test_cross_motion_with_out(self) -> None:
        v = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        m = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
        out = np.zeros(6)
        result = cross_motion(v, m, out=out)
        assert result is out


class TestCrossForce:
    """Tests for cross_force function."""

    def test_cross_force_shape(self) -> None:
        v = np.zeros(6)
        f = np.zeros(6)
        assert cross_force(v, f).shape == (6,)

    def test_cross_force_equals_crf(self) -> None:
        v = np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0])
        f = np.array([0.0, 0.0, 1.0, 0.0, 1.0, 0.0])
        assert np.allclose(cross_force(v, f), crf(v) @ f)

    def test_cross_force_with_out(self) -> None:
        v = np.zeros(6)
        f = np.zeros(6)
        out = np.zeros(6)
        result = cross_force(v, f, out=out)
        assert result is out


class TestFastCrossProducts:
    """Tests for cross_motion_fast, cross_force_fast, cross_motion_axis."""

    def test_cross_motion_fast_consistent(self) -> None:
        v = np.array([1.0, 2.0, 3.0, 0.5, 0.5, 0.5])
        m = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        out_fast = np.zeros(6)
        cross_motion_fast(v, m, out_fast)
        out_ref = cross_motion(v, m)
        assert np.allclose(out_fast, out_ref)

    def test_cross_force_fast_consistent(self) -> None:
        v = np.array([1.0, 2.0, 3.0, 0.5, 0.5, 0.5])
        f = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        out_fast = np.zeros(6)
        cross_force_fast(v, f, out_fast)
        out_ref = cross_force(v, f)
        assert np.allclose(out_fast, out_ref)

    def test_cross_motion_axis_index_0(self) -> None:
        v = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        m_full = np.zeros(6)
        m_full[0] = 1.0
        out_axis = np.zeros(6)
        cross_motion_axis(v, axis_idx=0, val=1.0, out=out_axis)
        out_ref = cross_motion(v, m_full)
        assert np.allclose(out_axis, out_ref)

    @pytest.mark.parametrize("axis_idx", [0, 1, 2, 3, 4, 5])
    def test_cross_motion_axis_all_indices(self, axis_idx: int) -> None:
        v = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        m = np.zeros(6)
        m[axis_idx] = 2.0
        out = np.zeros(6)
        cross_motion_axis(v, axis_idx=axis_idx, val=2.0, out=out)
        ref = cross_motion(v, m)
        assert np.allclose(out, ref)


class TestSpatialCross:
    """Tests for the spatial_cross dispatcher."""

    def test_motion_type(self) -> None:
        v = np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0])
        m = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        result = spatial_cross(v, m, cross_type="motion")
        expected = cross_motion(v, m)
        assert np.allclose(result, expected)

    def test_force_type(self) -> None:
        v = np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0])
        f = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        result = spatial_cross(v, f, cross_type="force")
        expected = cross_force(v, f)
        assert np.allclose(result, expected)

    def test_invalid_type_raises(self) -> None:
        v = np.zeros(6)
        u = np.zeros(6)
        with pytest.raises(ValueError):
            spatial_cross(v, u, cross_type="invalid")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# inertia.py
# ---------------------------------------------------------------------------


class TestMcI:
    """Tests for mcI and mci functions."""

    @pytest.fixture
    def sphere_inertia(self) -> tuple:
        mass = 2.0
        com = np.zeros(3)
        I_com = (2.0 / 5.0 * mass) * np.eye(3)
        return mass, com, I_com

    def test_mci_shape(self, sphere_inertia: tuple) -> None:
        mass, com, I_com = sphere_inertia
        M = mcI(mass, com, I_com)
        assert M.shape == (6, 6)

    def test_mci_symmetric(self, sphere_inertia: tuple) -> None:
        mass, com, I_com = sphere_inertia
        M = mcI(mass, com, I_com)
        assert np.allclose(M, M.T)

    def test_mci_mass_at_origin(self, sphere_inertia: tuple) -> None:
        mass, com, I_com = sphere_inertia
        M = mcI(mass, com, I_com)
        # Bottom-right 3x3 is mass * I
        assert np.allclose(M[3:, 3:], mass * np.eye(3))

    def test_mci_alias(self, sphere_inertia: tuple) -> None:
        mass, com, I_com = sphere_inertia
        assert np.allclose(mcI(mass, com, I_com), mci(mass, com, I_com))

    def test_mci_negative_mass_raises(self) -> None:
        with pytest.raises(ValueError):
            mcI(-1.0, np.zeros(3), np.eye(3))

    def test_mci_zero_mass_raises(self) -> None:
        with pytest.raises(ValueError):
            mcI(0.0, np.zeros(3), np.eye(3))

    def test_mci_wrong_com_shape(self) -> None:
        with pytest.raises(ValueError):
            mcI(1.0, np.zeros(2), np.eye(3))

    def test_mci_wrong_inertia_shape(self) -> None:
        with pytest.raises(ValueError):
            mcI(1.0, np.zeros(3), np.eye(4))

    def test_mci_offcentre_mass(self) -> None:
        """Non-zero COM should add offset inertia via parallel axis theorem."""
        mass = 1.0
        com = np.array([1.0, 0.0, 0.0])
        I_com = np.eye(3)
        M = mcI(mass, com, I_com)
        # Should not be same as zero-COM case
        M_origin = mcI(mass, np.zeros(3), I_com)
        assert not np.allclose(M, M_origin)


class TestTransformSpatialInertia:
    """Tests for transform_spatial_inertia."""

    def test_identity_transform(self) -> None:
        mass = 1.0
        I_mat = mcI(mass, np.zeros(3), np.eye(3))
        X = np.eye(6)
        I_transformed = transform_spatial_inertia(I_mat, X)
        assert np.allclose(I_transformed, I_mat)

    def test_transformed_is_symmetric(self) -> None:
        mass = 2.0
        I_mat = mcI(mass, np.array([0.1, 0.2, 0.3]), np.eye(3))
        X = np.eye(6)
        X[:3, :3] = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])  # 90deg rotation
        I_t = transform_spatial_inertia(I_mat, X)
        assert np.allclose(I_t, I_t.T)

    def test_invalid_I_shape(self) -> None:
        with pytest.raises(ValueError):
            transform_spatial_inertia(np.eye(5), np.eye(6))

    def test_invalid_X_shape(self) -> None:
        with pytest.raises(ValueError):
            transform_spatial_inertia(np.eye(6), np.eye(5))


# ---------------------------------------------------------------------------
# joints.py
# ---------------------------------------------------------------------------


class TestJcalc:
    """Tests for jcalc joint kinematics."""

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
    def test_motion_subspace(self, jtype: str, expected_s: np.ndarray) -> None:
        xj, s, dof = jcalc(jtype, q=0.0)
        assert np.allclose(s, expected_s)

    @pytest.mark.parametrize(
        "jtype,expected_dof",
        [
            ("Rx", 0),
            ("Ry", 1),
            ("Rz", 2),
            ("Px", 3),
            ("Py", 4),
            ("Pz", 5),
        ],
    )
    def test_dof_index(self, jtype: str, expected_dof: int) -> None:
        _, _, dof = jcalc(jtype, q=0.0)
        assert dof == expected_dof

    def test_Rx_at_zero(self) -> None:
        """At q=0 Rx should give identity-like rotation in xj."""
        xj, _, _ = jcalc("Rx", q=0.0)
        assert xj.shape == (6, 6)
        # cos(0)=1, sin(0)=0
        assert np.isclose(xj[1, 1], 1.0)
        assert np.isclose(xj[1, 2], 0.0)

    def test_Rz_at_pi_half(self) -> None:
        xj, _, _ = jcalc("Rz", q=np.pi / 2)
        assert np.isclose(xj[0, 0], 0.0, atol=1e-9)
        assert np.isclose(xj[0, 1], -1.0, atol=1e-9)

    def test_Px_transform(self) -> None:
        xj, _, _ = jcalc("Px", q=1.0)
        assert np.isclose(xj[4, 2], 1.0)
        assert np.isclose(xj[5, 1], -1.0)

    def test_Py_transform(self) -> None:
        xj, _, _ = jcalc("Py", q=2.0)
        assert np.isclose(xj[3, 2], -2.0)
        assert np.isclose(xj[5, 0], 2.0)

    def test_Pz_transform(self) -> None:
        xj, _, _ = jcalc("Pz", q=3.0)
        assert np.isclose(xj[3, 1], 3.0)
        assert np.isclose(xj[4, 0], -3.0)

    def test_invalid_joint_raises(self) -> None:
        with pytest.raises(ValueError):
            jcalc("Xy", q=0.0)

    def test_with_out_buffer(self) -> None:
        """Should use the provided buffer."""
        buf = np.zeros((6, 6))
        xj, _, _ = jcalc("Rx", q=0.5, out=buf)
        assert xj is buf

    def test_joint_axis_indices_complete(self) -> None:
        assert set(JOINT_AXIS_INDICES.keys()) == {"Rx", "Ry", "Rz", "Px", "Py", "Pz"}
