"""Tests for src.shared.python.spatial_algebra.spatial_vectors module."""

from __future__ import annotations

import numpy as np
import pytest

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


class TestSkew:
    """Tests for skew-symmetric matrix construction."""

    def test_skew_matrix_antisymmetry(self) -> None:
        v = np.array([1.0, 2.0, 3.0])
        S = skew(v)
        np.testing.assert_allclose(S + S.T, np.zeros((3, 3)), atol=1e-15)

    def test_skew_cross_product_equivalence(self) -> None:
        """skew(v) @ u == cross(v, u)."""
        v = np.array([1.0, 2.0, 3.0])
        u = np.array([4.0, 5.0, 6.0])
        result = skew(v) @ u
        expected = np.cross(v, u)
        np.testing.assert_allclose(result, expected, atol=1e-12)

    def test_skew_wrong_size(self) -> None:
        with pytest.raises(ValueError):
            skew(np.array([1.0, 2.0]))


class TestCrm:
    """Tests for spatial motion cross product operator."""

    def test_crm_shape(self) -> None:
        v = np.zeros(6)
        M = crm(v)
        assert M.shape == (6, 6)

    def test_crm_zero_vector(self) -> None:
        M = crm(np.zeros(6))
        np.testing.assert_allclose(M, np.zeros((6, 6)))

    def test_crm_wrong_size(self) -> None:
        with pytest.raises(ValueError):
            crm(np.zeros(3))

    def test_crm_antisymmetry_upper_block(self) -> None:
        """Upper-left 3x3 block is skew-symmetric of angular part."""
        v = np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0])
        M = crm(v)
        upper = M[:3, :3]
        np.testing.assert_allclose(upper + upper.T, np.zeros((3, 3)), atol=1e-15)


class TestCrf:
    """Tests for spatial force cross product operator."""

    def test_crf_shape(self) -> None:
        v = np.zeros(6)
        F = crf(v)
        assert F.shape == (6, 6)

    def test_crf_is_negative_transpose_of_crm(self) -> None:
        """For motion v, crf(v) = -crm(v).T"""
        v = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        np.testing.assert_allclose(crf(v), -crm(v).T, atol=1e-12)


class TestCrossMotion:
    """Tests for cross_motion function."""

    def test_cross_motion_zero(self) -> None:
        result = cross_motion(np.zeros(6), np.ones(6))
        np.testing.assert_allclose(result, np.zeros(6))

    def test_cross_motion_equals_crm_product(self) -> None:
        v = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        m = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        expected = crm(v) @ m
        actual = cross_motion(v, m)
        np.testing.assert_allclose(actual, expected, atol=1e-12)

    def test_cross_motion_with_out(self) -> None:
        v = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        m = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
        out = np.empty(6)
        result = cross_motion(v, m, out=out)
        assert result is out


class TestCrossForce:
    """Tests for cross_force function."""

    def test_cross_force_zero(self) -> None:
        result = cross_force(np.zeros(6), np.ones(6))
        np.testing.assert_allclose(result, np.zeros(6))

    def test_cross_force_equals_crf_product(self) -> None:
        v = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        f = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        expected = crf(v) @ f
        actual = cross_force(v, f)
        np.testing.assert_allclose(actual, expected, atol=1e-12)


class TestCrossFast:
    """Tests for optimized fast cross product functions."""

    def test_fast_motion_matches_regular(self) -> None:
        v = np.array([1.0, -2.0, 3.0, 4.0, -5.0, 6.0])
        m = np.array([0.5, 1.5, -0.5, 2.0, -1.0, 0.0])
        expected = cross_motion(v, m)
        out = np.empty(6)
        cross_motion_fast(v, m, out)
        np.testing.assert_allclose(out, expected, atol=1e-12)

    def test_fast_force_matches_regular(self) -> None:
        v = np.array([1.0, -2.0, 3.0, 4.0, -5.0, 6.0])
        f = np.array([0.5, 1.5, -0.5, 2.0, -1.0, 0.0])
        expected = cross_force(v, f)
        out = np.empty(6)
        cross_force_fast(v, f, out)
        np.testing.assert_allclose(out, expected, atol=1e-12)


class TestCrossMotionAxis:
    """Tests for sparse cross_motion_axis function."""

    def test_axis_0(self) -> None:
        v = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        m_sparse = np.zeros(6)
        m_sparse[0] = 1.5
        expected = cross_motion(v, m_sparse)
        out = np.empty(6)
        cross_motion_axis(v, 0, 1.5, out)
        np.testing.assert_allclose(out, expected, atol=1e-12)

    def test_axis_5(self) -> None:
        v = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        m_sparse = np.zeros(6)
        m_sparse[5] = 2.0
        expected = cross_motion(v, m_sparse)
        out = np.empty(6)
        cross_motion_axis(v, 5, 2.0, out)
        np.testing.assert_allclose(out, expected, atol=1e-12)


class TestSpatialCross:
    """Tests for the spatial_cross dispatcher."""

    def test_motion_type(self) -> None:
        v = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        u = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        result = spatial_cross(v, u, cross_type="motion")
        expected = cross_motion(v, u)
        np.testing.assert_allclose(result, expected)

    def test_force_type(self) -> None:
        v = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        u = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        result = spatial_cross(v, u, cross_type="force")
        expected = cross_force(v, u)
        np.testing.assert_allclose(result, expected)

    def test_invalid_type_raises(self) -> None:
        with pytest.raises(ValueError, match="cross_type"):
            spatial_cross(np.zeros(6), np.zeros(6), cross_type="invalid")
