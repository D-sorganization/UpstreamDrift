"""Tests for src.shared.python.spatial_algebra.transforms (Issues #1949, #1744)."""

from __future__ import annotations

import numpy as np
import pytest
from src.shared.python.spatial_algebra.transforms import (
    inv_xtrans,
    xlt,
    xrot,
    xtrans,
)


def _rotation_x(angle: float) -> np.ndarray:
    """Create a rotation matrix around X axis."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)


def _identity_rotation() -> np.ndarray:
    return np.eye(3, dtype=float)


class TestXrot:
    def test_identity_rotation_gives_6x6_identity(self) -> None:
        E = _identity_rotation()
        result = xrot(E)
        np.testing.assert_allclose(result, np.eye(6), atol=1e-12)

    def test_returns_6x6_matrix(self) -> None:
        E = _rotation_x(0.5)
        result = xrot(E)
        assert result.shape == (6, 6)

    def test_non_rotation_matrix_raises(self) -> None:
        E = np.ones((3, 3))  # not a rotation matrix
        with pytest.raises(ValueError):
            xrot(E)

    def test_transforms_wrong_shape_raises(self) -> None:
        E = np.eye(4)
        with pytest.raises(ValueError):
            xrot(E)

    def test_upper_left_3x3_is_rotation(self) -> None:
        E = _rotation_x(np.pi / 4)
        result = xrot(E)
        np.testing.assert_allclose(result[:3, :3], E, atol=1e-12)
        np.testing.assert_allclose(result[3:, 3:], E, atol=1e-12)

    def test_upper_right_and_lower_left_are_zero(self) -> None:
        E = _rotation_x(0.3)
        result = xrot(E)
        np.testing.assert_allclose(result[:3, 3:], 0.0, atol=1e-12)
        np.testing.assert_allclose(result[3:, :3], 0.0, atol=1e-12)


class TestXlt:
    def test_zero_translation_gives_identity(self) -> None:
        r = np.array([0.0, 0.0, 0.0])
        result = xlt(r)
        np.testing.assert_allclose(result, np.eye(6), atol=1e-12)

    def test_returns_6x6_matrix(self) -> None:
        r = np.array([1.0, 2.0, 3.0])
        result = xlt(r)
        assert result.shape == (6, 6)

    def test_transforms_wrong_shape_raises(self) -> None:
        r = np.array([1.0, 2.0])
        with pytest.raises((ValueError, Exception)):
            xlt(r)

    def test_upper_3x3_is_identity(self) -> None:
        r = np.array([1.0, 0.0, 0.0])
        result = xlt(r)
        np.testing.assert_allclose(result[:3, :3], np.eye(3), atol=1e-12)

    def test_lower_right_is_identity(self) -> None:
        r = np.array([1.0, 2.0, 0.0])
        result = xlt(r)
        np.testing.assert_allclose(result[3:, 3:], np.eye(3), atol=1e-12)


class TestXtrans:
    def test_identity_gives_6x6_identity_approx(self) -> None:
        E = _identity_rotation()
        r = np.zeros(3)
        result = xtrans(E, r)
        np.testing.assert_allclose(result, np.eye(6), atol=1e-12)

    def test_returns_6x6_matrix(self) -> None:
        E = _rotation_x(0.5)
        r = np.array([1.0, 0.0, 0.0])
        result = xtrans(E, r)
        assert result.shape == (6, 6)

    def test_wrong_rotation_shape_raises(self) -> None:
        E = np.eye(4)
        r = np.zeros(3)
        with pytest.raises(ValueError):
            xtrans(E, r)

    def test_wrong_translation_shape_raises(self) -> None:
        E = _identity_rotation()
        r = np.zeros(4)
        with pytest.raises(ValueError):
            xtrans(E, r)


class TestInvXtrans:
    def test_returns_6x6_matrix(self) -> None:
        E = _identity_rotation()
        r = np.array([1.0, 0.0, 0.0])
        result = inv_xtrans(E, r)
        assert result.shape == (6, 6)

    def test_inv_of_identity_is_identity(self) -> None:
        E = _identity_rotation()
        r = np.zeros(3)
        result = inv_xtrans(E, r)
        np.testing.assert_allclose(result, np.eye(6), atol=1e-12)

    def test_xtrans_times_inv_xtrans_is_identity(self) -> None:
        E = _rotation_x(0.3)
        r = np.array([0.5, -0.2, 0.1])
        X = xtrans(E, r)
        X_inv = inv_xtrans(E, r)
        np.testing.assert_allclose(X @ X_inv, np.eye(6), atol=1e-10)

    def test_wrong_rotation_shape_raises(self) -> None:
        with pytest.raises(ValueError):
            inv_xtrans(np.eye(4), np.zeros(3))
