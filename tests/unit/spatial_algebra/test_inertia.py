"""Tests for src.shared.python.spatial_algebra.inertia (Issues #1949, #1744)."""

from __future__ import annotations

import numpy as np
import pytest
from src.shared.python.spatial_algebra.inertia import (
    mcI,
    mci,
    transform_spatial_inertia,
)


def _spherical_inertia(mass: float, r: float) -> np.ndarray:
    """Inertia tensor for a uniform sphere: I = 2/5 * m * r^2 * I_3."""
    return (2.0 / 5.0) * mass * r**2 * np.eye(3)


class TestMcI:
    def test_returns_6x6_matrix(self) -> None:
        inertia = mcI(1.0, np.zeros(3), _spherical_inertia(1.0, 0.1))
        assert inertia.shape == (6, 6)

    def test_lower_right_is_mass_times_identity(self) -> None:
        mass = 2.5
        inertia = mcI(mass, np.zeros(3), _spherical_inertia(mass, 0.1))
        np.testing.assert_allclose(inertia[3:, 3:], mass * np.eye(3), atol=1e-12)

    def test_zero_com_gives_block_diagonal(self) -> None:
        mass = 1.0
        I_com = _spherical_inertia(mass, 0.1)
        result = mcI(mass, np.zeros(3), I_com)
        # Off-diagonal blocks should be zero when COM = 0
        np.testing.assert_allclose(result[:3, 3:], 0.0, atol=1e-12)
        np.testing.assert_allclose(result[3:, :3], 0.0, atol=1e-12)

    def test_inertia_negative_mass_raises(self) -> None:
        with pytest.raises(ValueError):
            mcI(-1.0, np.zeros(3), np.eye(3))

    def test_inertia_zero_mass_raises(self) -> None:
        with pytest.raises(ValueError):
            mcI(0.0, np.zeros(3), np.eye(3))

    def test_wrong_com_shape_raises(self) -> None:
        with pytest.raises(ValueError):
            mcI(1.0, np.zeros(4), np.eye(3))

    def test_wrong_inertia_shape_raises(self) -> None:
        with pytest.raises(ValueError):
            mcI(1.0, np.zeros(3), np.eye(4))

    def test_result_is_symmetric(self) -> None:
        mass = 1.5
        com = np.array([0.05, 0.0, 0.1])
        I_com = _spherical_inertia(mass, 0.08)
        result = mcI(mass, com, I_com)
        np.testing.assert_allclose(result, result.T, atol=1e-12)

    def test_upper_left_equals_inertia_at_com(self) -> None:
        # When COM = 0, upper-left block should equal I_com
        mass = 2.0
        I_com = _spherical_inertia(mass, 0.1)
        result = mcI(mass, np.zeros(3), I_com)
        np.testing.assert_allclose(result[:3, :3], I_com, atol=1e-12)


class TestMci:
    def test_alias_same_as_mcI(self) -> None:
        mass = 1.0
        com = np.zeros(3)
        I_com = np.eye(3) * 0.01
        r1 = mcI(mass, com, I_com)
        r2 = mci(mass, com, I_com)
        np.testing.assert_array_equal(r1, r2)


class TestTransformSpatialInertia:
    def test_identity_transform_preserves_inertia(self) -> None:
        mass = 1.0
        inertia = mcI(mass, np.zeros(3), _spherical_inertia(mass, 0.1))
        X = np.eye(6)
        result = transform_spatial_inertia(inertia, X)
        np.testing.assert_allclose(result, inertia, atol=1e-12)

    def test_returns_6x6_matrix(self) -> None:
        mass = 2.0
        inertia = mcI(mass, np.zeros(3), _spherical_inertia(mass, 0.15))
        X = np.eye(6)
        result = transform_spatial_inertia(inertia, X)
        assert result.shape == (6, 6)

    def test_result_is_symmetric(self) -> None:
        mass = 1.5
        inertia = mcI(mass, np.array([0.05, 0.0, 0.1]), _spherical_inertia(mass, 0.1))
        X = np.eye(6)
        result = transform_spatial_inertia(inertia, X)
        np.testing.assert_allclose(result, result.T, atol=1e-10)
