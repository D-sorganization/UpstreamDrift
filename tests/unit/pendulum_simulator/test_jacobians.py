"""Tests for src.shared.python.pendulum_simulator.jacobians (Issues #1949, #1744)."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.pendulum_simulator.jacobians import (
    ellipsoid_from_jacobian,
    ellipsoids_double,
    ellipsoids_triple,
    jacobian_double,
    jacobian_triple,
)


class TestEllipsoidFromJacobian:
    def test_returns_four_elements(self) -> None:
        J = np.eye(2)
        result = ellipsoid_from_jacobian(J)
        assert len(result) == 4

    def test_identity_jacobian_unit_semi_axes(self) -> None:
        J = np.eye(2)
        _, mob_semi_axes, force_semi_axes, svs = ellipsoid_from_jacobian(J)
        np.testing.assert_allclose(mob_semi_axes, [1.0, 1.0], atol=1e-10)
        np.testing.assert_allclose(force_semi_axes, [1.0, 1.0], atol=1e-10)

    def test_directions_shape(self) -> None:
        J = np.eye(2)
        directions, _, _, _ = ellipsoid_from_jacobian(J)
        assert directions.shape == (2, 2)

    def test_mob_semi_axes_shape(self) -> None:
        J = np.array([[2.0, 0.0], [0.0, 1.0]])
        _, mob, _, _ = ellipsoid_from_jacobian(J)
        assert mob.shape == (2,)

    def test_scaled_jacobian_correct_semi_axes(self) -> None:
        J = np.array([[3.0, 0.0], [0.0, 2.0]])
        _, mob, force, _ = ellipsoid_from_jacobian(J)
        # Singular values should be 3 and 2
        np.testing.assert_allclose(sorted(mob, reverse=True), [3.0, 2.0], atol=1e-10)
        np.testing.assert_allclose(sorted(force), [1.0 / 3.0, 1.0 / 2.0], atol=1e-10)

    def test_rectangular_jacobian_shape_2_3(self) -> None:
        J = np.array([[1.0, 0.5, 0.2], [0.0, 1.0, 0.3]])
        directions, mob, force, svs = ellipsoid_from_jacobian(J)
        assert mob.shape == (2,)

    def test_singular_jacobian_force_none(self) -> None:
        # Column of zeros → singular
        J = np.array([[1.0, 0.0], [0.0, 0.0]])
        _, _, force, _ = ellipsoid_from_jacobian(J)
        assert force is None

    def test_assertion_on_wrong_shape(self) -> None:
        with pytest.raises(AssertionError):
            ellipsoid_from_jacobian(np.eye(3))

    def test_assertion_on_nan(self) -> None:
        J = np.array([[1.0, np.nan], [0.0, 1.0]])
        with pytest.raises(AssertionError):
            ellipsoid_from_jacobian(J)


class TestJacobianDouble:
    def test_returns_wrist_and_tip_keys(self) -> None:
        result = jacobian_double(0.0, 0.0, 1.0, 1.0)
        assert "wrist" in result and "tip" in result

    def test_wrist_jacobian_shape(self) -> None:
        result = jacobian_double(0.0, 0.0, 1.0, 1.0)
        assert result["wrist"].shape == (2, 2)

    def test_tip_jacobian_shape(self) -> None:
        result = jacobian_double(0.0, 0.0, 1.0, 1.0)
        assert result["tip"].shape == (2, 2)

    def test_wrist_phi_column_is_zero(self) -> None:
        # Phi has no effect on wrist position
        result = jacobian_double(0.0, 0.0, 1.0, 1.0)
        np.testing.assert_allclose(result["wrist"][:, 1], [0.0, 0.0], atol=1e-12)

    def test_finite_outputs(self) -> None:
        result = jacobian_double(0.3, -0.2, 0.5, 0.4)
        for J in result.values():
            assert np.all(np.isfinite(J))

    def test_zero_l1_raises(self) -> None:
        with pytest.raises(AssertionError):
            jacobian_double(0.0, 0.0, 0.0, 1.0)

    def test_zero_l2_raises(self) -> None:
        with pytest.raises(AssertionError):
            jacobian_double(0.0, 0.0, 1.0, 0.0)


class TestEllipsoidsDouble:
    def test_returns_wrist_and_tip(self) -> None:
        result = ellipsoids_double(0.0, 0.0, 1.0, 1.0)
        assert "wrist" in result and "tip" in result

    def test_each_entry_has_required_keys(self) -> None:
        result = ellipsoids_double(0.1, 0.2, 0.6, 0.5)
        for name in ("wrist", "tip"):
            entry = result[name]
            for key in ("jacobian", "directions", "mob_semi_axes", "singular_values"):
                assert key in entry

    def test_mob_semi_axes_positive(self) -> None:
        result = ellipsoids_double(0.1, 0.2, 0.6, 0.5)
        np.testing.assert_array_less(0, result["tip"]["mob_semi_axes"])


class TestJacobianTriple:
    def test_returns_three_endpoint_keys(self) -> None:
        result = jacobian_triple(0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
        assert "wrist1" in result and "wrist2" in result and "tip" in result

    def test_all_shapes_are_2_3(self) -> None:
        result = jacobian_triple(0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
        for J in result.values():
            assert J.shape == (2, 3)

    def test_wrist1_phi1_phi2_columns_zero(self) -> None:
        # phi1 and phi2 have no effect on wrist1 position
        result = jacobian_triple(0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
        np.testing.assert_allclose(result["wrist1"][:, 1], [0.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(result["wrist1"][:, 2], [0.0, 0.0], atol=1e-12)

    def test_wrist2_phi2_column_zero(self) -> None:
        result = jacobian_triple(0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
        np.testing.assert_allclose(result["wrist2"][:, 2], [0.0, 0.0], atol=1e-12)

    def test_finite_outputs(self) -> None:
        result = jacobian_triple(0.2, -0.1, 0.3, 0.5, 0.4, 0.3)
        for J in result.values():
            assert np.all(np.isfinite(J))


class TestEllipsoidsTriple:
    def test_returns_three_keys(self) -> None:
        result = ellipsoids_triple(0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
        assert set(result.keys()) == {"wrist1", "wrist2", "tip"}

    def test_mob_semi_axes_shape(self) -> None:
        result = ellipsoids_triple(0.1, 0.2, -0.1, 0.5, 0.4, 0.3)
        for name in ("wrist1", "wrist2", "tip"):
            assert result[name]["mob_semi_axes"].shape == (2,)
