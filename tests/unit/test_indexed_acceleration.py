"""Tests for indexed acceleration analysis module.

Tests the indexed acceleration decomposition that implements Section H2
closure-verified acceleration decomposition.
"""

from __future__ import annotations

import numpy as np
import pytest

from shared.python.indexed_acceleration import IndexedAcceleration


class TestIndexedAccelerationDataclass:
    """Test IndexedAcceleration dataclass."""

    def test_initialization(self):
        """Test basic initialization with required components."""
        gravity = np.array([1.0, 2.0, 3.0])
        coriolis = np.array([0.1, 0.2, 0.3])
        applied_torque = np.array([0.5, 0.6, 0.7])
        constraint = np.array([0.0, 0.0, 0.0])
        external = np.array([0.2, 0.3, 0.4])

        acc = IndexedAcceleration(
            gravity=gravity,
            coriolis=coriolis,
            applied_torque=applied_torque,
            constraint=constraint,
            external=external,
        )

        np.testing.assert_array_equal(acc.gravity, gravity)
        np.testing.assert_array_equal(acc.coriolis, coriolis)
        np.testing.assert_array_equal(acc.applied_torque, applied_torque)
        np.testing.assert_array_equal(acc.constraint, constraint)
        np.testing.assert_array_equal(acc.external, external)
        assert acc.centrifugal is None

    def test_initialization_with_centrifugal(self):
        """Test initialization including optional centrifugal component."""
        gravity = np.array([1.0])
        coriolis = np.array([0.1])
        applied_torque = np.array([0.5])
        constraint = np.array([0.0])
        external = np.array([0.2])
        centrifugal = np.array([0.05])

        acc = IndexedAcceleration(
            gravity=gravity,
            coriolis=coriolis,
            applied_torque=applied_torque,
            constraint=constraint,
            external=external,
            centrifugal=centrifugal,
        )

        assert acc.centrifugal is not None
        np.testing.assert_array_equal(acc.centrifugal, centrifugal)

    def test_total_property_without_centrifugal(self):
        """Test that total property sums all components (without centrifugal)."""
        gravity = np.array([1.0, 0.0])
        coriolis = np.array([0.0, 2.0])
        applied_torque = np.array([0.5, 0.5])
        constraint = np.array([0.1, 0.1])
        external = np.array([0.2, 0.3])

        acc = IndexedAcceleration(
            gravity=gravity,
            coriolis=coriolis,
            applied_torque=applied_torque,
            constraint=constraint,
            external=external,
        )

        total = acc.total

        # Should sum all components
        expected = gravity + coriolis + applied_torque + constraint + external
        np.testing.assert_array_almost_equal(total, expected)

    def test_total_property_with_centrifugal(self):
        """Test that total property includes centrifugal when present."""
        gravity = np.array([1.0])
        coriolis = np.array([0.1])
        applied_torque = np.array([0.5])
        constraint = np.array([0.0])
        external = np.array([0.2])
        centrifugal = np.array([0.05])

        acc = IndexedAcceleration(
            gravity=gravity,
            coriolis=coriolis,
            applied_torque=applied_torque,
            constraint=constraint,
            external=external,
            centrifugal=centrifugal,
        )

        total = acc.total

        # Should include centrifugal in sum
        expected = (
            gravity + coriolis + applied_torque + constraint + external + centrifugal
        )
        np.testing.assert_array_almost_equal(total, expected)

    def test_total_with_zero_components(self):
        """Test total with all zero components."""
        zeros = np.zeros(3)

        acc = IndexedAcceleration(
            gravity=zeros.copy(),
            coriolis=zeros.copy(),
            applied_torque=zeros.copy(),
            constraint=zeros.copy(),
            external=zeros.copy(),
        )

        total = acc.total

        np.testing.assert_array_almost_equal(total, zeros)

    def test_total_with_negative_components(self):
        """Test that total correctly handles negative accelerations."""
        acc = IndexedAcceleration(
            gravity=np.array([5.0]),
            coriolis=np.array([-2.0]),
            applied_torque=np.array([3.0]),
            constraint=np.array([-1.0]),
            external=np.array([0.0]),
        )

        total = acc.total

        # 5 - 2 + 3 - 1 + 0 = 5
        expected = np.array([5.0])
        np.testing.assert_array_almost_equal(total, expected)

    def test_multidimensional_components(self):
        """Test with multi-DOF system (multiple joints/dimensions)."""
        n_dof = 5
        gravity = np.ones(n_dof) * 9.81
        coriolis = np.ones(n_dof) * 0.1
        applied_torque = np.ones(n_dof) * 2.0
        constraint = np.zeros(n_dof)
        external = np.ones(n_dof) * 0.5

        acc = IndexedAcceleration(
            gravity=gravity,
            coriolis=coriolis,
            applied_torque=applied_torque,
            constraint=constraint,
            external=external,
        )

        total = acc.total

        assert total.shape == (n_dof,)
        expected = gravity + coriolis + applied_torque + constraint + external
        np.testing.assert_array_almost_equal(total, expected)


class TestPhysicalRealism:
    """Test physical realism of indexed accelerations."""

    def test_gravity_dominant_in_free_fall(self):
        """Test that gravity dominates in free fall scenario."""
        acc = IndexedAcceleration(
            gravity=np.array([0.0, 0.0, -9.81]),  # Downward
            coriolis=np.array([0.0, 0.0, 0.0]),
            applied_torque=np.array([0.0, 0.0, 0.0]),
            constraint=np.array([0.0, 0.0, 0.0]),
            external=np.array([0.0, 0.0, 0.0]),
        )

        total = acc.total

        # Total should be mostly gravity
        np.testing.assert_array_almost_equal(total, acc.gravity)

    def test_applied_torque_accelerates_system(self):
        """Test that applied torque contributes to acceleration."""
        acc = IndexedAcceleration(
            gravity=np.zeros(3),
            coriolis=np.zeros(3),
            applied_torque=np.array([10.0, 0.0, 0.0]),  # Torque in first DOF
            constraint=np.zeros(3),
            external=np.zeros(3),
        )

        total = acc.total

        # Total should equal applied torque
        assert total[0] == 10.0
        assert total[1] == 0.0
        assert total[2] == 0.0

    def test_constraint_opposes_motion(self):
        """Test that constraints can oppose other accelerations."""
        # Example: Ground reaction force opposes gravity
        acc = IndexedAcceleration(
            gravity=np.array([0.0, 0.0, -9.81]),
            coriolis=np.zeros(3),
            applied_torque=np.zeros(3),
            constraint=np.array([0.0, 0.0, 9.81]),  # Exactly cancels gravity
            external=np.zeros(3),
        )

        total = acc.total

        # Net acceleration should be zero (static equilibrium)
        np.testing.assert_array_almost_equal(total, np.zeros(3))


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_dof_system(self):
        """Test with single degree of freedom."""
        acc = IndexedAcceleration(
            gravity=np.array([1.0]),
            coriolis=np.array([0.1]),
            applied_torque=np.array([0.5]),
            constraint=np.array([0.0]),
            external=np.array([0.2]),
        )

        total = acc.total

        assert total.shape == (1,)
        assert total[0] == pytest.approx(1.8, rel=1e-10)

    def test_very_large_accelerations(self):
        """Test with very large acceleration values."""
        acc = IndexedAcceleration(
            gravity=np.array([1e6]),
            coriolis=np.array([1e5]),
            applied_torque=np.array([1e7]),
            constraint=np.array([0.0]),
            external=np.array([1e4]),
        )

        total = acc.total

        # Should handle large values correctly
        assert np.isfinite(total[0])

    def test_very_small_accelerations(self):
        """Test with very small acceleration values."""
        acc = IndexedAcceleration(
            gravity=np.array([1e-10]),
            coriolis=np.array([1e-11]),
            applied_torque=np.array([1e-12]),
            constraint=np.array([0.0]),
            external=np.array([1e-13]),
        )

        total = acc.total

        # Should handle small values correctly
        assert np.isfinite(total[0])

    def test_mixed_positive_negative_cancellation(self):
        """Test cancellation of positive and negative components."""
        acc = IndexedAcceleration(
            gravity=np.array([10.0, -5.0, 3.0]),
            coriolis=np.array([-10.0, 5.0, -3.0]),  # Exactly opposite
            applied_torque=np.zeros(3),
            constraint=np.zeros(3),
            external=np.zeros(3),
        )

        total = acc.total

        # Should cancel out to zero
        np.testing.assert_array_almost_equal(total, np.zeros(3))


class TestNumericalAccuracy:
    """Test numerical accuracy of summation."""

    def test_summation_order_independence(self):
        """Test that total is computed consistently."""
        gravity = np.array([1.0, 2.0])
        coriolis = np.array([0.1, 0.2])
        applied_torque = np.array([0.5, 0.6])
        constraint = np.array([0.01, 0.02])
        external = np.array([0.001, 0.002])

        acc1 = IndexedAcceleration(
            gravity=gravity,
            coriolis=coriolis,
            applied_torque=applied_torque,
            constraint=constraint,
            external=external,
        )

        # Create another instance with same values
        acc2 = IndexedAcceleration(
            gravity=gravity.copy(),
            coriolis=coriolis.copy(),
            applied_torque=applied_torque.copy(),
            constraint=constraint.copy(),
            external=external.copy(),
        )

        # Totals should be identical
        np.testing.assert_array_equal(acc1.total, acc2.total)

    def test_total_recomputed_each_time(self):
        """Test that total is a computed property, not cached."""
        acc = IndexedAcceleration(
            gravity=np.array([1.0]),
            coriolis=np.array([0.1]),
            applied_torque=np.array([0.5]),
            constraint=np.array([0.0]),
            external=np.array([0.2]),
        )

        total1 = acc.total

        # Modify a component
        acc.gravity[0] = 2.0

        total2 = acc.total

        # Total should have changed
        assert total2[0] != total1[0]
        assert total2[0] == pytest.approx(2.8, rel=1e-10)
