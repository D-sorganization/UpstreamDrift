"""Runtime DbC tests for coordination_metrics contracts.

Tests the require()/ensure() contracts added to:
- compute_coupling_angles: ensure() angles in [0, 360)
- compute_coordination_metrics: ensure() percentages sum to 100,
  mean angle in [0, 360), variability >= 0
- compute_rolling_correlation: require() window_size >= 2,
  ensure() correlations in [-1, 1]
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

import numpy as np


def _make_mixin(n: int = 200, n_joints: int = 3) -> object:
    """Create a mock CoordinationMetricsMixin with realistic data."""
    from src.shared.python.analysis.coordination_metrics import (
        CoordinationMetricsMixin,
    )

    rng = np.random.default_rng(42)
    obj = MagicMock(spec=CoordinationMetricsMixin)
    obj.times = np.linspace(0, 2, n)
    obj.dt = obj.times[1] - obj.times[0]
    obj.joint_positions = rng.standard_normal((n, n_joints)).cumsum(axis=0)
    obj.joint_velocities = rng.standard_normal((n, n_joints))
    obj.joint_torques = rng.standard_normal((n, n_joints))

    obj.compute_coupling_angles = (
        CoordinationMetricsMixin.compute_coupling_angles.__get__(obj)
    )
    obj.compute_coordination_metrics = (
        CoordinationMetricsMixin.compute_coordination_metrics.__get__(obj)
    )
    obj.compute_phase_angle = CoordinationMetricsMixin.compute_phase_angle.__get__(obj)
    obj.compute_continuous_relative_phase = (
        CoordinationMetricsMixin.compute_continuous_relative_phase.__get__(obj)
    )
    obj.compute_correlations = CoordinationMetricsMixin.compute_correlations.__get__(
        obj
    )
    obj.compute_rolling_correlation = (
        CoordinationMetricsMixin.compute_rolling_correlation.__get__(obj)
    )
    return obj


class TestCouplingAnglesPostconditions(unittest.TestCase):
    """Verify ensure() on compute_coupling_angles."""

    def test_angles_in_range(self) -> None:
        obj = _make_mixin()
        angles = obj.compute_coupling_angles(0, 1)
        self.assertGreater(len(angles), 0)
        self.assertTrue(np.all(angles >= 0))
        self.assertTrue(np.all(angles < 360.0))

    def test_out_of_range_returns_empty(self) -> None:
        obj = _make_mixin()
        angles = obj.compute_coupling_angles(0, 99)
        self.assertEqual(len(angles), 0)

    def test_same_joint(self) -> None:
        obj = _make_mixin()
        angles = obj.compute_coupling_angles(0, 0)
        self.assertGreater(len(angles), 0)
        self.assertTrue(np.all(angles >= 0))
        self.assertTrue(np.all(angles < 360.0))


class TestCoordinationMetricsPostconditions(unittest.TestCase):
    """Verify ensure() on compute_coordination_metrics."""

    def test_percentages_sum_to_100(self) -> None:
        obj = _make_mixin()
        result = obj.compute_coordination_metrics(0, 1)
        self.assertIsNotNone(result)
        pct_sum = (
            result.in_phase_pct
            + result.anti_phase_pct
            + result.proximal_leading_pct
            + result.distal_leading_pct
        )
        self.assertAlmostEqual(pct_sum, 100.0, places=4)

    def test_variability_non_negative(self) -> None:
        obj = _make_mixin()
        result = obj.compute_coordination_metrics(0, 1)
        self.assertIsNotNone(result)
        self.assertGreaterEqual(result.coordination_variability, 0)

    def test_mean_angle_in_range(self) -> None:
        obj = _make_mixin()
        result = obj.compute_coordination_metrics(0, 1)
        self.assertIsNotNone(result)
        self.assertGreaterEqual(result.mean_coupling_angle, 0.0)
        self.assertLess(result.mean_coupling_angle, 360.0)

    def test_out_of_range_returns_none(self) -> None:
        obj = _make_mixin()
        result = obj.compute_coordination_metrics(0, 99)
        self.assertIsNone(result)

    def test_all_percentages_non_negative(self) -> None:
        obj = _make_mixin()
        result = obj.compute_coordination_metrics(0, 1)
        self.assertIsNotNone(result)
        self.assertGreaterEqual(result.in_phase_pct, 0)
        self.assertGreaterEqual(result.anti_phase_pct, 0)
        self.assertGreaterEqual(result.proximal_leading_pct, 0)
        self.assertGreaterEqual(result.distal_leading_pct, 0)


class TestRollingCorrelationContracts(unittest.TestCase):
    """Verify require()/ensure() on compute_rolling_correlation."""

    def test_valid_rolling_correlation(self) -> None:
        obj = _make_mixin()
        times, corr = obj.compute_rolling_correlation(0, 1, window_size=10)
        self.assertGreater(len(corr), 0)
        self.assertTrue(np.all(corr >= -1.0 - 1e-6))
        self.assertTrue(np.all(corr <= 1.0 + 1e-6))

    def test_window_size_1_raises(self) -> None:
        obj = _make_mixin()
        with self.assertRaises((ValueError, Exception)):
            obj.compute_rolling_correlation(0, 1, window_size=1)

    def test_window_size_0_raises(self) -> None:
        obj = _make_mixin()
        with self.assertRaises((ValueError, Exception)):
            obj.compute_rolling_correlation(0, 1, window_size=0)

    def test_out_of_range_joint_returns_empty(self) -> None:
        obj = _make_mixin()
        times, corr = obj.compute_rolling_correlation(0, 99, window_size=10)
        self.assertEqual(len(corr), 0)

    def test_perfect_correlation_self(self) -> None:
        obj = _make_mixin()
        times, corr = obj.compute_rolling_correlation(0, 0, window_size=10)
        self.assertGreater(len(corr), 0)
        self.assertTrue(np.all(corr >= 1.0 - 1e-6))


class TestPhaseAnglePostconditions(unittest.TestCase):
    """Verify compute_phase_angle edge cases."""

    def test_valid_phase_angle(self) -> None:
        obj = _make_mixin()
        angles = obj.compute_phase_angle(0)
        self.assertGreater(len(angles), 0)
        self.assertTrue(np.all(np.isfinite(angles)))

    def test_out_of_range_returns_empty(self) -> None:
        obj = _make_mixin()
        angles = obj.compute_phase_angle(99)
        self.assertEqual(len(angles), 0)


class TestCRPPostconditions(unittest.TestCase):
    """Verify compute_continuous_relative_phase output."""

    def test_valid_crp(self) -> None:
        obj = _make_mixin()
        crp = obj.compute_continuous_relative_phase(0, 1)
        self.assertGreater(len(crp), 0)
        self.assertTrue(np.all(np.isfinite(crp)))

    def test_out_of_range_returns_empty(self) -> None:
        obj = _make_mixin()
        crp = obj.compute_continuous_relative_phase(0, 99)
        self.assertEqual(len(crp), 0)


if __name__ == "__main__":
    unittest.main()
