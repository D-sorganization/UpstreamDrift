"""Runtime DbC tests for GRF metrics contracts.

Tests the ensure() postconditions added to compute_grf_metrics:
- cop_path_length >= 0
- cop_max_velocity >= 0
- cop_x_range >= 0, cop_y_range >= 0
- cop_path_length is finite
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

import numpy as np


class TestGRFMetricsPostconditions(unittest.TestCase):
    """Verify ensure() postconditions on compute_grf_metrics."""

    def _make_mixin(
        self,
        cop: np.ndarray | None = None,
        forces: np.ndarray | None = None,
        dt: float = 0.01,
    ) -> object:
        from src.shared.python.analysis.grf_metrics import GRFMetricsMixin

        obj = MagicMock(spec=GRFMetricsMixin)
        obj.cop_position = cop
        obj.ground_forces = forces
        obj.dt = dt
        # Bind the real method
        obj.compute_grf_metrics = GRFMetricsMixin.compute_grf_metrics.__get__(obj)
        return obj

    def test_returns_none_for_no_cop(self) -> None:
        obj = self._make_mixin(cop=None)
        result = obj.compute_grf_metrics()
        self.assertIsNone(result)

    def test_returns_none_for_empty_cop(self) -> None:
        obj = self._make_mixin(cop=np.empty((0, 2)))
        result = obj.compute_grf_metrics()
        self.assertIsNone(result)

    def test_path_length_non_negative_2d(self) -> None:
        rng = np.random.default_rng(42)
        cop = rng.standard_normal((100, 2)).cumsum(axis=0)
        obj = self._make_mixin(cop=cop)
        result = obj.compute_grf_metrics()
        self.assertIsNotNone(result)
        self.assertGreaterEqual(result.cop_path_length, 0)

    def test_path_length_non_negative_3d(self) -> None:
        rng = np.random.default_rng(42)
        cop = rng.standard_normal((100, 3)).cumsum(axis=0)
        obj = self._make_mixin(cop=cop)
        result = obj.compute_grf_metrics()
        self.assertIsNotNone(result)
        self.assertGreaterEqual(result.cop_path_length, 0)

    def test_max_velocity_non_negative(self) -> None:
        rng = np.random.default_rng(42)
        cop = rng.standard_normal((100, 2)).cumsum(axis=0)
        obj = self._make_mixin(cop=cop, dt=0.01)
        result = obj.compute_grf_metrics()
        self.assertGreaterEqual(result.cop_max_velocity, 0)

    def test_max_velocity_zero_for_zero_dt(self) -> None:
        rng = np.random.default_rng(42)
        cop = rng.standard_normal((50, 2)).cumsum(axis=0)
        obj = self._make_mixin(cop=cop, dt=0.0)
        result = obj.compute_grf_metrics()
        self.assertEqual(result.cop_max_velocity, 0.0)

    def test_x_range_non_negative(self) -> None:
        cop = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        obj = self._make_mixin(cop=cop)
        result = obj.compute_grf_metrics()
        self.assertGreaterEqual(result.cop_x_range, 0)

    def test_y_range_non_negative(self) -> None:
        cop = np.array([[0.0, 0.0], [0.0, 1.0], [0.0, 2.0]])
        obj = self._make_mixin(cop=cop)
        result = obj.compute_grf_metrics()
        self.assertGreaterEqual(result.cop_y_range, 0)

    def test_path_length_is_finite(self) -> None:
        cop = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]])
        obj = self._make_mixin(cop=cop)
        result = obj.compute_grf_metrics()
        self.assertTrue(np.isfinite(result.cop_path_length))

    def test_stationary_cop_zero_path(self) -> None:
        cop = np.ones((50, 2))
        obj = self._make_mixin(cop=cop)
        result = obj.compute_grf_metrics()
        self.assertAlmostEqual(result.cop_path_length, 0.0)

    def test_force_metrics_with_ground_forces(self) -> None:
        cop = np.ones((50, 2))
        forces = np.zeros((50, 3))
        forces[:, 2] = 500.0  # vertical
        forces[:, 0] = 10.0  # shear x
        obj = self._make_mixin(cop=cop, forces=forces)
        result = obj.compute_grf_metrics()
        self.assertIsNotNone(result.peak_vertical_force)
        self.assertAlmostEqual(result.peak_vertical_force, 500.0)
        self.assertIsNotNone(result.peak_shear_force)
        self.assertGreater(result.peak_shear_force, 0)

    def test_no_force_metrics_without_ground_forces(self) -> None:
        cop = np.ones((50, 2))
        obj = self._make_mixin(cop=cop, forces=None)
        result = obj.compute_grf_metrics()
        self.assertIsNone(result.peak_vertical_force)
        self.assertIsNone(result.peak_shear_force)


if __name__ == "__main__":
    unittest.main()
