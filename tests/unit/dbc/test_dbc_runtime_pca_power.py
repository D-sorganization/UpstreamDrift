"""Runtime DbC tests for PCA analysis and power/work contracts.

Tests contracts added to:
- PCA: variance ratios sum <= 1, non-negative variance, n_components >= 1
- Kinematic sequence: efficiency in [0,1], peak velocities >= 0
- Power/Work: positive_work >= 0, negative_work <= 0, finite values,
  duration >= 0, path length >= 0
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

import numpy as np


def _make_pca_mixin(n: int = 100, n_joints: int = 3) -> object:
    """Create a mock PCAAnalysisMixin."""
    from src.shared.python.analysis.pca_analysis import PCAAnalysisMixin

    rng = np.random.default_rng(42)
    obj = MagicMock(spec=PCAAnalysisMixin)
    obj.times = np.linspace(0, 1, n)
    obj.joint_positions = rng.standard_normal((n, n_joints))
    obj.joint_velocities = rng.standard_normal((n, n_joints))

    obj.compute_principal_component_analysis = (
        PCAAnalysisMixin.compute_principal_component_analysis.__get__(obj)
    )
    obj.compute_principal_movements = (
        PCAAnalysisMixin.compute_principal_movements.__get__(obj)
    )
    obj.analyze_kinematic_sequence = (
        PCAAnalysisMixin.analyze_kinematic_sequence.__get__(obj)
    )
    return obj


def _make_power_mixin(n: int = 100, n_joints: int = 3) -> object:
    """Create a mock PowerWorkMetricsMixin."""
    from src.shared.python.analysis.power_work_metrics import PowerWorkMetricsMixin

    rng = np.random.default_rng(42)
    obj = MagicMock(spec=PowerWorkMetricsMixin)
    obj.times = np.linspace(0, 1, n)
    obj.dt = 0.01
    obj.joint_positions = rng.standard_normal((n, n_joints))
    obj.joint_velocities = rng.standard_normal((n, n_joints))
    obj.joint_torques = rng.standard_normal((n, n_joints))
    obj.ground_forces = None
    obj._work_metrics_cache = {}

    obj.compute_work_metrics = PowerWorkMetricsMixin.compute_work_metrics.__get__(obj)
    obj.compute_joint_power_metrics = (
        PowerWorkMetricsMixin.compute_joint_power_metrics.__get__(obj)
    )
    obj.compute_impulse_metrics = PowerWorkMetricsMixin.compute_impulse_metrics.__get__(
        obj
    )
    obj.compute_phase_space_path_length = (
        PowerWorkMetricsMixin.compute_phase_space_path_length.__get__(obj)
    )
    return obj


# ==================== PCA Tests ====================


class TestPCAPreconditions(unittest.TestCase):
    """Verify require() preconditions on PCA."""

    def test_valid_pca_returns_result(self) -> None:
        obj = _make_pca_mixin()
        result = obj.compute_principal_component_analysis(n_components=2)
        self.assertIsNotNone(result)

    def test_n_components_zero_raises(self) -> None:
        obj = _make_pca_mixin()
        with self.assertRaises((ValueError, Exception)):
            obj.compute_principal_component_analysis(n_components=0)

    def test_n_components_negative_raises(self) -> None:
        obj = _make_pca_mixin()
        with self.assertRaises((ValueError, Exception)):
            obj.compute_principal_component_analysis(n_components=-1)

    def test_n_components_none_valid(self) -> None:
        obj = _make_pca_mixin()
        result = obj.compute_principal_component_analysis(n_components=None)
        self.assertIsNotNone(result)


class TestPCAPostconditions(unittest.TestCase):
    """Verify ensure() postconditions on PCA results."""

    def test_variance_non_negative(self) -> None:
        obj = _make_pca_mixin()
        result = obj.compute_principal_component_analysis()
        self.assertTrue(np.all(result.explained_variance >= 0))

    def test_variance_ratio_sum_leq_one(self) -> None:
        obj = _make_pca_mixin()
        result = obj.compute_principal_component_analysis()
        ratio_sum = np.sum(result.explained_variance_ratio)
        self.assertLessEqual(ratio_sum, 1.0 + 1e-6)

    def test_variance_finite(self) -> None:
        obj = _make_pca_mixin()
        result = obj.compute_principal_component_analysis()
        self.assertTrue(np.all(np.isfinite(result.explained_variance)))

    def test_subset_components_variance_ratio_leq_one(self) -> None:
        obj = _make_pca_mixin()
        result = obj.compute_principal_component_analysis(n_components=2)
        ratio_sum = np.sum(result.explained_variance_ratio)
        self.assertLessEqual(ratio_sum, 1.0 + 1e-6)


class TestKinematicSequencePostconditions(unittest.TestCase):
    """Verify ensure() postconditions on analyze_kinematic_sequence."""

    def test_efficiency_in_range(self) -> None:
        obj = _make_pca_mixin()
        segments = {"hip": 0, "shoulder": 1, "hand": 2}
        sequence, score = obj.analyze_kinematic_sequence(segments)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_peak_velocities_non_negative(self) -> None:
        obj = _make_pca_mixin()
        segments = {"hip": 0, "shoulder": 1, "hand": 2}
        sequence, _ = obj.analyze_kinematic_sequence(segments)
        for info in sequence:
            self.assertGreaterEqual(info.peak_velocity, 0.0)

    def test_empty_segments_zero_efficiency(self) -> None:
        obj = _make_pca_mixin()
        sequence, score = obj.analyze_kinematic_sequence({})
        self.assertEqual(score, 0.0)
        self.assertEqual(len(sequence), 0)

    def test_single_segment(self) -> None:
        obj = _make_pca_mixin()
        sequence, score = obj.analyze_kinematic_sequence({"hip": 0})
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
        self.assertEqual(len(sequence), 1)


# ==================== Power/Work Tests ====================


class TestWorkMetricsPostconditions(unittest.TestCase):
    """Verify ensure() postconditions on compute_work_metrics."""

    def test_positive_work_non_negative(self) -> None:
        obj = _make_power_mixin()
        result = obj.compute_work_metrics(0)
        self.assertIsNotNone(result)
        self.assertGreaterEqual(result["positive_work"], 0)

    def test_negative_work_non_positive(self) -> None:
        obj = _make_power_mixin()
        result = obj.compute_work_metrics(0)
        self.assertLessEqual(result["negative_work"], 0)

    def test_net_work_finite(self) -> None:
        obj = _make_power_mixin()
        result = obj.compute_work_metrics(0)
        self.assertTrue(np.isfinite(result["net_work"]))

    def test_cached_result_same(self) -> None:
        obj = _make_power_mixin()
        r1 = obj.compute_work_metrics(0)
        r2 = obj.compute_work_metrics(0)
        self.assertEqual(r1, r2)

    def test_out_of_range_joint_returns_none(self) -> None:
        obj = _make_power_mixin()
        result = obj.compute_work_metrics(99)
        self.assertIsNone(result)


class TestJointPowerMetricsPostconditions(unittest.TestCase):
    """Verify ensure() postconditions on compute_joint_power_metrics."""

    def test_peak_generation_non_negative(self) -> None:
        obj = _make_power_mixin()
        result = obj.compute_joint_power_metrics(0)
        self.assertIsNotNone(result)
        self.assertGreaterEqual(result.peak_generation, 0)

    def test_peak_absorption_non_positive(self) -> None:
        obj = _make_power_mixin()
        result = obj.compute_joint_power_metrics(0)
        self.assertLessEqual(result.peak_absorption, 0)

    def test_durations_non_negative(self) -> None:
        obj = _make_power_mixin()
        result = obj.compute_joint_power_metrics(0)
        self.assertGreaterEqual(result.generation_duration, 0)
        self.assertGreaterEqual(result.absorption_duration, 0)

    def test_net_work_finite(self) -> None:
        obj = _make_power_mixin()
        result = obj.compute_joint_power_metrics(0)
        self.assertTrue(np.isfinite(result.net_work))


class TestPhaseSpacePathLengthPostconditions(unittest.TestCase):
    """Verify ensure() postconditions on compute_phase_space_path_length."""

    def test_path_length_non_negative(self) -> None:
        obj = _make_power_mixin()
        result = obj.compute_phase_space_path_length(0)
        self.assertGreaterEqual(result, 0.0)

    def test_stationary_zero_path(self) -> None:
        obj = _make_power_mixin()
        obj.joint_positions = np.ones((100, 3))
        obj.joint_velocities = np.ones((100, 3))
        result = obj.compute_phase_space_path_length(0)
        self.assertAlmostEqual(result, 0.0)

    def test_out_of_range_returns_zero(self) -> None:
        obj = _make_power_mixin()
        result = obj.compute_phase_space_path_length(99)
        self.assertEqual(result, 0.0)


if __name__ == "__main__":
    unittest.main()
