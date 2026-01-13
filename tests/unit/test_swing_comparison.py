"""Unit tests for shared/python/swing_comparison.py."""

import unittest
from unittest.mock import patch

import numpy as np

from shared.python.statistical_analysis import StatisticalAnalyzer
from shared.python.swing_comparison import (
    ComparisonMetric,
    DTWResult,
    SwingComparator,
)


class TestSwingComparison(unittest.TestCase):
    """Test suite for swing comparison logic."""

    def setUp(self):
        """Set up test data."""
        # Create mock data for reference
        self.ref_data = {
            "times": np.linspace(0, 1, 100),
            "joint_positions": np.random.rand(100, 3),
            "joint_velocities": np.random.rand(100, 3),
            "joint_torques": np.random.rand(100, 3),
        }
        # Create mock data for student (slightly different)
        self.stu_data = {
            "times": np.linspace(0, 1, 100),
            "joint_positions": self.ref_data["joint_positions"] + 0.1,
            "joint_velocities": self.ref_data["joint_velocities"] * 1.1,
            "joint_torques": self.ref_data["joint_torques"],
        }

    def test_init_with_dict(self):
        """Test initialization with dictionaries."""
        comparator = SwingComparator(self.ref_data, self.stu_data)
        self.assertIsInstance(comparator.ref, StatisticalAnalyzer)
        self.assertIsInstance(comparator.student, StatisticalAnalyzer)

    def test_init_with_analyzer(self):
        """Test initialization with StatisticalAnalyzer objects."""
        ref_analyzer = StatisticalAnalyzer(**self.ref_data)
        stu_analyzer = StatisticalAnalyzer(**self.stu_data)
        comparator = SwingComparator(ref_analyzer, stu_analyzer)
        self.assertIs(comparator.ref, ref_analyzer)
        self.assertIs(comparator.student, stu_analyzer)

    @patch.object(StatisticalAnalyzer, "compute_tempo")
    def test_compare_tempo(self, mock_tempo):
        """Test tempo comparison."""
        # Mock tempo returns (start, top, ratio, ...)
        mock_tempo.side_effect = [
            (0.0, 0.7, 3.0),  # ref: 3.0 ratio
            (0.0, 0.8, 3.3),  # stu: 3.3 ratio
        ]

        comparator = SwingComparator(self.ref_data, self.stu_data)
        metric = comparator.compare_tempo()

        self.assertIsNotNone(metric)
        if metric:
            self.assertEqual(metric.name, "Tempo Ratio")
            self.assertEqual(metric.reference_value, 3.0)
            self.assertEqual(metric.student_value, 3.3)
            self.assertAlmostEqual(metric.difference, 0.3)
            self.assertAlmostEqual(metric.percent_diff, 10.0)

    @patch("shared.python.signal_processing.compute_dtw_path")
    def test_compute_kinematic_similarity(self, mock_dtw):
        """Test kinematic similarity with mocked DTW."""
        # Mock DTW result (dist, path)
        mock_dtw.return_value = (10.0, [(i, i) for i in range(100)])

        comparator = SwingComparator(self.ref_data, self.stu_data)
        res = comparator.compute_kinematic_similarity(0, feature="velocity")

        self.assertIsInstance(res, DTWResult)
        if res:
            self.assertEqual(res.distance, 10.0)
            self.assertEqual(len(res.path), 100)
            self.assertAlmostEqual(res.normalized_distance, 0.1)

    def test_compare_peak_speeds(self):
        """Test peak speed comparison."""
        comparator = SwingComparator(self.ref_data, self.stu_data)
        segment_indices = {"Joint1": 0, "Joint2": 1}

        metrics = comparator.compare_peak_speeds(segment_indices)

        self.assertIn("Joint1", metrics)
        self.assertIn("Joint2", metrics)
        self.assertIsInstance(metrics["Joint1"], ComparisonMetric)

        # Check values logic
        # stu velocity was ref * 1.1, so should be higher
        ref_peak = np.max(np.abs(self.ref_data["joint_velocities"][:, 0]))
        stu_peak = np.max(np.abs(self.stu_data["joint_velocities"][:, 0]))
        self.assertAlmostEqual(metrics["Joint1"].student_value, stu_peak)
        self.assertAlmostEqual(metrics["Joint1"].reference_value, ref_peak)

    def test_generate_comparison_report(self):
        """Test full report generation."""
        comparator = SwingComparator(self.ref_data, self.stu_data)
        segment_indices = {"Joint1": 0}

        with (
            patch.object(comparator, "compare_tempo") as mock_tempo,
            patch.object(comparator, "compute_kinematic_similarity") as mock_sim,
        ):
            mock_tempo.return_value = ComparisonMetric(
                "Tempo", 3.0, 3.0, 0.0, 0.0, 100.0
            )
            mock_sim.return_value = DTWResult(0.0, [], 0.0, 100.0)

            report = comparator.generate_comparison_report(segment_indices)

            self.assertIn("metrics", report)
            self.assertIn("tempo", report["metrics"])
            self.assertIn("Joint1", report["metrics"])
            self.assertIn("sequence_similarity", report)
            self.assertIn("Joint1", report["sequence_similarity"])


if __name__ == "__main__":
    unittest.main()
