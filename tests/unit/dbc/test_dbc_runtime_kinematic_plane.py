"""Runtime DbC tests for kinematic_sequence and swing_plane_analysis contracts.

Tests contracts added to:
- SegmentTimingAnalyzer.analyze: require() non-empty times,
  ensure() consistency in [0,1], peak velocities non-negative, timing gaps finite
- SwingPlaneAnalyzer: require() >= 3 points for fit_plane,
  ensure() unit normal, rmse >= 0, max_deviation >= 0, steepness in [0,180]
"""

from __future__ import annotations

import unittest

import numpy as np

# ==================== Kinematic Sequence Tests ====================


class TestSegmentTimingAnalyzerPreconditions(unittest.TestCase):
    """Verify require() preconditions on SegmentTimingAnalyzer.analyze."""

    def test_empty_times_raises(self) -> None:
        from src.shared.python.biomechanics.kinematic_sequence import (
            SegmentTimingAnalyzer,
        )

        analyzer = SegmentTimingAnalyzer(expected_order=["a", "b"])
        with self.assertRaises((ValueError, Exception)):
            analyzer.analyze(
                segment_velocities={"a": np.ones(10), "b": np.ones(10)},
                times=np.array([]),
            )

    def test_valid_analysis(self) -> None:
        from src.shared.python.biomechanics.kinematic_sequence import (
            SegmentTimingAnalyzer,
        )

        times = np.linspace(0, 1, 100)
        rng = np.random.default_rng(42)
        analyzer = SegmentTimingAnalyzer(expected_order=["hip", "shoulder", "hand"])
        result = analyzer.analyze(
            segment_velocities={
                "hip": rng.standard_normal(100),
                "shoulder": rng.standard_normal(100),
                "hand": rng.standard_normal(100),
            },
            times=times,
        )
        self.assertIsNotNone(result)


class TestSegmentTimingAnalyzerPostconditions(unittest.TestCase):
    """Verify ensure() postconditions on SegmentTimingAnalyzer.analyze."""

    def _make_result(self) -> object:
        from src.shared.python.biomechanics.kinematic_sequence import (
            SegmentTimingAnalyzer,
        )

        rng = np.random.default_rng(42)
        times = np.linspace(0, 1, 200)

        # Create sequential peaks to ensure valid ordering
        hip_vel = np.zeros(200)
        hip_vel[40] = 10.0  # Peak at t=0.2
        shoulder_vel = np.zeros(200)
        shoulder_vel[80] = 15.0  # Peak at t=0.4
        hand_vel = np.zeros(200)
        hand_vel[120] = 20.0  # Peak at t=0.6

        analyzer = SegmentTimingAnalyzer(
            expected_order=["hip", "shoulder", "hand"],
        )
        return analyzer.analyze(
            segment_velocities={
                "hip": hip_vel + rng.standard_normal(200) * 0.01,
                "shoulder": shoulder_vel + rng.standard_normal(200) * 0.01,
                "hand": hand_vel + rng.standard_normal(200) * 0.01,
            },
            times=times,
        )

    def test_consistency_in_range(self) -> None:
        result = self._make_result()
        self.assertGreaterEqual(result.sequence_consistency, 0.0)
        self.assertLessEqual(result.sequence_consistency, 1.0)

    def test_peak_velocities_non_negative(self) -> None:
        result = self._make_result()
        for peak in result.peaks:
            self.assertGreaterEqual(peak.peak_velocity, 0.0)

    def test_timing_gaps_finite(self) -> None:
        result = self._make_result()
        for gap_name, gap_val in result.timing_gaps.items():
            self.assertTrue(
                np.isfinite(gap_val),
                f"Timing gap '{gap_name}' is not finite: {gap_val}",
            )

    def test_perfect_sequence_consistency(self) -> None:
        result = self._make_result()
        # With clear sequential peaks, consistency should be 1.0
        self.assertAlmostEqual(result.sequence_consistency, 1.0)
        self.assertTrue(result.is_valid_sequence)

    def test_no_expected_order_zero_consistency(self) -> None:
        from src.shared.python.biomechanics.kinematic_sequence import (
            SegmentTimingAnalyzer,
        )

        analyzer = SegmentTimingAnalyzer(expected_order=None)
        result = analyzer.analyze(
            segment_velocities={"a": np.ones(50)},
            times=np.linspace(0, 1, 50),
        )
        self.assertEqual(result.sequence_consistency, 0.0)

    def test_single_segment(self) -> None:
        from src.shared.python.biomechanics.kinematic_sequence import (
            SegmentTimingAnalyzer,
        )

        analyzer = SegmentTimingAnalyzer(expected_order=["hip"])
        result = analyzer.analyze(
            segment_velocities={"hip": np.ones(50)},
            times=np.linspace(0, 1, 50),
        )
        self.assertEqual(len(result.peaks), 1)
        self.assertGreaterEqual(result.peaks[0].peak_velocity, 0)


# ==================== Swing Plane Analysis Tests ====================


class TestSwingPlaneAnalyzerPreconditions(unittest.TestCase):
    """Verify require() preconditions on SwingPlaneAnalyzer."""

    def test_fewer_than_3_points_raises(self) -> None:
        from src.shared.python.biomechanics.swing_plane_analysis import (
            SwingPlaneAnalyzer,
        )

        analyzer = SwingPlaneAnalyzer()
        with self.assertRaises((ValueError, Exception)):
            analyzer.fit_plane(np.array([[0, 0, 0], [1, 1, 1]]))

    def test_one_point_raises(self) -> None:
        from src.shared.python.biomechanics.swing_plane_analysis import (
            SwingPlaneAnalyzer,
        )

        analyzer = SwingPlaneAnalyzer()
        with self.assertRaises((ValueError, Exception)):
            analyzer.fit_plane(np.array([[0, 0, 0]]))

    def test_valid_fit(self) -> None:
        from src.shared.python.biomechanics.swing_plane_analysis import (
            SwingPlaneAnalyzer,
        )

        analyzer = SwingPlaneAnalyzer()
        points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]])
        centroid, normal = analyzer.fit_plane(points)
        self.assertAlmostEqual(np.linalg.norm(normal), 1.0, places=5)


class TestSwingPlanePostconditions(unittest.TestCase):
    """Verify ensure() postconditions on SwingPlaneAnalyzer.analyze."""

    def _make_trajectory(self, n: int = 50) -> np.ndarray:
        """Create a noisy trajectory near a plane."""
        rng = np.random.default_rng(42)
        t = np.linspace(0, 2 * np.pi, n)
        x = np.cos(t)
        y = np.sin(t)
        z = rng.standard_normal(n) * 0.05  # Small noise off-plane
        return np.column_stack([x, y, z])

    def test_rmse_non_negative(self) -> None:
        from src.shared.python.biomechanics.swing_plane_analysis import (
            SwingPlaneAnalyzer,
        )

        analyzer = SwingPlaneAnalyzer()
        result = analyzer.analyze(self._make_trajectory())
        self.assertGreaterEqual(result.rmse, 0)

    def test_max_deviation_non_negative(self) -> None:
        from src.shared.python.biomechanics.swing_plane_analysis import (
            SwingPlaneAnalyzer,
        )

        analyzer = SwingPlaneAnalyzer()
        result = analyzer.analyze(self._make_trajectory())
        self.assertGreaterEqual(result.max_deviation, 0)

    def test_steepness_in_range(self) -> None:
        from src.shared.python.biomechanics.swing_plane_analysis import (
            SwingPlaneAnalyzer,
        )

        analyzer = SwingPlaneAnalyzer()
        result = analyzer.analyze(self._make_trajectory())
        self.assertGreaterEqual(result.steepness_deg, 0.0)
        self.assertLessEqual(result.steepness_deg, 180.0)

    def test_normal_is_unit_vector(self) -> None:
        from src.shared.python.biomechanics.swing_plane_analysis import (
            SwingPlaneAnalyzer,
        )

        analyzer = SwingPlaneAnalyzer()
        result = analyzer.analyze(self._make_trajectory())
        norm = np.linalg.norm(result.normal_vector)
        self.assertAlmostEqual(norm, 1.0, places=5)

    def test_flat_plane_steepness_near_zero(self) -> None:
        from src.shared.python.biomechanics.swing_plane_analysis import (
            SwingPlaneAnalyzer,
        )

        analyzer = SwingPlaneAnalyzer()
        # Points exactly in the XY plane
        rng = np.random.default_rng(42)
        points = np.column_stack(
            [
                rng.standard_normal(20),
                rng.standard_normal(20),
                np.zeros(20),
            ]
        )
        result = analyzer.analyze(points)
        self.assertAlmostEqual(result.steepness_deg, 0.0, places=2)
        self.assertAlmostEqual(result.rmse, 0.0, places=5)

    def test_vertical_plane(self) -> None:
        from src.shared.python.biomechanics.swing_plane_analysis import (
            SwingPlaneAnalyzer,
        )

        analyzer = SwingPlaneAnalyzer()
        # Points in the XZ plane (vertical)
        rng = np.random.default_rng(42)
        points = np.column_stack(
            [
                rng.standard_normal(20),
                np.zeros(20),
                rng.standard_normal(20),
            ]
        )
        result = analyzer.analyze(points)
        self.assertAlmostEqual(result.steepness_deg, 90.0, places=1)


if __name__ == "__main__":
    unittest.main()
