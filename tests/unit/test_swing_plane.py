"""Tests for the Swing Plane Analyzer."""

import numpy as np
import pytest

from src.shared.python.swing_plane_analysis import SwingPlaneAnalyzer


class TestSwingPlaneAnalyzer:
    """Test suite for SwingPlaneAnalyzer."""

    @pytest.fixture
    def analyzer(self) -> SwingPlaneAnalyzer:
        """Return a SwingPlaneAnalyzer instance."""
        return SwingPlaneAnalyzer()

    def test_fit_plane_exact_horizontal(self, analyzer: SwingPlaneAnalyzer) -> None:
        """Test fitting a plane to points on a horizontal plane (Z=constant)."""
        # Create points on z=5 plane
        points = np.array(
            [
                [0, 0, 5],
                [1, 0, 5],
                [0, 1, 5],
                [1, 1, 5],
            ],
            dtype=np.float64,
        )

        centroid, normal = analyzer.fit_plane(points)

        # Centroid should be mean
        expected_centroid = np.mean(points, axis=0)
        np.testing.assert_allclose(centroid, expected_centroid)

        # Normal should be [0, 0, 1] or [0, 0, -1]
        assert abs(normal[0]) < 1e-10
        assert abs(normal[1]) < 1e-10
        assert abs(abs(normal[2]) - 1.0) < 1e-10

    def test_fit_plane_exact_vertical(self, analyzer: SwingPlaneAnalyzer) -> None:
        """Test fitting a plane to points on a vertical plane (X=constant)."""
        # Create points on x=2 plane
        points = np.array(
            [
                [2, 0, 0],
                [2, 1, 0],
                [2, 0, 1],
                [2, 1, 1],
            ],
            dtype=np.float64,
        )

        _, normal = analyzer.fit_plane(points)

        # Normal should be [1, 0, 0] or [-1, 0, 0]
        assert abs(abs(normal[0]) - 1.0) < 1e-10
        assert abs(normal[1]) < 1e-10
        assert abs(normal[2]) < 1e-10

    def test_analyze_metrics(self, analyzer: SwingPlaneAnalyzer) -> None:
        """Test calculation of metrics."""
        # Create points on a known inclined plane
        # 45 degree slope around Y axis.
        # Normal should be [-1, 0, 1] normalized -> [-0.707, 0, 0.707]
        points = np.array(
            [
                [0, 0, 0],
                [1, 0, 1],
                [0, 1, 0],
                [1, 1, 1],
            ],
            dtype=np.float64,
        )

        metrics = analyzer.analyze(points)

        # Steepness should be 45 degrees
        assert abs(metrics.steepness_deg - 45.0) < 1e-5

        # RMSE should be effectively zero (perfect plane)
        assert metrics.rmse < 1e-10
        assert metrics.max_deviation < 1e-10

    def test_deviation_calculation(self, analyzer: SwingPlaneAnalyzer) -> None:
        """Test deviation calculation."""
        # Plane Z=0
        centroid = np.array([0, 0, 0])
        normal = np.array([0, 0, 1])

        # Test point at Z=1 (dist 1) and Z=-2 (dist -2)
        test_points = np.array(
            [
                [0.5, 0.5, 1],
                [0.5, 0.5, -2],
            ],
            dtype=np.float64,
        )

        deviations = analyzer.calculate_deviation(test_points, centroid, normal)
        np.testing.assert_allclose(deviations, [1.0, -2.0])

    def test_insufficient_points(self, analyzer: SwingPlaneAnalyzer) -> None:
        """Test error handling for insufficient points."""
        points = np.array([[0, 0, 0], [1, 1, 1]])
        with pytest.raises(ValueError, match="At least 3 points"):
            analyzer.fit_plane(points)
