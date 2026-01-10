import numpy as np
import pytest

from shared.python.swing_plane_analysis import SwingPlaneAnalyzer


class TestSwingPlaneAnalysis:
    def test_fit_plane_perfect(self):
        """Test fitting a plane to points that lie perfectly on it."""
        # Plane z = 0 (XY plane)
        points = np.array(
            [[1.0, 1.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 0.0]]
        )

        analyzer = SwingPlaneAnalyzer()
        centroid, normal = analyzer.fit_plane(points)

        # Centroid should be on z=0
        assert centroid[2] == 0.0

        # Normal should be [0, 0, 1] (or -1)
        assert abs(normal[0]) < 1e-10
        assert abs(normal[1]) < 1e-10
        assert abs(abs(normal[2]) - 1.0) < 1e-10

    def test_fit_plane_inclined(self):
        """Test fitting to an inclined plane."""
        # Points: x=i, y=i, z=-x.
        # Vector on plane 1: (1, 1, -1) [from i=0 to i=1]
        # This defines a LINE, not a plane!
        # Because y = x, z = -x.
        # All points lie on the line (1, 1, -1) passing through origin.
        # SVD on a line in 3D: small singular values for TWO dimensions.
        # The normal is not unique. It can be any vector orthogonal to (1, 1, -1).
        # We need points that span a plane.

        points = []
        for i in range(10):
            x = float(i)
            y = float(i)  # y = x
            z = -x  # z = -x
            points.append([x, y, z])

        # Add a point off the line to define a plane
        # Current line vector v = (1, 1, -1)
        # Add point (1, 0, -1).
        # Vector u = (1, 0, -1)
        # Plane normal n = v x u
        # (1, 1, -1) x (1, 0, -1)
        # x: 1*-1 - -1*0 = -1
        # y: -1*1 - 1*-1 = 0
        # z: 1*0 - 1*1 = -1
        # Normal ~ (-1, 0, -1) -> (1, 0, 1)

        points.append([1.0, 0.0, -1.0])
        points = np.array(points)

        analyzer = SwingPlaneAnalyzer()
        centroid, normal = analyzer.fit_plane(points)

        # Expected normal (1, 0, 1) normalized
        expected = np.array([1.0, 0.0, 1.0])
        expected = expected / np.linalg.norm(expected)

        # Dot product should be 1 or -1
        dot = np.abs(np.dot(normal, expected))
        np.testing.assert_allclose(dot, 1.0, atol=1e-10)

    def test_calculate_deviation(self):
        """Test deviation calculation."""
        points = np.array(
            [
                [0.0, 0.0, 1.0],  # 1 unit above
                [0.0, 0.0, -1.0],  # 1 unit below
                [0.0, 0.0, 0.0],  # on plane
            ]
        )
        centroid = np.array([0.0, 0.0, 0.0])
        normal = np.array([0.0, 0.0, 1.0])

        analyzer = SwingPlaneAnalyzer()
        devs = analyzer.calculate_deviation(points, centroid, normal)

        expected = np.array([1.0, -1.0, 0.0])
        np.testing.assert_allclose(devs, expected, atol=1e-10)

    def test_full_analysis(self):
        """Test full analysis flow."""
        points = np.random.rand(10, 3)
        # Flatten to Z=0 to ensure perfect fit
        points[:, 2] = 0.0

        analyzer = SwingPlaneAnalyzer()
        metrics = analyzer.analyze(points)

        assert metrics.rmse < 1e-10
        assert (
            abs(metrics.steepness_deg) < 1e-5
        )  # Normal is vertical (0 deg to vertical?)
        # Wait, analyze ensures normal Z > 0.
        # If normal is [0, 0, 1], angle with [0, 0, 1] is 0.

    def test_insufficient_points(self):
        """Test error handling for too few points."""
        points = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        analyzer = SwingPlaneAnalyzer()
        with pytest.raises(ValueError, match="At least 3 points"):
            analyzer.fit_plane(points)
