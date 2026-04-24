"""Tests for src.shared.python.biomechanics.swing_plane_analysis (Issues #1949, #1744)."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.biomechanics.swing_plane_analysis import (
    SwingPlaneAnalyzer,
    SwingPlaneMetrics,
)


def _make_planar_points(n: int = 50) -> np.ndarray:
    """Create points lying in the XZ plane (y=0)."""
    t = np.linspace(0, 1, n)
    pts = np.zeros((n, 3))
    pts[:, 0] = np.sin(np.pi * t)  # X component
    pts[:, 2] = np.cos(np.pi * t)  # Z component
    # Add tiny noise to avoid perfect degeneracy
    rng = np.random.default_rng(42)
    pts[:, 1] = rng.standard_normal(n) * 1e-4
    return pts


class TestSwingPlaneAnalyzer:
    def setup_method(self) -> None:
        self.analyzer = SwingPlaneAnalyzer()

    def test_fit_plane_returns_centroid_and_normal(self) -> None:
        pts = _make_planar_points()
        centroid, normal = self.analyzer.fit_plane(pts)
        assert centroid.shape == (3,)
        assert normal.shape == (3,)

    def test_normal_is_unit_length(self) -> None:
        pts = _make_planar_points()
        _, normal = self.analyzer.fit_plane(pts)
        assert np.linalg.norm(normal) == pytest.approx(1.0, abs=1e-6)

    def test_fewer_than_3_points_raises(self) -> None:
        pts = np.array([[0, 0, 0], [1, 1, 1]])
        with pytest.raises((ValueError, TypeError, AssertionError)):
            self.analyzer.fit_plane(pts)

    def test_planar_points_have_normal_perpendicular_to_plane(self) -> None:
        # XZ plane → normal should be close to [0, 1, 0]
        pts = _make_planar_points(n=100)
        _, normal = self.analyzer.fit_plane(pts)
        # Y component of normal should be close to 1 (or -1)
        assert abs(abs(normal[1]) - 1.0) < 0.01

    def test_deviation_shape(self) -> None:
        pts = _make_planar_points()
        centroid, normal = self.analyzer.fit_plane(pts)
        deviations = self.analyzer.calculate_deviation(pts, centroid, normal)
        assert deviations.shape == (len(pts),)

    def test_planar_points_near_zero_deviation(self) -> None:
        pts = _make_planar_points(n=100)
        centroid, normal = self.analyzer.fit_plane(pts)
        deviations = self.analyzer.calculate_deviation(pts, centroid, normal)
        assert np.max(np.abs(deviations)) < 0.001  # small noise

    def test_analyze_returns_metrics(self) -> None:
        pts = _make_planar_points()
        metrics = self.analyzer.analyze(pts)
        assert isinstance(metrics, SwingPlaneMetrics)

    def test_analyze_rmse_non_negative(self) -> None:
        pts = _make_planar_points()
        metrics = self.analyzer.analyze(pts)
        assert metrics.rmse >= 0.0

    def test_analyze_max_deviation_non_negative(self) -> None:
        pts = _make_planar_points()
        metrics = self.analyzer.analyze(pts)
        assert metrics.max_deviation >= 0.0

    def test_analyze_steepness_in_range(self) -> None:
        pts = _make_planar_points()
        metrics = self.analyzer.analyze(pts)
        # Steepness is angle with horizontal, in degrees
        assert 0.0 <= metrics.steepness_deg <= 90.0

    def test_normal_vector_unit_length_after_analyze(self) -> None:
        pts = _make_planar_points()
        metrics = self.analyzer.analyze(pts)
        assert np.linalg.norm(metrics.normal_vector) == pytest.approx(1.0, abs=1e-6)

    def test_all_metrics_finite(self) -> None:
        pts = _make_planar_points()
        metrics = self.analyzer.analyze(pts)
        assert np.isfinite(metrics.steepness_deg)
        assert np.isfinite(metrics.direction_deg)
        assert np.isfinite(metrics.rmse)
        assert np.isfinite(metrics.max_deviation)
