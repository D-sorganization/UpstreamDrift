"""Tests for src.shared.python.pendulum_simulator.segment_geometry (Issues #1949, #1744)."""

from __future__ import annotations

import numpy as np
import pytest
from src.shared.python.pendulum_simulator.segment_geometry import (
    SegmentStyle,
    auto_radius_from_mass,
    cylinder_cross_section,
    depth_sort_segments,
    ellipsoid_cross_section,
    project_3d_to_2d,
    tapered_cylinder_cross_section,
)


class TestSegmentStyle:
    def test_line_value(self) -> None:
        assert SegmentStyle.LINE.value == "line"

    def test_cylinder_value(self) -> None:
        assert SegmentStyle.CYLINDER.value == "cylinder"

    def test_ellipsoid_value(self) -> None:
        assert SegmentStyle.ELLIPSOID.value == "ellipsoid"

    def test_tapered_value(self) -> None:
        assert SegmentStyle.TAPERED.value == "tapered"


class TestCylinderCrossSection:
    def test_returns_shape_4_2(self) -> None:
        start = np.array([0.0, 0.0])
        end = np.array([1.0, 0.0])
        result = cylinder_cross_section(start, end, 0.1)
        assert result.shape == (4, 2)

    def test_horizontal_segment_symmetric_y(self) -> None:
        start = np.array([0.0, 0.0])
        end = np.array([1.0, 0.0])
        result = cylinder_cross_section(start, end, 0.1)
        # Top corners should have y > 0, bottom corners y < 0
        assert result[0, 1] > 0  # top-left y
        assert result[3, 1] < 0  # bottom-left y

    def test_radius_assertion_fails(self) -> None:
        with pytest.raises(AssertionError):
            cylinder_cross_section(np.array([0.0, 0.0]), np.array([1.0, 0.0]), -0.1)

    def test_degenerate_segment_returns_4_points(self) -> None:
        start = np.array([0.5, 0.5])
        end = np.array([0.5, 0.5])  # zero length
        result = cylinder_cross_section(start, end, 0.05)
        assert result.shape == (4, 2)


class TestEllipsoidCrossSection:
    def test_returns_correct_shape(self) -> None:
        centre = np.array([0.0, 0.0])
        result = ellipsoid_cross_section(centre, 1.0, 0.5)
        assert result.shape == (32, 2)

    def test_custom_n_points(self) -> None:
        centre = np.array([0.0, 0.0])
        result = ellipsoid_cross_section(centre, 1.0, 0.5, n_points=16)
        assert result.shape == (16, 2)

    def test_semi_a_assertion(self) -> None:
        with pytest.raises(AssertionError):
            ellipsoid_cross_section(np.array([0.0, 0.0]), -1.0, 0.5)

    def test_semi_b_assertion(self) -> None:
        with pytest.raises(AssertionError):
            ellipsoid_cross_section(np.array([0.0, 0.0]), 1.0, 0.0)

    def test_n_points_min_3(self) -> None:
        with pytest.raises(AssertionError):
            ellipsoid_cross_section(np.array([0.0, 0.0]), 1.0, 0.5, n_points=2)

    def test_circle_is_symmetric(self) -> None:
        centre = np.array([0.0, 0.0])
        result = ellipsoid_cross_section(centre, 1.0, 1.0, n_points=8)
        radii = np.sqrt(result[:, 0] ** 2 + result[:, 1] ** 2)
        np.testing.assert_allclose(radii, 1.0, atol=1e-10)


class TestTaperedCylinderCrossSection:
    def test_returns_shape_4_2(self) -> None:
        start = np.array([0.0, 0.0])
        end = np.array([1.0, 0.0])
        result = tapered_cylinder_cross_section(start, end, 0.2, 0.1)
        assert result.shape == (4, 2)

    def test_assertion_on_zero_radius(self) -> None:
        with pytest.raises(AssertionError):
            tapered_cylinder_cross_section(
                np.array([0.0, 0.0]), np.array([1.0, 0.0]), 0.0, 0.1
            )

    def test_degenerate_fallback(self) -> None:
        start = np.array([0.5, 0.5])
        result = tapered_cylinder_cross_section(start, start.copy(), 0.1, 0.05)
        assert result.shape == (4, 2)


class TestProject3dTo2d:
    def test_origin_projects_to_origin(self) -> None:
        pt = np.array([0.0, 0.0, 0.0])
        result = project_3d_to_2d(pt)
        np.testing.assert_allclose(result, [0.0, 0.0], atol=1e-10)

    def test_segment_geometry_returns_shape_2(self) -> None:
        result = project_3d_to_2d(np.array([1.0, 2.0, 3.0]))
        assert result.shape == (2,)

    def test_with_return_depth(self) -> None:
        pt = np.array([1.0, 0.0, 0.0])
        result, depth = project_3d_to_2d(pt, return_depth=True)
        assert result.shape == (2,)
        assert isinstance(depth, float)

    def test_no_rotation_x_axis_point(self) -> None:
        pt = np.array([3.0, 0.0, 0.0])
        result = project_3d_to_2d(pt)
        assert result[0] == pytest.approx(3.0)


class TestDepthSortSegments:
    def test_sorts_far_to_near(self) -> None:
        segs = [{"depth": 1.0}, {"depth": 3.0}, {"depth": 2.0}]
        result = depth_sort_segments(segs)
        depths = [s["depth"] for s in result]
        assert depths == [3.0, 2.0, 1.0]

    def test_empty_list(self) -> None:
        assert depth_sort_segments([]) == []

    def test_segment_geometry_single_segment(self) -> None:
        segs = [{"depth": 5.0}]
        result = depth_sort_segments(segs)
        assert len(result) == 1


class TestAutoRadiusFromMass:
    def test_positive_result(self) -> None:
        r = auto_radius_from_mass(10.0, 1.0)
        assert r > 0.0

    def test_segment_geometry_returns_float(self) -> None:
        r = auto_radius_from_mass(5.0, 2.0)
        assert isinstance(r, float)

    def test_heavier_mass_larger_radius(self) -> None:
        r1 = auto_radius_from_mass(1.0, 1.0)
        r2 = auto_radius_from_mass(4.0, 1.0)
        assert r2 > r1

    def test_assertion_on_zero_mass(self) -> None:
        with pytest.raises(AssertionError):
            auto_radius_from_mass(0.0, 1.0)
