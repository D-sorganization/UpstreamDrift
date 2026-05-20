"""Tests for collision primitives: Sphere, Box, Capsule, Cylinder, ConvexHull."""

from __future__ import annotations

import numpy as np
import pytest

from src.robotics.planning.collision._primitive_shapes import (
    Box,
    Capsule,
    ConvexHull,
    Cylinder,
    Sphere,
)


class TestSphere:
    def test_default_aabb(self) -> None:
        s = Sphere()
        lo, hi = s.get_aabb()
        assert np.allclose(lo, -1.0)
        assert np.allclose(hi, 1.0)

    def test_center_shape_invalid(self) -> None:
        with pytest.raises(ValueError, match="center"):
            Sphere(center=np.zeros(2))

    def test_radius_invalid(self) -> None:
        with pytest.raises(ValueError, match="radius"):
            Sphere(radius=0.0)

    def test_non_finite_center(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            Sphere(center=np.array([np.nan, 0.0, 0.0]))

    def test_contains_point(self) -> None:
        s = Sphere(radius=2.0)
        assert s.contains_point(np.array([1.0, 0.0, 0.0]))
        assert not s.contains_point(np.array([3.0, 0.0, 0.0]))

    def test_compute_support(self) -> None:
        s = Sphere(radius=2.0)
        sp = s.compute_support(np.array([1.0, 0.0, 0.0]))
        assert np.allclose(sp, [2.0, 0.0, 0.0])

    def test_compute_support_zero_direction(self) -> None:
        s = Sphere(center=np.array([1.0, 2.0, 3.0]))
        sp = s.compute_support(np.zeros(3))
        assert np.allclose(sp, [1.0, 2.0, 3.0])


class TestBox:
    def test_default(self) -> None:
        b = Box()
        lo, hi = b.get_aabb()
        assert lo.shape == (3,) and hi.shape == (3,)

    def test_bad_center(self) -> None:
        with pytest.raises(ValueError, match="center"):
            Box(center=np.zeros(2))

    def test_bad_half_extents(self) -> None:
        with pytest.raises(ValueError, match="half_extents"):
            Box(half_extents=np.zeros(2))

    def test_bad_rotation(self) -> None:
        with pytest.raises(ValueError, match="rotation"):
            Box(rotation=np.eye(2))

    def test_neg_half_extents(self) -> None:
        with pytest.raises(ValueError, match="half_extents"):
            Box(half_extents=np.array([-1.0, 1.0, 1.0]))

    def test_contains_point(self) -> None:
        b = Box(half_extents=np.array([1.0, 1.0, 1.0]))
        assert b.contains_point(np.zeros(3))
        assert not b.contains_point(np.array([2.0, 0.0, 0.0]))

    def test_compute_support(self) -> None:
        b = Box(half_extents=np.array([1.0, 1.0, 1.0]))
        sp = b.compute_support(np.array([1.0, 1.0, 1.0]))
        assert np.allclose(sp, [1.0, 1.0, 1.0])

    def test_compute_support_with_rotation(self) -> None:
        R = np.array(
            [
                [0.0, -1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        b = Box(half_extents=np.array([1.0, 0.5, 0.25]), rotation=R)
        sp = b.compute_support(np.array([1.0, 0.0, 0.0]))
        assert sp.shape == (3,)


class TestCapsule:
    def test_default(self) -> None:
        c = Capsule()
        assert c.length == pytest.approx(1.0)
        assert np.allclose(c.center, [0.0, 0.0, 0.0])

    def test_axis(self) -> None:
        c = Capsule(point_a=np.zeros(3), point_b=np.array([0.0, 0.0, 1.0]))
        assert np.allclose(c.axis, [0, 0, 1])

    def test_zero_length_axis(self) -> None:
        c = Capsule(point_a=np.zeros(3), point_b=np.array([1e-15, 0.0, 0.0]))
        assert np.allclose(c.axis, [0, 0, 1])

    def test_invalid_points(self) -> None:
        with pytest.raises(ValueError, match="point_a"):
            Capsule(point_a=np.zeros(2))

    def test_invalid_radius(self) -> None:
        with pytest.raises(ValueError, match="radius"):
            Capsule(radius=0.0)

    def test_aabb(self) -> None:
        c = Capsule(
            point_a=np.array([0.0, 0.0, -1.0]),
            point_b=np.array([0.0, 0.0, 1.0]),
            radius=0.5,
        )
        lo, hi = c.get_aabb()
        assert lo[2] == pytest.approx(-1.5)
        assert hi[2] == pytest.approx(1.5)

    def test_contains(self) -> None:
        c = Capsule(
            point_a=np.array([0.0, 0.0, -1.0]),
            point_b=np.array([0.0, 0.0, 1.0]),
            radius=0.5,
        )
        assert c.contains_point(np.zeros(3))
        assert not c.contains_point(np.array([10.0, 0.0, 0.0]))

    def test_support_endpoint(self) -> None:
        c = Capsule(
            point_a=np.array([0.0, 0.0, -1.0]),
            point_b=np.array([0.0, 0.0, 1.0]),
            radius=0.5,
        )
        sp_pos = c.compute_support(np.array([0.0, 0.0, 1.0]))
        sp_neg = c.compute_support(np.array([0.0, 0.0, -1.0]))
        assert sp_pos[2] > sp_neg[2]
        # zero direction
        assert c.compute_support(np.zeros(3)).shape == (3,)


class TestCylinder:
    def test_default(self) -> None:
        c = Cylinder()
        assert c.half_height == pytest.approx(0.5)

    def test_zero_axis(self) -> None:
        with pytest.raises(ValueError, match="axis"):
            Cylinder(axis=np.zeros(3))

    def test_bad_dims(self) -> None:
        with pytest.raises(ValueError, match="radius"):
            Cylinder(radius=0.0)
        with pytest.raises(ValueError, match="height"):
            Cylinder(height=0.0)

    def test_contains(self) -> None:
        c = Cylinder(radius=0.5, height=2.0)
        assert c.contains_point(np.zeros(3))
        assert not c.contains_point(np.array([0.0, 0.0, 2.0]))
        assert not c.contains_point(np.array([1.0, 0.0, 0.0]))

    def test_support_with_axis_z(self) -> None:
        c = Cylinder(radius=1.0, height=2.0)
        sp = c.compute_support(np.array([1.0, 0.0, 1.0]))
        assert sp.shape == (3,)

    def test_support_zero_direction(self) -> None:
        c = Cylinder()
        sp = c.compute_support(np.zeros(3))
        assert np.allclose(sp, c.center)

    def test_aabb(self) -> None:
        c = Cylinder(radius=1.0, height=2.0)
        lo, hi = c.get_aabb()
        assert lo.shape == (3,) and hi.shape == (3,)


class TestConvexHull:
    def _verts(self) -> np.ndarray:
        return np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )

    def test_default(self) -> None:
        h = ConvexHull(vertices=self._verts())
        assert h.center is not None and h.center.shape == (3,)

    def test_with_explicit_center(self) -> None:
        h = ConvexHull(vertices=self._verts(), center=np.zeros(3))
        assert h.center is not None
        assert np.allclose(h.center, 0.0)

    def test_invalid_shape(self) -> None:
        with pytest.raises(ValueError, match="vertices"):
            ConvexHull(vertices=np.zeros((4, 2)))

    def test_too_few(self) -> None:
        with pytest.raises(ValueError, match="at least 4"):
            ConvexHull(vertices=np.zeros((3, 3)))

    def test_non_finite(self) -> None:
        v = self._verts()
        v[0, 0] = np.nan
        with pytest.raises(ValueError, match="finite"):
            ConvexHull(vertices=v)

    def test_aabb(self) -> None:
        h = ConvexHull(vertices=self._verts())
        lo, hi = h.get_aabb()
        assert np.allclose(lo, 0.0)
        assert np.allclose(hi, 1.0)

    def test_support(self) -> None:
        h = ConvexHull(vertices=self._verts())
        sp = h.compute_support(np.array([1.0, 0.0, 0.0]))
        assert np.allclose(sp, [1.0, 0.0, 0.0])

    def test_contains_at_center(self) -> None:
        h = ConvexHull(vertices=self._verts(), center=np.array([0.25, 0.25, 0.25]))
        assert h.contains_point(np.array([0.25, 0.25, 0.25]))
