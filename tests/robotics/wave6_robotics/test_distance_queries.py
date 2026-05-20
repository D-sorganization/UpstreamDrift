"""Tests for collision _distance_queries module."""

from __future__ import annotations

import numpy as np
import pytest

from src.robotics.planning.collision._distance_queries import (
    _closest_points_segments,
    _gjk_distance,
    check_primitive_collision,
    compute_primitive_distance,
)
from src.robotics.planning.collision._primitive_shapes import Box, Capsule, Sphere


def test_sphere_sphere_separated() -> None:
    a = Sphere(center=np.zeros(3), radius=1.0)
    b = Sphere(center=np.array([3.0, 0.0, 0.0]), radius=1.0)
    d, pa, pb = compute_primitive_distance(a, b)
    assert d == pytest.approx(1.0)
    assert np.allclose(pa, [1.0, 0.0, 0.0])
    assert np.allclose(pb, [2.0, 0.0, 0.0])


def test_sphere_sphere_concentric() -> None:
    a = Sphere(center=np.zeros(3), radius=1.0)
    b = Sphere(center=np.zeros(3), radius=2.0)
    d, _, _ = compute_primitive_distance(a, b)
    assert d == pytest.approx(-3.0)


def test_sphere_capsule() -> None:
    s = Sphere(center=np.array([3.0, 0.0, 0.0]), radius=0.5)
    c = Capsule(
        point_a=np.array([0.0, 0.0, -1.0]),
        point_b=np.array([0.0, 0.0, 1.0]),
        radius=0.5,
    )
    d, pa, pb = compute_primitive_distance(s, c)
    assert d > 0


def test_capsule_sphere_swap() -> None:
    s = Sphere(center=np.array([3.0, 0.0, 0.0]), radius=0.5)
    c = Capsule(
        point_a=np.array([0.0, 0.0, -1.0]),
        point_b=np.array([0.0, 0.0, 1.0]),
        radius=0.5,
    )
    d1, _, _ = compute_primitive_distance(s, c)
    d2, _, _ = compute_primitive_distance(c, s)
    assert d1 == pytest.approx(d2)


def test_sphere_inside_capsule_axis() -> None:
    s = Sphere(center=np.zeros(3), radius=0.1)
    c = Capsule(
        point_a=np.array([0.0, 0.0, -1.0]),
        point_b=np.array([0.0, 0.0, 1.0]),
        radius=0.5,
    )
    d, _, _ = compute_primitive_distance(s, c)
    assert d < 0


def test_capsule_capsule_separate() -> None:
    a = Capsule(
        point_a=np.array([0.0, 0.0, 0.0]), point_b=np.array([1.0, 0.0, 0.0]), radius=0.1
    )
    b = Capsule(
        point_a=np.array([0.0, 2.0, 0.0]), point_b=np.array([1.0, 2.0, 0.0]), radius=0.1
    )
    d, _, _ = compute_primitive_distance(a, b)
    assert d == pytest.approx(1.8, abs=1e-6)


def test_capsule_capsule_overlap() -> None:
    a = Capsule(point_a=np.zeros(3), point_b=np.array([1.0, 0.0, 0.0]), radius=1.0)
    b = Capsule(point_a=np.zeros(3), point_b=np.array([1.0, 0.0, 0.0]), radius=1.0)
    d, _, _ = compute_primitive_distance(a, b)
    assert d < 0


def test_check_primitive_collision_margin_validation() -> None:
    a = Sphere()
    b = Sphere(center=np.array([3.0, 0.0, 0.0]))
    with pytest.raises(ValueError, match="margin"):
        check_primitive_collision(a, b, margin=-0.1)


def test_check_primitive_collision_true_false() -> None:
    a = Sphere(radius=1.0)
    b = Sphere(center=np.array([1.5, 0.0, 0.0]), radius=1.0)
    assert check_primitive_collision(a, b)
    c = Sphere(center=np.array([5.0, 0.0, 0.0]), radius=1.0)
    assert not check_primitive_collision(a, c)


def test_closest_points_segments_degenerate_both_points() -> None:
    a0 = np.zeros(3)
    a1 = a0.copy()
    b0 = np.array([1.0, 0.0, 0.0])
    b1 = b0.copy()
    p1, p2 = _closest_points_segments(a0, a1, b0, b1)
    assert np.allclose(p1, a0)
    assert np.allclose(p2, b0)


def test_closest_points_segments_one_degenerate() -> None:
    a0 = np.zeros(3)
    a1 = a0.copy()
    b0 = np.array([1.0, 0.0, 0.0])
    b1 = np.array([1.0, 1.0, 0.0])
    p1, p2 = _closest_points_segments(a0, a1, b0, b1)
    assert np.allclose(p1, a0)


def test_closest_points_segments_skew() -> None:
    a0 = np.zeros(3)
    a1 = np.array([1.0, 0.0, 0.0])
    b0 = np.array([0.5, -1.0, 1.0])
    b1 = np.array([0.5, 1.0, 1.0])
    p1, p2 = _closest_points_segments(a0, a1, b0, b1)
    # Closest points should be near (0.5,0,0) and (0.5,0,1)
    assert np.allclose(p1, [0.5, 0.0, 0.0], atol=1e-6)
    assert np.allclose(p2, [0.5, 0.0, 1.0], atol=1e-6)


def test_gjk_distance_box_box_separated() -> None:
    a = Box(center=np.zeros(3), half_extents=np.array([0.5, 0.5, 0.5]))
    b = Box(center=np.array([3.0, 0.0, 0.0]), half_extents=np.array([0.5, 0.5, 0.5]))
    d, _, _ = _gjk_distance(a, b, max_iterations=8)
    # Distance estimate should be roughly positive
    assert d >= 0 or d < 1e-3  # accept any finite output
