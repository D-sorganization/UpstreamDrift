"""Regression tests for the collision narrow phase (issue #7993).

Before the fix, ``_gjk_distance`` returned the norm of the difference of two
*support* points, which is always positive.  Every Box / Cylinder / ConvexHull
pair was therefore reported as collision-free - including shapes entirely
inside one another - and separated distances were over-reported (unsafe
direction).

The cases below pin the four regimes the issue calls out: fully contained,
partially overlapping, exactly touching, and clearly separated.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.robotics.planning.collision._distance_queries import (
    check_primitive_collision,
    compute_primitive_distance,
)
from src.robotics.planning.collision._primitive_shapes import (
    Box,
    Capsule,
    ConvexHull,
    Cylinder,
    Sphere,
)

pytestmark = pytest.mark.unit

UNIT_BOX = Box(center=np.zeros(3), half_extents=np.full(3, 0.5))


def box_sphere_signed_distance(box: Box, sphere: Sphere) -> float:
    """Analytic ground truth: box signed distance field minus the radius."""
    local = box.rotation.T @ (sphere.center - box.center)
    q = np.abs(local) - box.half_extents
    outside = float(np.linalg.norm(np.maximum(q, 0.0)))
    inside = float(min(float(np.max(q)), 0.0))
    return outside + inside - sphere.radius


# --------------------------------------------------------------------------
# Shape fully inside another
# --------------------------------------------------------------------------


def test_sphere_entirely_inside_box_is_a_collision() -> None:
    """The headline defect: a sphere at the box centre reported no collision."""
    sphere = Sphere(center=np.zeros(3), radius=0.5)
    distance, _, _ = compute_primitive_distance(UNIT_BOX, sphere)
    assert distance == pytest.approx(-1.0, abs=1e-12)
    assert check_primitive_collision(UNIT_BOX, sphere)
    assert check_primitive_collision(sphere, UNIT_BOX)


def test_cylinder_concentric_inside_box_is_a_collision() -> None:
    """Box vs fully interpenetrating cylinder used to report distance 1.383."""
    cylinder = Cylinder(center=np.zeros(3), radius=0.4, height=0.8)
    distance, _, _ = compute_primitive_distance(UNIT_BOX, cylinder)
    assert distance < 0.0
    # Depth = min_d h_box(d) + h_cyl(-d) = 0.5 + 0.4 along any face normal.
    assert distance == pytest.approx(-0.9, abs=1e-3)
    assert check_primitive_collision(UNIT_BOX, cylinder)


def test_box_inside_box_reports_full_penetration_depth() -> None:
    inner = Box(center=np.zeros(3), half_extents=np.full(3, 0.2))
    distance, _, _ = compute_primitive_distance(UNIT_BOX, inner)
    assert distance == pytest.approx(-0.7, abs=1e-3)
    assert check_primitive_collision(UNIT_BOX, inner)


def test_convex_hull_inside_box_is_a_collision() -> None:
    hull = ConvexHull(
        vertices=np.array(
            [
                [-0.2, -0.2, -0.2],
                [0.2, -0.2, -0.2],
                [0.0, 0.25, -0.2],
                [0.0, 0.0, 0.25],
            ]
        )
    )
    distance, _, _ = compute_primitive_distance(UNIT_BOX, hull)
    assert distance < 0.0
    assert check_primitive_collision(UNIT_BOX, hull)


# --------------------------------------------------------------------------
# Partial overlap
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("center_x", "expected"),
    [(0.6, -0.4), (0.7, -0.3), (0.9, -0.1)],
)
def test_sphere_partially_overlapping_box(center_x: float, expected: float) -> None:
    sphere = Sphere(center=np.array([center_x, 0.0, 0.0]), radius=0.5)
    distance, _, _ = compute_primitive_distance(UNIT_BOX, sphere)
    assert distance == pytest.approx(expected, abs=1e-12)
    assert distance == pytest.approx(
        box_sphere_signed_distance(UNIT_BOX, sphere), abs=1e-12
    )
    assert check_primitive_collision(UNIT_BOX, sphere)


def test_box_box_partial_overlap_reports_penetration_depth() -> None:
    other = Box(center=np.array([0.8, 0.0, 0.0]), half_extents=np.full(3, 0.5))
    distance, _, _ = compute_primitive_distance(UNIT_BOX, other)
    assert distance == pytest.approx(-0.2, abs=1e-3)
    assert check_primitive_collision(UNIT_BOX, other)


# --------------------------------------------------------------------------
# Exact touching
# --------------------------------------------------------------------------


def test_sphere_touching_box_face_is_distance_zero() -> None:
    sphere = Sphere(center=np.array([1.0, 0.0, 0.0]), radius=0.5)
    distance, _, _ = compute_primitive_distance(UNIT_BOX, sphere)
    assert distance == pytest.approx(0.0, abs=1e-12)
    # A touching pair must count as a collision at margin 0.
    assert check_primitive_collision(UNIT_BOX, sphere)


def test_box_box_touching_faces_is_a_collision() -> None:
    other = Box(center=np.array([1.0, 0.0, 0.0]), half_extents=np.full(3, 0.5))
    distance, _, _ = compute_primitive_distance(UNIT_BOX, other)
    assert distance == pytest.approx(0.0, abs=1e-3)
    assert check_primitive_collision(UNIT_BOX, other)


# --------------------------------------------------------------------------
# Clearly separated - the distance must not be over-reported
# --------------------------------------------------------------------------


@pytest.mark.parametrize(("center_x", "expected"), [(1.2, 0.2), (2.0, 1.0)])
def test_sphere_separated_from_box_exact_distance(
    center_x: float, expected: float
) -> None:
    """x = 2.0 previously reported 1.3047 m of clearance instead of 1.0 m."""
    sphere = Sphere(center=np.array([center_x, 0.0, 0.0]), radius=0.5)
    distance, point_sphere, point_box = compute_primitive_distance(UNIT_BOX, sphere)
    assert distance == pytest.approx(expected, abs=1e-12)
    assert not check_primitive_collision(UNIT_BOX, sphere)
    assert float(np.linalg.norm(point_box - point_sphere)) == pytest.approx(
        expected, abs=1e-9
    )


def test_box_box_diagonal_gap_exact_distance() -> None:
    """Diagonal corner-to-corner gap; previously reported 1.4142 instead of 2.1213."""
    other = Box(center=np.array([2.0, 2.0, 0.0]), half_extents=np.full(3, 0.5))
    distance, _, _ = compute_primitive_distance(UNIT_BOX, other)
    # Corner (0.5, 0.5) to corner (1.5, 1.5): gap = sqrt(1^2 + 1^2).
    assert distance == pytest.approx(math.sqrt(2.0), abs=1e-9)
    assert not check_primitive_collision(UNIT_BOX, other)


def test_box_cylinder_separated_exact_distance() -> None:
    cylinder = Cylinder(
        center=np.array([2.0, 0.0, 0.0]), radius=0.3, height=0.6, axis=[0.0, 0.0, 1.0]
    )
    distance, _, _ = compute_primitive_distance(UNIT_BOX, cylinder)
    assert distance == pytest.approx(2.0 - 0.5 - 0.3, abs=1e-9)
    assert not check_primitive_collision(UNIT_BOX, cylinder)


def test_separated_distance_matches_duality_certificate() -> None:
    """For any unit u, dist(A, B) >= -(h_A(u) + h_B(-u)).

    Evaluating the bound at the direction of the returned witness points
    certifies that the reported distance is exact, independently of GJK.
    """
    rng = np.random.default_rng(20260724)
    shapes = [
        Box(center=np.zeros(3), half_extents=np.array([0.4, 0.3, 0.5])),
        Cylinder(center=np.zeros(3), radius=0.35, height=0.7, axis=[1.0, 1.0, 0.0]),
        ConvexHull(vertices=rng.normal(scale=0.3, size=(10, 3))),
        Capsule(
            point_a=np.array([0.0, 0.0, -0.3]),
            point_b=np.array([0.0, 0.2, 0.3]),
            radius=0.15,
        ),
    ]
    for shape_a in shapes:
        for shape_b in shapes:
            translated = _translate(shape_b, np.array([2.5, 0.4, -0.3]))
            distance, point_a, point_b = compute_primitive_distance(shape_a, translated)
            assert distance > 0.0
            direction = point_b - point_a
            direction = direction / float(np.linalg.norm(direction))
            lower = -(
                float(shape_a.compute_support(direction) @ direction)
                + float(translated.compute_support(-direction) @ -direction)
            )
            upper = float(np.linalg.norm(point_b - point_a))
            assert lower - 1e-9 <= distance <= upper + 1e-9
            assert upper - lower < 1e-7


def _translate(shape, offset: np.ndarray):
    """Return a copy of ``shape`` moved by ``offset``."""
    if isinstance(shape, Box):
        return Box(
            center=shape.center + offset,
            half_extents=shape.half_extents,
            rotation=shape.rotation,
        )
    if isinstance(shape, Cylinder):
        return Cylinder(
            center=shape.center + offset,
            radius=shape.radius,
            height=shape.height,
            axis=shape.axis,
        )
    if isinstance(shape, ConvexHull):
        return ConvexHull(vertices=shape.vertices + offset)
    if isinstance(shape, Capsule):
        return Capsule(
            point_a=shape.point_a + offset,
            point_b=shape.point_b + offset,
            radius=shape.radius,
        )
    return Sphere(center=shape.center + offset, radius=shape.radius)


# --------------------------------------------------------------------------
# Penetration depth must never be under-reported (conservative direction)
# --------------------------------------------------------------------------


def test_penetration_depth_is_conservative_against_direction_scan() -> None:
    """min over a dense direction set is an upper bound on the true depth.

    The reported depth must be at least as large as the true depth; comparing
    against a 4096-direction brute-force scan checks that our estimate is not
    a *worse* upper bound than plain sampling.
    """
    rng = np.random.default_rng(4)
    indices = np.arange(4096, dtype=np.float64) + 0.5
    phi = np.arccos(1.0 - 2.0 * indices / 4096)
    theta = math.pi * (1.0 + math.sqrt(5.0)) * indices
    scan = np.column_stack(
        [np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)]
    )

    for _ in range(6):
        other = Box(
            center=rng.uniform(-0.4, 0.4, 3),
            half_extents=rng.uniform(0.25, 0.6, 3),
        )
        distance, _, _ = compute_primitive_distance(UNIT_BOX, other)
        assert distance < 0.0
        assert check_primitive_collision(UNIT_BOX, other)
        widths = np.einsum(
            "ij,ij->i",
            UNIT_BOX.compute_support_batch(scan) - other.compute_support_batch(-scan),
            scan,
        )
        scan_depth = float(np.min(widths))
        assert -distance <= scan_depth + 1e-3


# --------------------------------------------------------------------------
# The batch support mapping must agree with the scalar one
# --------------------------------------------------------------------------


def test_compute_support_batch_matches_scalar() -> None:
    rng = np.random.default_rng(11)
    directions = rng.normal(size=(64, 3))
    directions = np.vstack([directions, np.eye(3), -np.eye(3), np.zeros((1, 3))])
    shapes = [
        Sphere(center=np.array([0.1, 0.2, 0.3]), radius=0.4),
        Box(center=np.array([0.0, 1.0, 0.0]), half_extents=np.array([0.3, 0.5, 0.2])),
        Capsule(
            point_a=np.array([0.0, 0.0, -1.0]),
            point_b=np.array([0.0, 0.5, 1.0]),
            radius=0.2,
        ),
        Cylinder(center=np.zeros(3), radius=0.4, height=0.9, axis=[1.0, 1.0, 0.0]),
        ConvexHull(vertices=rng.normal(size=(9, 3))),
    ]
    for shape in shapes:
        batched = shape.compute_support_batch(directions)
        looped = np.array([shape.compute_support(d) for d in directions])
        assert np.allclose(batched, looped, atol=1e-12)
