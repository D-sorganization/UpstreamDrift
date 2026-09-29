from __future__ import annotations

import math

import numpy as np

from ._convex_distance import convex_signed_distance
from ._primitive_shapes import Box, Capsule, Sphere
from ._primitives_base import GeometricPrimitive


def compute_primitive_distance(
    prim_a: GeometricPrimitive,
    prim_b: GeometricPrimitive,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Compute signed distance between two primitives.

    Design by Contract:
        Preconditions:
            - Both primitives must be valid (positive dimensions)

        Postconditions:
            - Returns (distance, point_a, point_b)
            - distance < 0 indicates penetration
            - point_a is closest point on prim_a
            - point_b is closest point on prim_b

    Args:
        prim_a: First geometric primitive.
        prim_b: Second geometric primitive.

    Returns:
        Tuple of (signed_distance, closest_point_a, closest_point_b).
    """
    # Dispatch based on primitive types for specialized algorithms
    if prim_a is None:
        raise ValueError("prim_a must be provided")
    if isinstance(prim_a, Sphere) and isinstance(prim_b, Sphere):
        return _sphere_sphere_distance(prim_a, prim_b)
    if isinstance(prim_a, Sphere) and isinstance(prim_b, Capsule):
        return _sphere_capsule_distance(prim_a, prim_b)
    if isinstance(prim_a, Capsule) and isinstance(prim_b, Sphere):
        d, pb, pa = _sphere_capsule_distance(prim_b, prim_a)
        return d, pa, pb
    if isinstance(prim_a, Capsule) and isinstance(prim_b, Capsule):
        return _capsule_capsule_distance(prim_a, prim_b)
    if isinstance(prim_a, Sphere) and isinstance(prim_b, Box):
        return _sphere_box_distance(prim_a, prim_b)
    if isinstance(prim_a, Box) and isinstance(prim_b, Sphere):
        d, pb, pa = _sphere_box_distance(prim_b, prim_a)
        return d, pa, pb

    # General convex case: GJK for separation, support-function minimisation
    # for penetration depth.  See ``_convex_distance`` for the guarantees.
    return convex_signed_distance(prim_a, prim_b)


def _sphere_sphere_distance(
    sphere_a: Sphere,
    sphere_b: Sphere,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Distance between two spheres."""
    if sphere_a is None:
        raise ValueError("sphere_a must be provided")
    diff = sphere_b.center - sphere_a.center
    center_dist = math.hypot(diff[0], diff[1], diff[2])

    if center_dist < 1e-10:
        # Concentric spheres
        return -(sphere_a.radius + sphere_b.radius), sphere_a.center, sphere_b.center

    direction = diff / center_dist
    distance = float(center_dist - sphere_a.radius - sphere_b.radius)

    point_a = sphere_a.center + sphere_a.radius * direction
    point_b = sphere_b.center - sphere_b.radius * direction

    return distance, point_a, point_b


def _sphere_capsule_distance(
    sphere: Sphere,
    capsule: Capsule,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Distance between sphere and capsule."""
    # Closest point on capsule axis to sphere center
    if sphere is None:
        raise ValueError("sphere must be provided")
    closest_on_axis = capsule._closest_point_on_segment(sphere.center)

    # Now it's sphere-sphere distance
    diff = sphere.center - closest_on_axis
    center_dist = math.hypot(diff[0], diff[1], diff[2])

    if center_dist < 1e-10:
        # Sphere center on capsule axis
        direction = np.array([1.0, 0.0, 0.0])
        distance = -(sphere.radius + capsule.radius)
        return distance, sphere.center, closest_on_axis

    direction = diff / center_dist
    distance = float(center_dist - sphere.radius - capsule.radius)

    point_sphere = sphere.center - sphere.radius * direction
    point_capsule = closest_on_axis + capsule.radius * direction

    return distance, point_sphere, point_capsule


def _capsule_capsule_distance(
    cap_a: Capsule,
    cap_b: Capsule,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Distance between two capsules."""
    # Find closest points between line segments
    if cap_a is None:
        raise ValueError("cap_a must be provided")
    closest_a, closest_b = _closest_points_segments(
        cap_a.point_a, cap_a.point_b, cap_b.point_a, cap_b.point_b
    )

    diff = closest_b - closest_a
    center_dist = math.hypot(diff[0], diff[1], diff[2])

    if center_dist < 1e-10:
        direction = np.array([1.0, 0.0, 0.0])
        distance = -(cap_a.radius + cap_b.radius)
        return distance, closest_a, closest_b

    direction = diff / center_dist
    distance = float(center_dist - cap_a.radius - cap_b.radius)

    point_a = closest_a + cap_a.radius * direction
    point_b = closest_b - cap_b.radius * direction

    return distance, point_a, point_b


def _closest_points_segments(
    a0: np.ndarray,
    a1: np.ndarray,
    b0: np.ndarray,
    b1: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Find closest points between two line segments."""
    if a0 is None:
        raise ValueError("a0 must be provided")
    d1 = a1 - a0  # Direction of segment 1
    d2 = b1 - b0  # Direction of segment 2
    r = a0 - b0

    a = np.dot(d1, d1)
    e = np.dot(d2, d2)
    f = np.dot(d2, r)

    # Check if either segment degenerates to a point
    if a < 1e-10 and e < 1e-10:
        return a0.copy(), b0.copy()

    if a < 1e-10:
        s = 0.0
        t = np.clip(f / e, 0.0, 1.0)
    else:
        c = np.dot(d1, r)
        if e < 1e-10:
            t = 0.0
            s = np.clip(-c / a, 0.0, 1.0)
        else:
            b_coef = np.dot(d1, d2)
            denom = a * e - b_coef * b_coef

            if abs(denom) > 1e-10:
                s = np.clip((b_coef * f - c * e) / denom, 0.0, 1.0)
            else:
                s = 0.0

            t = (b_coef * s + f) / e

            if t < 0.0:
                t = 0.0
                s = np.clip(-c / a, 0.0, 1.0)
            elif t > 1.0:
                t = 1.0
                s = np.clip((b_coef - c) / a, 0.0, 1.0)

    return a0 + s * d1, b0 + t * d2


def _sphere_box_distance(
    sphere: Sphere,
    box: Box,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Exact signed distance between a sphere and an oriented box.

    Uses the closed-form box signed distance field evaluated at the sphere
    centre, which is exact both outside (positive) and inside (negative) the
    box:

        q = |R^T (p - c)| - h
        sdf(p) = ||max(q, 0)|| + min(max(q_x, q_y, q_z), 0)

    Design by Contract:
        Postconditions:
            - distance < 0 iff the sphere overlaps the box
            - |distance| is the exact separation / penetration depth

    Args:
        sphere: Sphere primitive.
        box: Box primitive.

    Returns:
        Tuple of (signed_distance, point_on_sphere, point_on_box).
    """
    if sphere is None:
        raise ValueError("sphere must be provided")
    if box is None:
        raise ValueError("box must be provided")

    local = box.rotation.T @ (sphere.center - box.center)
    half = box.half_extents
    q = np.abs(local) - half

    outside = np.maximum(q, 0.0)
    outside_dist = float(math.hypot(outside[0], outside[1], outside[2]))
    inside_dist = float(min(float(np.max(q)), 0.0))
    box_sdf = outside_dist + inside_dist
    distance = float(box_sdf - sphere.radius)

    if outside_dist > 1e-12:
        # Sphere centre is outside the box: nearest surface point is the
        # componentwise clamp of the centre into the box.
        local_closest = np.clip(local, -half, half)
        point_box = box.rotation @ local_closest + box.center
        direction = point_box - sphere.center
        norm = float(math.hypot(direction[0], direction[1], direction[2]))
        unit = direction / norm if norm > 1e-12 else np.array([1.0, 0.0, 0.0])
    else:
        # Sphere centre is inside the box: nearest surface point lies on the
        # face whose slab the centre is closest to.
        axis = int(np.argmax(q))
        sign = 1.0 if local[axis] >= 0.0 else -1.0
        local_closest = local.copy()
        local_closest[axis] = sign * half[axis]
        point_box = box.rotation @ local_closest + box.center
        unit = box.rotation @ (sign * np.eye(3)[axis])

    point_sphere = sphere.center + sphere.radius * unit
    return distance, point_sphere, point_box


def check_primitive_collision(
    prim_a: GeometricPrimitive,
    prim_b: GeometricPrimitive,
    margin: float = 0.0,
) -> bool:
    """Check if two primitives are in collision.

    Design by Contract:
        Preconditions:
            - margin >= 0

        Postconditions:
            - Returns True if distance <= margin

    Args:
        prim_a: First geometric primitive.
        prim_b: Second geometric primitive.
        margin: Safety margin [m]. Collision if distance < margin.

    Returns:
        True if primitives are in collision (within margin).
    """
    if margin < 0:
        raise ValueError("margin must be non-negative")

    distance, _, _ = compute_primitive_distance(prim_a, prim_b)
    return distance <= margin
