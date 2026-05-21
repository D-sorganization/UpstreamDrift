from __future__ import annotations

import math

import numpy as np

from ._primitive_shapes import Capsule, Sphere
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

    # Fallback: GJK-based distance (simplified)
    return _gjk_distance(prim_a, prim_b)


def _sphere_sphere_distance(
    sphere_a: Sphere,
    sphere_b: Sphere,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Distance between two spheres."""
    if sphere_a is None:
        raise ValueError("sphere_a must be provided")
    diff = sphere_b.center - sphere_a.center
    center_dist = math.hypot(*diff)

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
    center_dist = math.hypot(*diff)

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
    center_dist = math.hypot(*diff)

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


def _gjk_distance(  # noqa: C901
    prim_a: GeometricPrimitive,
    prim_b: GeometricPrimitive,
    max_iterations: int = 32,
) -> tuple[float, np.ndarray, np.ndarray]:
    """GJK-based distance computation (simplified).

    This is a simplified implementation. For production use,
    consider using a proper GJK library.
    """
    # Initial direction from A to B
    if prim_a is None:
        raise ValueError("prim_a must be provided")
    direction = prim_b.compute_support(np.array([1, 0, 0])) - prim_a.compute_support(
        np.array([-1, 0, 0])
    )
    if math.hypot(*direction) < 1e-10:
        direction = np.array([1.0, 0.0, 0.0])
    else:
        direction = direction / math.hypot(*direction)

    # Simplex vertices
    simplex: list[np.ndarray] = []

    for _ in range(max_iterations):
        # Support point in Minkowski difference
        support_a = prim_a.compute_support(direction)
        support_b = prim_b.compute_support(-direction)
        support = support_a - support_b

        # Check if we've passed the origin
        # Origin not contained, compute distance
        if np.dot(support, direction) < 0 and len(simplex) == 0:
            # Return distance between supports
            diff = support_b - support_a
            dist = float(math.hypot(*diff))
            return dist, support_a, support_b

        simplex.append(support)

        # Update simplex and direction
        if len(simplex) == 1:
            direction = -simplex[0]
            norm = math.hypot(*direction)
            if norm < 1e-10:
                # Origin at support point (collision)
                return 0.0, support_a, support_b
            direction = direction / norm
        elif len(simplex) == 2:
            # Line case
            ab = simplex[1] - simplex[0]
            ao = -simplex[0]
            t = np.dot(ao, ab) / (np.dot(ab, ab) + 1e-10)
            t = np.clip(t, 0.0, 1.0)
            closest = simplex[0] + t * ab
            dist = float(math.hypot(*closest))
            if dist < 1e-6:
                # Origin very close to simplex (collision)
                return 0.0, support_a, support_b
            direction = -closest / dist
        else:
            # For simplicity, just use last two points
            simplex = simplex[-2:]
            ab = simplex[1] - simplex[0]
            ao = -simplex[0]
            t = np.dot(ao, ab) / (np.dot(ab, ab) + 1e-10)
            t = np.clip(t, 0.0, 1.0)
            closest = simplex[0] + t * ab
            dist = float(math.hypot(*closest))
            if dist < 1e-6:
                return 0.0, support_a, support_b
            direction = -closest / dist

    # Max iterations reached, estimate distance
    support_a = prim_a.compute_support(direction)
    support_b = prim_b.compute_support(-direction)
    diff = support_b - support_a
    return float(math.hypot(*diff)), support_a, support_b


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
