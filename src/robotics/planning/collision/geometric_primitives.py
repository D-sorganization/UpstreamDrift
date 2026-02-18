"""Geometric primitives for collision detection.

This module provides basic geometric shapes and collision algorithms:
- Sphere, Box, Capsule, Cylinder primitives
- Distance computation between primitives
- Collision detection between primitives

Design by Contract:
    All primitives must have positive dimensions.
    All transformations must be valid (finite, proper rotation).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


class GeometricPrimitive(ABC):
    """Abstract base class for geometric primitives.

    Design by Contract:
        Preconditions:
            - All dimension parameters must be positive
            - Position must be finite 3D vector
            - Rotation must be valid 3x3 rotation matrix

        Postconditions:
            - get_aabb() returns valid axis-aligned bounding box
            - contains_point() returns correct membership test

        Invariants:
            - Primitive dimensions are immutable after construction
    """

    @abstractmethod
    def get_aabb(self) -> tuple[np.ndarray, np.ndarray]:
        """Get axis-aligned bounding box.

        Returns:
            Tuple of (min_corner, max_corner) in world frame.
        """
        ...

    @abstractmethod
    def contains_point(self, point: np.ndarray) -> bool:
        """Check if point is inside primitive.

        Args:
            point: 3D point in world frame.

        Returns:
            True if point is inside or on surface.
        """
        ...

    @abstractmethod
    def compute_support(self, direction: np.ndarray) -> np.ndarray:
        """Compute support point in given direction.

        Support mapping for GJK/EPA algorithms.

        Args:
            direction: Unit direction vector.

        Returns:
            Point on primitive surface furthest in direction.
        """
        ...


@dataclass
class Sphere(GeometricPrimitive):
    """Sphere primitive.

    Attributes:
        center: Center position in world frame [m].
        radius: Sphere radius [m].
    """

    center: np.ndarray = field(default_factory=lambda: np.zeros(3))
    radius: float = 1.0

    def __post_init__(self) -> None:
        """Validate sphere parameters."""
        self.center = np.asarray(self.center, dtype=np.float64)
        if self.center.shape != (3,):
            raise ValueError("center must be shape (3,)")
        if not np.all(np.isfinite(self.center)):
            raise ValueError("center must be finite")
        if self.radius <= 0:
            raise ValueError("radius must be positive")

    def get_aabb(self) -> tuple[np.ndarray, np.ndarray]:
        """Get axis-aligned bounding box."""
        r = np.array([self.radius, self.radius, self.radius])
        return self.center - r, self.center + r

    def contains_point(self, point: np.ndarray) -> bool:
        """Check if point is inside sphere."""
        point = np.asarray(point)
        return float(np.linalg.norm(point - self.center)) <= self.radius

    def compute_support(self, direction: np.ndarray) -> np.ndarray:
        """Compute support point."""
        direction = np.asarray(direction)
        norm = np.linalg.norm(direction)
        if norm < 1e-10:
            return self.center.copy()
        return self.center + self.radius * direction / norm


@dataclass
class Box(GeometricPrimitive):
    """Axis-aligned box primitive.

    Attributes:
        center: Center position in world frame [m].
        half_extents: Half-sizes along each axis [m].
        rotation: Rotation matrix (3x3) from local to world frame.
    """

    center: np.ndarray = field(default_factory=lambda: np.zeros(3))
    half_extents: np.ndarray = field(default_factory=lambda: np.ones(3) * 0.5)
    rotation: np.ndarray = field(default_factory=lambda: np.eye(3))

    def __post_init__(self) -> None:
        """Validate box parameters."""
        self.center = np.asarray(self.center, dtype=np.float64)
        self.half_extents = np.asarray(self.half_extents, dtype=np.float64)
        self.rotation = np.asarray(self.rotation, dtype=np.float64)

        if self.center.shape != (3,):
            raise ValueError("center must be shape (3,)")
        if self.half_extents.shape != (3,):
            raise ValueError("half_extents must be shape (3,)")
        if self.rotation.shape != (3, 3):
            raise ValueError("rotation must be shape (3, 3)")
        if not np.all(np.isfinite(self.center)):
            raise ValueError("center must be finite")
        if not np.all(self.half_extents > 0):
            raise ValueError("half_extents must be positive")

    def get_aabb(self) -> tuple[np.ndarray, np.ndarray]:
        """Get axis-aligned bounding box."""
        # Compute AABB of oriented box
        corners = self._get_corners()
        return np.min(corners, axis=0), np.max(corners, axis=0)

    def _get_corners(self) -> np.ndarray:
        """Get all 8 corners of the box in world frame."""
        h = self.half_extents
        local_corners = np.array(
            [
                [-h[0], -h[1], -h[2]],
                [-h[0], -h[1], h[2]],
                [-h[0], h[1], -h[2]],
                [-h[0], h[1], h[2]],
                [h[0], -h[1], -h[2]],
                [h[0], -h[1], h[2]],
                [h[0], h[1], -h[2]],
                [h[0], h[1], h[2]],
            ]
        )
        return (self.rotation @ local_corners.T).T + self.center

    def contains_point(self, point: np.ndarray) -> bool:
        """Check if point is inside box."""
        point = np.asarray(point)
        # Transform to local frame
        local_point = self.rotation.T @ (point - self.center)
        return bool(np.all(np.abs(local_point) <= self.half_extents))

    def compute_support(self, direction: np.ndarray) -> np.ndarray:
        """Compute support point."""
        direction = np.asarray(direction)
        # Transform direction to local frame
        local_dir = self.rotation.T @ direction
        # Support in local frame
        local_support = np.sign(local_dir) * self.half_extents
        # Handle zero components
        local_support = np.where(
            np.abs(local_dir) < 1e-10, self.half_extents, local_support
        )
        # Transform back to world
        return self.rotation @ local_support + self.center


@dataclass
class Capsule(GeometricPrimitive):
    """Capsule primitive (sphere-swept line segment).

    Attributes:
        point_a: First endpoint in world frame [m].
        point_b: Second endpoint in world frame [m].
        radius: Capsule radius [m].
    """

    point_a: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, -0.5]))
    point_b: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.5]))
    radius: float = 0.1

    def __post_init__(self) -> None:
        """Validate capsule parameters."""
        self.point_a = np.asarray(self.point_a, dtype=np.float64)
        self.point_b = np.asarray(self.point_b, dtype=np.float64)

        if self.point_a.shape != (3,):
            raise ValueError("point_a must be shape (3,)")
        if self.point_b.shape != (3,):
            raise ValueError("point_b must be shape (3,)")
        if not np.all(np.isfinite(self.point_a)):
            raise ValueError("point_a must be finite")
        if not np.all(np.isfinite(self.point_b)):
            raise ValueError("point_b must be finite")
        if self.radius <= 0:
            raise ValueError("radius must be positive")

    @property
    def length(self) -> float:
        """Get capsule length (distance between endpoints)."""
        return float(np.linalg.norm(self.point_b - self.point_a))

    @property
    def axis(self) -> np.ndarray:
        """Get capsule axis direction (normalized)."""
        diff = self.point_b - self.point_a
        length = np.linalg.norm(diff)
        if length < 1e-10:
            return np.array([0.0, 0.0, 1.0])
        return diff / length

    @property
    def center(self) -> np.ndarray:
        """Get capsule center."""
        return (self.point_a + self.point_b) / 2

    def get_aabb(self) -> tuple[np.ndarray, np.ndarray]:
        """Get axis-aligned bounding box."""
        r = np.array([self.radius, self.radius, self.radius])
        min_corner = np.minimum(self.point_a, self.point_b) - r
        max_corner = np.maximum(self.point_a, self.point_b) + r
        return min_corner, max_corner

    def _closest_point_on_segment(self, point: np.ndarray) -> np.ndarray:
        """Get closest point on capsule's central line segment."""
        ab = self.point_b - self.point_a
        t = np.dot(point - self.point_a, ab) / (np.dot(ab, ab) + 1e-10)
        t = np.clip(t, 0.0, 1.0)
        return self.point_a + t * ab

    def contains_point(self, point: np.ndarray) -> bool:
        """Check if point is inside capsule."""
        point = np.asarray(point)
        closest = self._closest_point_on_segment(point)
        return float(np.linalg.norm(point - closest)) <= self.radius

    def compute_support(self, direction: np.ndarray) -> np.ndarray:
        """Compute support point."""
        direction = np.asarray(direction)
        norm = np.linalg.norm(direction)
        if norm < 1e-10:
            return self.point_a.copy()
        d = direction / norm
        # Choose endpoint further in direction
        if np.dot(d, self.point_b - self.point_a) >= 0:
            return self.point_b + self.radius * d
        return self.point_a + self.radius * d


@dataclass
class Cylinder(GeometricPrimitive):
    """Cylinder primitive.

    Attributes:
        center: Center position in world frame [m].
        radius: Cylinder radius [m].
        height: Cylinder height [m].
        axis: Cylinder axis direction (normalized).
    """

    center: np.ndarray = field(default_factory=lambda: np.zeros(3))
    radius: float = 0.5
    height: float = 1.0
    axis: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))

    def __post_init__(self) -> None:
        """Validate cylinder parameters."""
        self.center = np.asarray(self.center, dtype=np.float64)
        self.axis = np.asarray(self.axis, dtype=np.float64)

        if self.center.shape != (3,):
            raise ValueError("center must be shape (3,)")
        if self.axis.shape != (3,):
            raise ValueError("axis must be shape (3,)")
        if not np.all(np.isfinite(self.center)):
            raise ValueError("center must be finite")
        if self.radius <= 0:
            raise ValueError("radius must be positive")
        if self.height <= 0:
            raise ValueError("height must be positive")

        # Normalize axis
        norm = np.linalg.norm(self.axis)
        if norm < 1e-10:
            raise ValueError("axis must be non-zero")
        self.axis = self.axis / norm

    @property
    def half_height(self) -> float:
        """Get half height."""
        return self.height / 2

    def get_aabb(self) -> tuple[np.ndarray, np.ndarray]:
        """Get axis-aligned bounding box."""
        # Get endpoints
        top = self.center + self.half_height * self.axis
        bottom = self.center - self.half_height * self.axis

        # Compute AABB including radius
        # For arbitrary axis, the AABB is more complex
        r_vec = np.sqrt(1 - self.axis**2) * self.radius
        r_vec = np.maximum(r_vec, self.radius * 0.01)  # Avoid degenerate case

        min_corner = np.minimum(top, bottom) - r_vec - np.array([0, 0, 0])
        max_corner = np.maximum(top, bottom) + r_vec

        # Add radius in all directions for safety
        r = np.array([self.radius, self.radius, self.radius])
        min_corner = np.minimum(min_corner, np.minimum(top, bottom) - r)
        max_corner = np.maximum(max_corner, np.maximum(top, bottom) + r)

        return min_corner, max_corner

    def contains_point(self, point: np.ndarray) -> bool:
        """Check if point is inside cylinder."""
        point = np.asarray(point)
        # Project onto axis
        to_point = point - self.center
        along_axis = np.dot(to_point, self.axis)

        # Check height
        if abs(along_axis) > self.half_height:
            return False

        # Check radius (perpendicular distance)
        perp = to_point - along_axis * self.axis
        return float(np.linalg.norm(perp)) <= self.radius

    def compute_support(self, direction: np.ndarray) -> np.ndarray:
        """Compute support point."""
        direction = np.asarray(direction)
        norm = np.linalg.norm(direction)
        if norm < 1e-10:
            return self.center.copy()

        d = direction / norm

        # Component along axis
        d_along = np.dot(d, self.axis) * self.axis
        # Component perpendicular to axis
        d_perp = d - d_along

        # Support on axis
        if np.dot(d, self.axis) >= 0:
            axis_support = self.center + self.half_height * self.axis
        else:
            axis_support = self.center - self.half_height * self.axis

        # Support on radius (perpendicular)
        perp_norm = np.linalg.norm(d_perp)
        if perp_norm > 1e-10:
            return axis_support + self.radius * d_perp / perp_norm

        return axis_support


@dataclass
class ConvexHull(GeometricPrimitive):
    """Convex hull primitive from point cloud.

    Attributes:
        vertices: Array of vertices (N, 3) in world frame [m].
        center: Center of mass of vertices.
    """

    vertices: np.ndarray = field(default_factory=lambda: np.zeros((4, 3)))
    center: np.ndarray | None = None

    def __post_init__(self) -> None:
        """Validate convex hull parameters."""
        self.vertices = np.asarray(self.vertices, dtype=np.float64)

        if self.vertices.ndim != 2 or self.vertices.shape[1] != 3:
            raise ValueError("vertices must be shape (N, 3)")
        if len(self.vertices) < 4:
            raise ValueError("convex hull requires at least 4 vertices")
        if not np.all(np.isfinite(self.vertices)):
            raise ValueError("vertices must be finite")

        # Compute center if not provided
        if self.center is None:
            self.center = np.mean(self.vertices, axis=0)
        else:
            self.center = np.asarray(self.center, dtype=np.float64)

    def get_aabb(self) -> tuple[np.ndarray, np.ndarray]:
        """Get axis-aligned bounding box."""
        return np.min(self.vertices, axis=0), np.max(self.vertices, axis=0)

    def contains_point(self, point: np.ndarray) -> bool:
        """Check if point is inside convex hull.

        Uses a simple heuristic - point should be on the "inside"
        of all faces. For exact test, use proper convex hull algorithm.
        """
        point = np.asarray(point)
        # Simple heuristic: point is inside if closer to center than
        # all vertices in the same direction
        to_point = point - self.center
        norm = np.linalg.norm(to_point)
        if norm < 1e-10:
            return True  # At center

        direction = to_point / norm
        support = self.compute_support(direction)
        support_dist = np.dot(support - self.center, direction)
        return norm <= support_dist

    def compute_support(self, direction: np.ndarray) -> np.ndarray:
        """Compute support point."""
        direction = np.asarray(direction)
        # Find vertex with maximum dot product
        dots = self.vertices @ direction
        idx = np.argmax(dots)
        return self.vertices[idx].copy()


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
    diff = sphere_b.center - sphere_a.center
    center_dist = np.linalg.norm(diff)

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
    closest_on_axis = capsule._closest_point_on_segment(sphere.center)

    # Now it's sphere-sphere distance
    diff = sphere.center - closest_on_axis
    center_dist = np.linalg.norm(diff)

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
    closest_a, closest_b = _closest_points_segments(
        cap_a.point_a, cap_a.point_b, cap_b.point_a, cap_b.point_b
    )

    diff = closest_b - closest_a
    center_dist = np.linalg.norm(diff)

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


def _gjk_distance(
    prim_a: GeometricPrimitive,
    prim_b: GeometricPrimitive,
    max_iterations: int = 32,
) -> tuple[float, np.ndarray, np.ndarray]:
    """GJK-based distance computation (simplified).

    This is a simplified implementation. For production use,
    consider using a proper GJK library.
    """
    # Initial direction from A to B
    direction = prim_b.compute_support(np.array([1, 0, 0])) - prim_a.compute_support(
        np.array([-1, 0, 0])
    )
    if np.linalg.norm(direction) < 1e-10:
        direction = np.array([1.0, 0.0, 0.0])
    else:
        direction = direction / np.linalg.norm(direction)

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
            dist = float(np.linalg.norm(diff))
            return dist, support_a, support_b

        simplex.append(support)

        # Update simplex and direction
        if len(simplex) == 1:
            direction = -simplex[0]
            norm = np.linalg.norm(direction)
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
            dist = float(np.linalg.norm(closest))
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
            dist = float(np.linalg.norm(closest))
            if dist < 1e-6:
                return 0.0, support_a, support_b
            direction = -closest / dist

    # Max iterations reached, estimate distance
    support_a = prim_a.compute_support(direction)
    support_b = prim_b.compute_support(-direction)
    diff = support_b - support_a
    return float(np.linalg.norm(diff)), support_a, support_b


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
