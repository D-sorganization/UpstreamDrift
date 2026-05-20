from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ._primitives_base import GeometricPrimitive


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
        if point is None:
            raise ValueError("point must be provided")
        point = np.asarray(point)
        diff = np.ravel(point - self.center)
        return math.hypot(*np.ravel(diff)) <= self.radius

    def compute_support(self, direction: np.ndarray) -> np.ndarray:
        """Compute support point."""
        if direction is None:
            raise ValueError("direction must be provided")
        direction = np.asarray(direction)
        norm = math.hypot(*np.ravel(direction))
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
        if point is None:
            raise ValueError("point must be provided")
        point = np.asarray(point)
        # Transform to local frame
        local_point = self.rotation.T @ (point - self.center)
        return bool(np.all(np.abs(local_point) <= self.half_extents))

    def compute_support(self, direction: np.ndarray) -> np.ndarray:
        """Compute support point."""
        if direction is None:
            raise ValueError("direction must be provided")
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
        return math.hypot(*np.ravel(self.point_b - self.point_a))

    @property
    def axis(self) -> np.ndarray:
        """Get capsule axis direction (normalized)."""
        diff = self.point_b - self.point_a
        length = math.hypot(*np.ravel(diff))
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
        if point is None:
            raise ValueError("point must be provided")
        ab = self.point_b - self.point_a
        t = np.dot(point - self.point_a, ab) / (np.dot(ab, ab) + 1e-10)
        t = np.clip(t, 0.0, 1.0)
        return self.point_a + t * ab

    def contains_point(self, point: np.ndarray) -> bool:
        """Check if point is inside capsule."""
        if point is None:
            raise ValueError("point must be provided")
        point = np.asarray(point)
        closest = self._closest_point_on_segment(point)
        diff = np.ravel(point - closest)
        return math.hypot(*np.ravel(diff)) <= self.radius

    def compute_support(self, direction: np.ndarray) -> np.ndarray:
        """Compute support point."""
        if direction is None:
            raise ValueError("direction must be provided")
        direction = np.asarray(direction)
        norm = math.hypot(*np.ravel(direction))
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
        diff = np.ravel(self.axis)
        norm = math.hypot(*np.ravel(diff))
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
        if point is None:
            raise ValueError("point must be provided")
        point = np.asarray(point)
        # Project onto axis
        to_point = point - self.center
        along_axis = np.dot(to_point, self.axis)

        # Check height
        if abs(along_axis) > self.half_height:
            return False

        # Check radius (perpendicular distance)
        perp = to_point - along_axis * self.axis
        return math.hypot(*np.ravel(perp)) <= self.radius

    def compute_support(self, direction: np.ndarray) -> np.ndarray:
        """Compute support point."""
        if direction is None:
            raise ValueError("direction must be provided")
        direction = np.asarray(direction)
        norm = math.hypot(*np.ravel(direction))
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
        perp_norm = math.hypot(*np.ravel(d_perp))
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
        if point is None:
            raise ValueError("point must be provided")
        point = np.asarray(point)
        # Simple heuristic: point is inside if closer to center than
        # all vertices in the same direction
        to_point = point - self.center
        norm = math.hypot(*np.ravel(to_point))
        if norm < 1e-10:
            return True  # At center

        direction = to_point / norm
        support = self.compute_support(direction)
        support_dist = np.dot(support - self.center, direction)
        return norm <= support_dist

    def compute_support(self, direction: np.ndarray) -> np.ndarray:
        """Compute support point."""
        if direction is None:
            raise ValueError("direction must be provided")
        direction = np.asarray(direction)
        # Find vertex with maximum dot product
        dots = self.vertices @ direction
        idx = np.argmax(dots)
        return self.vertices[idx].copy()
