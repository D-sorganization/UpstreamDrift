from __future__ import annotations

from abc import ABC, abstractmethod

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

    def compute_support_batch(self, directions: np.ndarray) -> np.ndarray:
        """Compute support points for many directions at once.

        Subclasses override this with a vectorised implementation; the default
        simply loops over :meth:`compute_support`.  Overrides must agree with
        :meth:`compute_support` row for row.

        Design by Contract:
            Preconditions:
                - directions has shape (n, 3)
            Postconditions:
                - result[i] == compute_support(directions[i])

        Args:
            directions: Array of directions, shape (n, 3).

        Returns:
            Support points, shape (n, 3).
        """
        directions = np.asarray(directions, dtype=np.float64)
        if directions.ndim != 2 or directions.shape[1] != 3:
            raise ValueError("directions must have shape (n, 3)")
        return np.array([self.compute_support(d) for d in directions])
