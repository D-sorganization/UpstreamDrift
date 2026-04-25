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
