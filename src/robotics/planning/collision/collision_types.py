"""Collision detection type definitions.

This module defines data structures for collision queries and results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto

import numpy as np


class CollisionQueryType(Enum):
    """Type of collision query to perform."""

    BOOLEAN = auto()  # Just check if collision exists
    DISTANCE = auto()  # Compute signed distance
    PENETRATION = auto()  # Compute penetration depth
    CONTACTS = auto()  # Compute contact points


@dataclass
class CollisionPair:
    """A pair of bodies to check for collision.

    Attributes:
        body_a: Name of first body.
        body_b: Name of second body.
        enabled: Whether this pair should be checked.
    """

    body_a: str
    body_b: str
    enabled: bool = True

    def __post_init__(self) -> None:
        """Validate collision pair."""
        if not self.body_a:
            raise ValueError("body_a cannot be empty")
        if not self.body_b:
            raise ValueError("body_b cannot be empty")

    def __hash__(self) -> int:
        """Hash based on sorted body names for order independence."""
        return hash(tuple(sorted([self.body_a, self.body_b])))

    def __eq__(self, other: object) -> bool:
        """Equality based on sorted body names."""
        if not isinstance(other, CollisionPair):
            return NotImplemented
        return set([self.body_a, self.body_b]) == set([other.body_a, other.body_b])


@dataclass
class CollisionResult:
    """Result of a collision check.

    Attributes:
        in_collision: Whether collision was detected.
        collision_pairs: List of colliding body pairs.
        num_contacts: Number of contact points detected.
        computation_time: Time taken for collision check [s].
    """

    in_collision: bool
    collision_pairs: list[CollisionPair] = field(default_factory=list)
    num_contacts: int = 0
    computation_time: float = 0.0

    def __post_init__(self) -> None:
        """Validate result consistency."""
        if self.in_collision and len(self.collision_pairs) == 0:
            raise ValueError("in_collision=True but no collision pairs provided")
        if not self.in_collision and len(self.collision_pairs) > 0:
            raise ValueError("in_collision=False but collision pairs exist")
        if self.num_contacts < 0:
            raise ValueError("num_contacts cannot be negative")
        if self.computation_time < 0:
            raise ValueError("computation_time cannot be negative")


@dataclass
class DistanceResult:
    """Result of a distance query.

    Attributes:
        distance: Minimum signed distance (negative = penetration).
        closest_pair: Pair of bodies with minimum distance.
        point_a: Closest point on body A in world frame.
        point_b: Closest point on body B in world frame.
        normal: Contact normal pointing from A to B.
        computation_time: Time taken for distance computation [s].
    """

    distance: float
    closest_pair: CollisionPair | None = None
    point_a: np.ndarray | None = None
    point_b: np.ndarray | None = None
    normal: np.ndarray | None = None
    computation_time: float = 0.0

    def __post_init__(self) -> None:
        """Validate distance result."""
        if self.computation_time < 0:
            raise ValueError("computation_time cannot be negative")
        if self.point_a is not None and self.point_a.shape != (3,):
            raise ValueError("point_a must be shape (3,)")
        if self.point_b is not None and self.point_b.shape != (3,):
            raise ValueError("point_b must be shape (3,)")
        if self.normal is not None:
            if self.normal.shape != (3,):
                raise ValueError("normal must be shape (3,)")
            # Normalize the normal vector
            norm = np.linalg.norm(self.normal)
            if norm > 1e-10:
                object.__setattr__(self, "normal", self.normal / norm)

    @property
    def in_collision(self) -> bool:
        """Check if distance indicates collision."""
        return self.distance < 0.0

    @property
    def penetration_depth(self) -> float:
        """Get penetration depth (0 if not in collision)."""
        return max(0.0, -self.distance)


@dataclass
class CollisionQuery:
    """Configuration for a collision query.

    Attributes:
        query_type: Type of query to perform.
        include_pairs: Specific pairs to check (None = all pairs).
        exclude_pairs: Pairs to exclude from checking.
        max_distance: Maximum distance to compute (for distance queries).
        compute_contacts: Whether to compute contact points.
        early_exit: Stop on first collision (for boolean queries).
    """

    query_type: CollisionQueryType = CollisionQueryType.BOOLEAN
    include_pairs: list[CollisionPair] | None = None
    exclude_pairs: list[CollisionPair] = field(default_factory=list)
    max_distance: float = float("inf")
    compute_contacts: bool = False
    early_exit: bool = True

    def __post_init__(self) -> None:
        """Validate query configuration."""
        if self.max_distance <= 0:
            raise ValueError("max_distance must be positive")

    def should_check_pair(self, pair: CollisionPair) -> bool:
        """Determine if a pair should be checked.

        Args:
            pair: Collision pair to evaluate.

        Returns:
            True if pair should be included in query.
        """
        # Check exclusion list first
        if pair in self.exclude_pairs:
            return False
        # If inclusion list specified, pair must be in it
        if self.include_pairs is not None:
            return pair in self.include_pairs
        return True
