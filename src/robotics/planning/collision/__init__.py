"""Collision detection module for motion planning.

This module provides collision checking capabilities:
- Geometric primitive collision detection
- Configuration space collision checking
- Distance and penetration queries
- Swept volume collision checking

Example:
    >>> from src.robotics.planning.collision import CollisionChecker
    >>>
    >>> checker = CollisionChecker(engine)
    >>> is_collision = checker.check_collision(q)
    >>> distance = checker.compute_distance(q)
"""

from __future__ import annotations

from src.robotics.planning.collision.collision_checker import (
    CollisionChecker,
    CollisionCheckerConfig,
)
from src.robotics.planning.collision.collision_types import (
    CollisionPair,
    CollisionQuery,
    CollisionQueryType,
    CollisionResult,
    DistanceResult,
)
from src.robotics.planning.collision.geometric_primitives import (
    Box,
    Capsule,
    ConvexHull,
    Cylinder,
    GeometricPrimitive,
    Sphere,
    check_primitive_collision,
    compute_primitive_distance,
)

__all__ = [
    # Types
    "CollisionPair",
    "CollisionQueryType",
    "CollisionResult",
    "DistanceResult",
    "CollisionQuery",
    # Checker
    "CollisionChecker",
    "CollisionCheckerConfig",
    # Primitives
    "GeometricPrimitive",
    "Sphere",
    "Box",
    "Capsule",
    "Cylinder",
    "ConvexHull",
    "compute_primitive_distance",
    "check_primitive_collision",
]
