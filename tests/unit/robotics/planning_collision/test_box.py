"""Tests for collision detection module.

Tests cover:
- Geometric primitives (Sphere, Box, Capsule, Cylinder, ConvexHull)
- Distance computation between primitives
- Collision checking
- CollisionChecker configuration-space collision detection
"""

from __future__ import annotations

import numpy as np
import pytest
from src.robotics.planning.collision import (
    Box,
    Capsule,
    CollisionChecker,
    CollisionCheckerConfig,
    CollisionPair,
    CollisionQuery,
    CollisionQueryType,
    CollisionResult,
    ConvexHull,
    Cylinder,
    DistanceResult,
    Sphere,
    check_primitive_collision,
    compute_primitive_distance,
)

# =============================================================================
# Mock Engine for Testing
# =============================================================================


class MockCollisionEngine:
    """Mock engine implementing CollisionCapable protocol."""

    def __init__(self, n_dof: int = 7) -> None:
        """Initialize mock engine."""
        self._n_dof = n_dof
        self._q = np.zeros(n_dof)
        self._v = np.zeros(n_dof)
        self._bodies: dict[str, dict] = {
            "link1": {"position": np.array([0.0, 0.0, 0.5]), "radius": 0.1},
            "link2": {"position": np.array([0.0, 0.0, 1.0]), "radius": 0.1},
            "link3": {"position": np.array([0.0, 0.0, 1.5]), "radius": 0.1},
        }

    def get_state(self) -> tuple[np.ndarray, np.ndarray]:
        """Get current state."""
        return self._q.copy(), self._v.copy()

    def set_state(self, q: np.ndarray, v: np.ndarray) -> None:
        """Set state."""
        self._q = q.copy()
        self._v = v.copy()
        # Update body positions based on q (simplified kinematics)
        self._bodies["link1"]["position"] = np.array([q[0] * 0.1, q[1] * 0.1, 0.5])
        self._bodies["link2"]["position"] = np.array([q[0] * 0.1, q[1] * 0.1, 1.0])
        self._bodies["link3"]["position"] = np.array([q[2] * 0.1, q[3] * 0.1, 1.5])

    def get_body_names(self) -> list[str]:
        """Get body names."""
        return list(self._bodies.keys())

    def get_body_position(self, body_name: str) -> np.ndarray | None:
        """Get body position."""
        if body_name in self._bodies:
            return self._bodies[body_name]["position"].copy()
        return None

    def get_body_collision_geometry(self, body_name: str) -> Sphere | None:
        """Get collision geometry as sphere."""
        if body_name in self._bodies:
            return Sphere(
                center=self._bodies[body_name]["position"],
                radius=self._bodies[body_name]["radius"],
            )
        return None


# =============================================================================
# Sphere Tests
# =============================================================================


# =============================================================================
# Box Tests
# =============================================================================


class TestBox:
    """Tests for Box primitive."""

    def test_create_box(self) -> None:
        """Test box creation."""
        box = Box(
            center=np.array([1.0, 0.0, 0.0]),
            half_extents=np.array([0.5, 0.5, 0.5]),
        )
        assert np.allclose(box.center, [1.0, 0.0, 0.0])
        assert np.allclose(box.half_extents, [0.5, 0.5, 0.5])

    def test_box_invalid_extents(self) -> None:
        """Test that non-positive extents raise error."""
        with pytest.raises(ValueError, match="half_extents must be positive"):
            Box(half_extents=np.array([0.5, -0.1, 0.5]))

    def test_box_aabb_axis_aligned(self) -> None:
        """Test AABB for axis-aligned box."""
        box = Box(
            center=np.array([1.0, 2.0, 3.0]),
            half_extents=np.array([0.5, 0.5, 0.5]),
        )
        min_corner, max_corner = box.get_aabb()
        assert np.allclose(min_corner, [0.5, 1.5, 2.5])
        assert np.allclose(max_corner, [1.5, 2.5, 3.5])

    def test_box_contains_point_inside(self) -> None:
        """Test point inside box."""
        box = Box(center=np.zeros(3), half_extents=np.ones(3))
        assert box.contains_point(np.array([0.5, 0.5, 0.5]))
        assert box.contains_point(np.zeros(3))

    def test_box_contains_point_outside(self) -> None:
        """Test point outside box."""
        box = Box(center=np.zeros(3), half_extents=np.ones(3))
        assert not box.contains_point(np.array([2.0, 0.0, 0.0]))

    def test_box_support(self) -> None:
        """Test support function."""
        box = Box(center=np.zeros(3), half_extents=np.array([1.0, 1.0, 1.0]))
        support = box.compute_support(np.array([1.0, 1.0, 1.0]))
        assert np.allclose(support, [1.0, 1.0, 1.0])


# =============================================================================
# Capsule Tests
# =============================================================================


# =============================================================================
# Cylinder Tests
# =============================================================================


# =============================================================================
# ConvexHull Tests
# =============================================================================


# =============================================================================
# Distance Computation Tests
# =============================================================================


# =============================================================================
# CollisionPair Tests
# =============================================================================


# =============================================================================
# CollisionResult Tests
# =============================================================================


# =============================================================================
# DistanceResult Tests
# =============================================================================


# =============================================================================
# CollisionQuery Tests
# =============================================================================


# =============================================================================
# CollisionChecker Tests
# =============================================================================
