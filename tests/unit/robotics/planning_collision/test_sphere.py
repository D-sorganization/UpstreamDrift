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


class TestSphere:
    """Tests for Sphere primitive."""

    def test_create_sphere(self) -> None:
        """Test sphere creation with valid parameters."""
        sphere = Sphere(center=np.array([1.0, 2.0, 3.0]), radius=0.5)
        assert sphere.radius == 0.5
        assert np.allclose(sphere.center, [1.0, 2.0, 3.0])

    def test_sphere_default_values(self) -> None:
        """Test sphere with default values."""
        sphere = Sphere()
        assert sphere.radius == 1.0
        assert np.allclose(sphere.center, [0.0, 0.0, 0.0])

    def test_sphere_invalid_radius(self) -> None:
        """Test that negative radius raises error."""
        with pytest.raises(ValueError, match="radius must be positive"):
            Sphere(radius=-1.0)

    def test_sphere_invalid_center(self) -> None:
        """Test that invalid center raises error."""
        with pytest.raises(ValueError, match="center must be shape"):
            Sphere(center=np.array([1.0, 2.0]))

    def test_sphere_aabb(self) -> None:
        """Test AABB computation."""
        sphere = Sphere(center=np.array([1.0, 2.0, 3.0]), radius=0.5)
        min_corner, max_corner = sphere.get_aabb()
        assert np.allclose(min_corner, [0.5, 1.5, 2.5])
        assert np.allclose(max_corner, [1.5, 2.5, 3.5])

    def test_sphere_contains_point_inside(self) -> None:
        """Test point inside sphere."""
        sphere = Sphere(center=np.zeros(3), radius=1.0)
        assert sphere.contains_point(np.array([0.5, 0.0, 0.0]))
        assert sphere.contains_point(np.zeros(3))

    def test_sphere_contains_point_outside(self) -> None:
        """Test point outside sphere."""
        sphere = Sphere(center=np.zeros(3), radius=1.0)
        assert not sphere.contains_point(np.array([2.0, 0.0, 0.0]))

    def test_sphere_support(self) -> None:
        """Test support function."""
        sphere = Sphere(center=np.array([1.0, 0.0, 0.0]), radius=0.5)
        direction = np.array([1.0, 0.0, 0.0])
        support = sphere.compute_support(direction)
        assert np.allclose(support, [1.5, 0.0, 0.0])


# =============================================================================
# Box Tests
# =============================================================================


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
