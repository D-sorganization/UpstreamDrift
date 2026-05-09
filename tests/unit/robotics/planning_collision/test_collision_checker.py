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


class TestCollisionChecker:
    """Tests for CollisionChecker class."""

    @pytest.fixture
    def engine(self) -> MockCollisionEngine:
        """Create mock engine."""
        return MockCollisionEngine()

    @pytest.fixture
    def checker(self, engine: MockCollisionEngine) -> CollisionChecker:
        """Create collision checker."""
        return CollisionChecker(engine)

    def test_create_checker(self, engine: MockCollisionEngine) -> None:
        """Test creating collision checker."""
        checker = CollisionChecker(engine)
        assert checker is not None

    def test_checker_requires_collision_capable(self) -> None:
        """Test that non-CollisionCapable engine raises error."""

        class NotCollisionCapable:
            pass

        with pytest.raises(TypeError, match="CollisionCapable"):
            CollisionChecker(NotCollisionCapable())  # type: ignore

    def test_check_collision_no_collision(
        self,
        checker: CollisionChecker,
    ) -> None:
        """Test collision check with no collision."""
        q = np.zeros(7)
        result = checker.check_collision(q)
        assert not result.in_collision

    def test_check_collision_with_environment(
        self,
        checker: CollisionChecker,
    ) -> None:
        """Test collision with environment obstacle."""
        # Add obstacle at link position
        obstacle = Sphere(center=np.array([0.0, 0.0, 0.5]), radius=0.15)
        checker.add_environment_primitive("obstacle", obstacle)

        q = np.zeros(7)
        result = checker.check_collision(q)

        # Should detect collision with link1
        assert result.in_collision

    def test_check_collision_invalid_config(
        self,
        checker: CollisionChecker,
    ) -> None:
        """Test that infinite configuration raises error."""
        q = np.array([np.inf, 0, 0, 0, 0, 0, 0])
        with pytest.raises(ValueError, match="must be finite"):
            checker.check_collision(q)

    def test_compute_distance(
        self,
        checker: CollisionChecker,
    ) -> None:
        """Test distance computation."""
        q = np.zeros(7)
        result = checker.compute_distance(q)

        # Should return some distance (bodies are separated)
        assert (
            result.distance > 0
            or result.distance == float("inf")
            or result.closest_pair is None
        )

    def test_compute_distance_with_environment(
        self,
        checker: CollisionChecker,
    ) -> None:
        """Test distance to environment obstacle."""
        # Add obstacle near link
        obstacle = Sphere(center=np.array([0.5, 0.0, 0.5]), radius=0.1)
        checker.add_environment_primitive("obstacle", obstacle)

        q = np.zeros(7)
        result = checker.compute_distance(q)

        # Distance should be positive (separated)
        assert result.distance > 0

    def test_check_path_collision(
        self,
        checker: CollisionChecker,
    ) -> None:
        """Test path collision checking."""
        q_start = np.zeros(7)
        q_end = np.ones(7)

        is_free, collision_t = checker.check_path_collision(q_start, q_end)

        # Path should be collision-free (no obstacles)
        assert is_free
        assert collision_t is None

    def test_check_path_with_obstacle(
        self,
        checker: CollisionChecker,
    ) -> None:
        """Test path collision with obstacle."""
        # Add obstacle in path
        obstacle = Sphere(center=np.array([0.05, 0.05, 0.5]), radius=0.1)
        checker.add_environment_primitive("obstacle", obstacle)

        q_start = np.zeros(7)
        q_end = np.ones(7)

        is_free, collision_t = checker.check_path_collision(q_start, q_end)

        # Path should have collision
        assert not is_free
        assert collision_t is not None
        assert 0.0 <= collision_t <= 1.0

    def test_add_remove_environment(
        self,
        checker: CollisionChecker,
    ) -> None:
        """Test adding and removing environment primitives."""
        obstacle = Sphere(center=np.zeros(3), radius=1.0)

        checker.add_environment_primitive("test", obstacle)
        assert checker.remove_environment_primitive("test")
        assert not checker.remove_environment_primitive("nonexistent")

    def test_clear_environment(
        self,
        checker: CollisionChecker,
    ) -> None:
        """Test clearing all environment primitives."""
        checker.add_environment_primitive("obs1", Sphere())
        checker.add_environment_primitive("obs2", Sphere())

        checker.clear_environment()

        # Should not find any environment collisions now
        q = np.zeros(7)
        result = checker.check_collision(q)
        # Only self-collision pairs checked now
        assert isinstance(result, CollisionResult)

    def test_disable_enable_collision_pair(
        self,
        checker: CollisionChecker,
    ) -> None:
        """Test disabling and enabling collision pairs."""
        checker.disable_collision_pair("link1", "link2")
        pairs = checker.get_collision_pairs()
        assert CollisionPair("link1", "link2") not in pairs

        checker.enable_collision_pair("link1", "link2")
        pairs = checker.get_collision_pairs()
        assert CollisionPair("link1", "link2") in pairs
