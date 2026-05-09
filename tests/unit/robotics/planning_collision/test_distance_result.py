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


class TestDistanceResult:
    """Tests for DistanceResult dataclass."""

    @pytest.mark.parametrize(
        "distance, in_collision, penetration_depth",
        [
            (0.5, False, 0.0),
            (-0.1, True, 0.1),
            (0.0, False, 0.0),
        ],
        ids=["separated", "penetrating", "touching"],
    )
    def test_distance_result_properties(
        self,
        distance: float,
        in_collision: bool,
        penetration_depth: float,
    ) -> None:
        """Test DistanceResult computed properties."""
        result = DistanceResult(distance=distance)
        assert result.distance == distance
        assert result.in_collision == in_collision
        assert result.penetration_depth == pytest.approx(penetration_depth)


# =============================================================================
# CollisionQuery Tests
# =============================================================================


# =============================================================================
# CollisionChecker Tests
# =============================================================================
