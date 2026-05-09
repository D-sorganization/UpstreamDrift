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


class TestPrimitiveDistance:
    """Tests for distance computation between primitives."""

    @pytest.mark.parametrize(
        "center_b, expected_distance",
        [
            (np.array([3.0, 0.0, 0.0]), 1.0),
            (np.array([2.0, 0.0, 0.0]), 0.0),
            (np.array([1.0, 0.0, 0.0]), -1.0),
        ],
        ids=["separated", "touching", "overlapping"],
    )
    def test_sphere_sphere_distance(
        self, center_b: np.ndarray, expected_distance: float
    ) -> None:
        """Test distance between sphere pairs at varying separation."""
        sphere_a = Sphere(center=np.array([0.0, 0.0, 0.0]), radius=1.0)
        sphere_b = Sphere(center=center_b, radius=1.0)

        distance, _, _ = compute_primitive_distance(sphere_a, sphere_b)

        assert distance == pytest.approx(expected_distance, abs=1e-6)

    def test_sphere_capsule_distance(self) -> None:
        """Test distance between sphere and capsule."""
        sphere = Sphere(center=np.array([2.0, 0.0, 0.5]), radius=0.5)
        capsule = Capsule(
            point_a=np.array([0.0, 0.0, 0.0]),
            point_b=np.array([0.0, 0.0, 1.0]),
            radius=0.5,
        )

        distance, _, _ = compute_primitive_distance(sphere, capsule)

        # Distance should be 2.0 - 0.5 - 0.5 = 1.0
        assert distance == pytest.approx(1.0, abs=1e-6)

    def test_capsule_capsule_parallel(self) -> None:
        """Test distance between parallel capsules."""
        cap_a = Capsule(
            point_a=np.array([0.0, 0.0, 0.0]),
            point_b=np.array([0.0, 0.0, 1.0]),
            radius=0.1,
        )
        cap_b = Capsule(
            point_a=np.array([1.0, 0.0, 0.0]),
            point_b=np.array([1.0, 0.0, 1.0]),
            radius=0.1,
        )

        distance, _, _ = compute_primitive_distance(cap_a, cap_b)

        # Distance between axes is 1.0, subtract radii
        assert distance == pytest.approx(0.8, abs=1e-6)


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
