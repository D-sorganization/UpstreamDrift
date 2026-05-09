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


class TestIssue2499CollisionDistanceEarlyExit:
    """Issue #2499: compute_distance must not exit before checking closer pairs."""

    @pytest.fixture
    def engine_with_spread_pairs(self) -> MockCollisionEngine:
        """Engine where (link1, link2) is far but (link1, link3) is close."""
        engine = MockCollisionEngine()
        # link1 at z=0.5, link2 far at z=5.0, link3 close at z=0.65
        engine._bodies["link1"]["position"] = np.array([0.0, 0.0, 0.5])
        engine._bodies["link2"]["position"] = np.array([0.0, 0.0, 5.0])
        engine._bodies["link3"]["position"] = np.array([0.0, 0.0, 0.65])
        return engine

    def test_compute_distance_finds_closer_pair_after_far_pair(
        self,
        engine_with_spread_pairs: MockCollisionEngine,
    ) -> None:
        """compute_distance must check all pairs, not exit when first is far."""
        checker = CollisionChecker(engine_with_spread_pairs)
        # Both pairs auto-setup; max_distance below link1-link2 gap but above link1-link3
        q = np.zeros(7)
        query = CollisionQuery(
            query_type=CollisionQueryType.DISTANCE,
            max_distance=2.0,
        )
        result_limited = checker.compute_distance(q, query=query)
        result_unlimited = checker.compute_distance(q)

        # Both must find the same closest pair (link1 ↔ link3)
        assert result_limited.closest_pair is not None
        assert result_unlimited.closest_pair is not None
        lim_names = {
            result_limited.closest_pair.body_a,
            result_limited.closest_pair.body_b,
        }
        unl_names = {
            result_unlimited.closest_pair.body_a,
            result_unlimited.closest_pair.body_b,
        }
        assert lim_names == unl_names

    def test_compute_distance_consistent_with_and_without_max_distance(
        self,
        engine_with_spread_pairs: MockCollisionEngine,
    ) -> None:
        """compute_distance result must not change just because max_distance is set."""
        checker = CollisionChecker(engine_with_spread_pairs)
        q = np.zeros(7)
        result_no_limit = checker.compute_distance(q)
        result_with_limit = checker.compute_distance(
            q,
            query=CollisionQuery(
                query_type=CollisionQueryType.DISTANCE, max_distance=2.0
            ),
        )
        assert abs(result_no_limit.distance - result_with_limit.distance) < 1e-6
