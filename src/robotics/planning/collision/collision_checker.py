"""Collision checker for motion planning.

This module provides configuration-space collision checking
for robot motion planning.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np

from src.robotics.planning.collision.collision_types import (
    CollisionPair,
    CollisionQuery,
    CollisionQueryType,
    CollisionResult,
    DistanceResult,
)
from src.robotics.planning.collision.geometric_primitives import (
    GeometricPrimitive,
    Sphere,
    compute_primitive_distance,
)


@runtime_checkable
class CollisionCapable(Protocol):
    """Protocol for engines with collision checking support."""

    def get_state(self) -> tuple[np.ndarray, np.ndarray]:
        """Get current (q, v) state."""
        ...

    def set_state(self, q: np.ndarray, v: np.ndarray) -> None:
        """Set robot state."""
        ...

    def get_body_names(self) -> list[str]:
        """Get list of body names."""
        ...

    def get_body_position(self, body_name: str) -> np.ndarray | None:
        """Get body position in world frame."""
        ...

    def get_body_collision_geometry(
        self,
        body_name: str,
    ) -> GeometricPrimitive | None:
        """Get collision geometry for body."""
        ...


@dataclass
class CollisionCheckerConfig:
    """Configuration for collision checker.

    Attributes:
        default_margin: Default collision margin [m].
        use_broad_phase: Enable AABB broad-phase filtering.
        max_contacts: Maximum contacts to return per query.
        self_collision_pairs: Pairs to check for self-collision.
        disabled_pairs: Pairs to skip (adjacent links, etc).
        environment_bodies: Names of static environment bodies.
    """

    default_margin: float = 0.01
    use_broad_phase: bool = True
    max_contacts: int = 100
    self_collision_pairs: list[CollisionPair] = field(default_factory=list)
    disabled_pairs: list[CollisionPair] = field(default_factory=list)
    environment_bodies: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.default_margin < 0:
            raise ValueError("default_margin must be non-negative")
        if self.max_contacts <= 0:
            raise ValueError("max_contacts must be positive")


class CollisionChecker:
    """Configuration-space collision checker.

    Design by Contract:
        Preconditions:
            - Engine must implement CollisionCapable protocol
            - Configuration arrays must match robot DOF

        Postconditions:
            - check_collision returns accurate collision status
            - compute_distance returns valid signed distance

        Invariants:
            - Does not modify engine state (restores original)

    Example:
        >>> checker = CollisionChecker(engine)
        >>> is_safe = not checker.check_collision(q_desired)
        >>> if is_safe:
        ...     engine.set_state(q_desired, np.zeros_like(q_desired))
    """

    def __init__(
        self,
        engine: CollisionCapable,
        config: CollisionCheckerConfig | None = None,
    ) -> None:
        """Initialize collision checker.

        Args:
            engine: Physics engine with collision support.
            config: Collision checker configuration.

        Raises:
            TypeError: If engine doesn't implement CollisionCapable.
        """
        if not isinstance(engine, CollisionCapable):
            raise TypeError("Engine must implement CollisionCapable protocol")

        self._engine = engine
        self._config = config or CollisionCheckerConfig()
        self._collision_pairs: list[CollisionPair] = []
        self._environment_primitives: dict[str, GeometricPrimitive] = {}

        self._setup_collision_pairs()

    def _setup_collision_pairs(self) -> None:
        """Set up default collision pairs."""
        body_names = self._engine.get_body_names()

        # Add self-collision pairs (all pairs except disabled)
        for i, body_a in enumerate(body_names):
            for body_b in body_names[i + 1 :]:
                pair = CollisionPair(body_a, body_b)
                if pair not in self._config.disabled_pairs:
                    self._collision_pairs.append(pair)

        # Add explicit self-collision pairs from config
        for pair in self._config.self_collision_pairs:
            if pair not in self._collision_pairs:
                self._collision_pairs.append(pair)

    def add_environment_primitive(
        self,
        name: str,
        primitive: GeometricPrimitive,
    ) -> None:
        """Add static environment collision geometry.

        Args:
            name: Unique name for the primitive.
            primitive: Geometric primitive for collision checking.
        """
        if not name:
            raise ValueError("name cannot be empty")
        self._environment_primitives[name] = primitive

    def remove_environment_primitive(self, name: str) -> bool:
        """Remove environment primitive by name.

        Args:
            name: Name of primitive to remove.

        Returns:
            True if primitive was removed, False if not found.
        """
        if name in self._environment_primitives:
            del self._environment_primitives[name]
            return True
        return False

    def clear_environment(self) -> None:
        """Remove all environment primitives."""
        self._environment_primitives.clear()

    def check_collision(
        self,
        q: np.ndarray,
        query: CollisionQuery | None = None,
    ) -> CollisionResult:
        """Check for collisions at configuration.

        Design by Contract:
            Preconditions:
                - q.shape matches robot DOF
                - All values in q are finite

            Postconditions:
                - Returns CollisionResult with accurate status
                - Engine state is restored to original

        Args:
            q: Robot configuration to check.
            query: Optional query configuration.

        Returns:
            CollisionResult with collision status and details.
        """
        q = np.asarray(q)
        if not np.all(np.isfinite(q)):
            raise ValueError("Configuration must be finite")

        query = query or CollisionQuery()
        start_time = time.perf_counter()

        # Save current state
        q_orig, v_orig = self._engine.get_state()

        try:
            # Set configuration
            self._engine.set_state(q, np.zeros_like(q))

            colliding_pairs: list[CollisionPair] = []
            num_contacts = 0

            # Check self-collisions
            for pair in self._collision_pairs:
                if not query.should_check_pair(pair):
                    continue

                if self._check_pair_collision(pair, self._config.default_margin):
                    colliding_pairs.append(pair)
                    num_contacts += 1

                    if query.early_exit:
                        break

            # Check environment collisions
            if not (query.early_exit and colliding_pairs):
                body_names = self._engine.get_body_names()
                for body_name in body_names:
                    for env_name, env_prim in self._environment_primitives.items():
                        env_pair = CollisionPair(body_name, env_name)
                        if not query.should_check_pair(env_pair):
                            continue

                        if self._check_body_environment_collision(
                            body_name, env_prim, self._config.default_margin
                        ):
                            colliding_pairs.append(env_pair)
                            num_contacts += 1

                            if query.early_exit:
                                break
                    if query.early_exit and colliding_pairs:
                        break

            computation_time = time.perf_counter() - start_time

            return CollisionResult(
                in_collision=len(colliding_pairs) > 0,
                collision_pairs=colliding_pairs,
                num_contacts=num_contacts,
                computation_time=computation_time,
            )

        finally:
            # Restore original state
            self._engine.set_state(q_orig, v_orig)

    def _check_pair_collision(
        self,
        pair: CollisionPair,
        margin: float,
    ) -> bool:
        """Check collision between body pair."""
        geom_a = self._engine.get_body_collision_geometry(pair.body_a)
        geom_b = self._engine.get_body_collision_geometry(pair.body_b)

        if geom_a is None or geom_b is None:
            return False

        # Broad phase: AABB check
        if self._config.use_broad_phase:
            if not self._aabb_overlap(geom_a, geom_b, margin):
                return False

        # Narrow phase: Actual distance check
        distance, _, _ = compute_primitive_distance(geom_a, geom_b)
        return distance <= margin

    def _check_body_environment_collision(
        self,
        body_name: str,
        env_primitive: GeometricPrimitive,
        margin: float,
    ) -> bool:
        """Check collision between robot body and environment."""
        body_geom = self._engine.get_body_collision_geometry(body_name)
        if body_geom is None:
            return False

        # Broad phase
        if self._config.use_broad_phase:
            if not self._aabb_overlap(body_geom, env_primitive, margin):
                return False

        # Narrow phase
        distance, _, _ = compute_primitive_distance(body_geom, env_primitive)
        return distance <= margin

    def _aabb_overlap(
        self,
        prim_a: GeometricPrimitive,
        prim_b: GeometricPrimitive,
        margin: float,
    ) -> bool:
        """Check if AABBs overlap (with margin)."""
        min_a, max_a = prim_a.get_aabb()
        min_b, max_b = prim_b.get_aabb()

        # Expand by margin
        min_a = min_a - margin
        max_a = max_a + margin

        # Check overlap
        return bool(
            np.all(max_a >= min_b) and np.all(max_b >= min_a)
        )

    def compute_distance(
        self,
        q: np.ndarray,
        query: CollisionQuery | None = None,
    ) -> DistanceResult:
        """Compute minimum distance to collision at configuration.

        Design by Contract:
            Preconditions:
                - q.shape matches robot DOF
                - All values in q are finite

            Postconditions:
                - Returns DistanceResult with signed distance
                - Negative distance indicates penetration
                - Engine state is restored to original

        Args:
            q: Robot configuration to check.
            query: Optional query configuration.

        Returns:
            DistanceResult with minimum distance and witness points.
        """
        q = np.asarray(q)
        if not np.all(np.isfinite(q)):
            raise ValueError("Configuration must be finite")

        query = query or CollisionQuery(query_type=CollisionQueryType.DISTANCE)
        start_time = time.perf_counter()

        # Save current state
        q_orig, v_orig = self._engine.get_state()

        try:
            self._engine.set_state(q, np.zeros_like(q))

            min_distance = float("inf")
            closest_pair: CollisionPair | None = None
            point_a: np.ndarray | None = None
            point_b: np.ndarray | None = None
            normal: np.ndarray | None = None

            # Check self-collision pairs
            for pair in self._collision_pairs:
                if not query.should_check_pair(pair):
                    continue

                dist, pa, pb = self._compute_pair_distance(pair)
                if dist < min_distance:
                    min_distance = dist
                    closest_pair = pair
                    point_a = pa
                    point_b = pb
                    if np.linalg.norm(pb - pa) > 1e-10:
                        normal = (pb - pa) / np.linalg.norm(pb - pa)

                    # Early exit if max_distance exceeded
                    if min_distance > query.max_distance:
                        break

            # Check environment
            body_names = self._engine.get_body_names()
            for body_name in body_names:
                for env_name, env_prim in self._environment_primitives.items():
                    env_pair = CollisionPair(body_name, env_name)
                    if not query.should_check_pair(env_pair):
                        continue

                    dist, pa, pb = self._compute_body_environment_distance(
                        body_name, env_prim
                    )
                    if dist < min_distance:
                        min_distance = dist
                        closest_pair = env_pair
                        point_a = pa
                        point_b = pb
                        if np.linalg.norm(pb - pa) > 1e-10:
                            normal = (pb - pa) / np.linalg.norm(pb - pa)

            computation_time = time.perf_counter() - start_time

            return DistanceResult(
                distance=min_distance if min_distance < float("inf") else 0.0,
                closest_pair=closest_pair,
                point_a=point_a,
                point_b=point_b,
                normal=normal,
                computation_time=computation_time,
            )

        finally:
            self._engine.set_state(q_orig, v_orig)

    def _compute_pair_distance(
        self,
        pair: CollisionPair,
    ) -> tuple[float, np.ndarray, np.ndarray]:
        """Compute distance between body pair."""
        geom_a = self._engine.get_body_collision_geometry(pair.body_a)
        geom_b = self._engine.get_body_collision_geometry(pair.body_b)

        if geom_a is None or geom_b is None:
            return float("inf"), np.zeros(3), np.zeros(3)

        return compute_primitive_distance(geom_a, geom_b)

    def _compute_body_environment_distance(
        self,
        body_name: str,
        env_primitive: GeometricPrimitive,
    ) -> tuple[float, np.ndarray, np.ndarray]:
        """Compute distance between body and environment primitive."""
        body_geom = self._engine.get_body_collision_geometry(body_name)
        if body_geom is None:
            return float("inf"), np.zeros(3), np.zeros(3)

        return compute_primitive_distance(body_geom, env_primitive)

    def check_path_collision(
        self,
        q_start: np.ndarray,
        q_end: np.ndarray,
        num_samples: int = 10,
    ) -> tuple[bool, float | None]:
        """Check if path between configurations is collision-free.

        Uses linear interpolation with discrete samples.

        Args:
            q_start: Start configuration.
            q_end: End configuration.
            num_samples: Number of samples along path.

        Returns:
            Tuple of (is_collision_free, collision_parameter).
            collision_parameter is t in [0,1] where collision occurs, or None.
        """
        if num_samples < 2:
            raise ValueError("num_samples must be at least 2")

        q_start = np.asarray(q_start)
        q_end = np.asarray(q_end)

        for i in range(num_samples):
            t = i / (num_samples - 1)
            q = q_start + t * (q_end - q_start)

            result = self.check_collision(q)
            if result.in_collision:
                return False, t

        return True, None

    def get_collision_pairs(self) -> list[CollisionPair]:
        """Get list of collision pairs being checked."""
        return self._collision_pairs.copy()

    def disable_collision_pair(self, body_a: str, body_b: str) -> None:
        """Disable collision checking between two bodies.

        Args:
            body_a: First body name.
            body_b: Second body name.
        """
        pair = CollisionPair(body_a, body_b)
        if pair in self._collision_pairs:
            self._collision_pairs.remove(pair)
        if pair not in self._config.disabled_pairs:
            self._config.disabled_pairs.append(pair)

    def enable_collision_pair(self, body_a: str, body_b: str) -> None:
        """Enable collision checking between two bodies.

        Args:
            body_a: First body name.
            body_b: Second body name.
        """
        pair = CollisionPair(body_a, body_b)
        if pair in self._config.disabled_pairs:
            self._config.disabled_pairs.remove(pair)
        if pair not in self._collision_pairs:
            self._collision_pairs.append(pair)
