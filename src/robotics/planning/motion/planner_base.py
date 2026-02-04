"""Base classes for motion planners.

This module defines abstract interfaces and common data structures
for sampling-based motion planners.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Protocol

import numpy as np


class PlannerStatus(Enum):
    """Status of planning attempt."""

    SUCCESS = auto()  # Path found
    FAILURE = auto()  # No path found within limits
    TIMEOUT = auto()  # Time limit exceeded
    INVALID_START = auto()  # Start configuration invalid
    INVALID_GOAL = auto()  # Goal configuration invalid


@dataclass
class PlannerConfig:
    """Base configuration for motion planners.

    Attributes:
        max_iterations: Maximum planning iterations.
        max_time: Maximum planning time [s].
        goal_bias: Probability of sampling goal.
        step_size: Maximum step size in configuration space.
        goal_tolerance: Distance tolerance for reaching goal.
        collision_check_resolution: Resolution for path collision checking.
    """

    max_iterations: int = 10000
    max_time: float = 30.0
    goal_bias: float = 0.05
    step_size: float = 0.1
    goal_tolerance: float = 0.01
    collision_check_resolution: int = 10

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        if self.max_time <= 0:
            raise ValueError("max_time must be positive")
        if not 0.0 <= self.goal_bias <= 1.0:
            raise ValueError("goal_bias must be in [0, 1]")
        if self.step_size <= 0:
            raise ValueError("step_size must be positive")
        if self.goal_tolerance <= 0:
            raise ValueError("goal_tolerance must be positive")
        if self.collision_check_resolution < 2:
            raise ValueError("collision_check_resolution must be at least 2")


@dataclass
class PlannerResult:
    """Result of a planning attempt.

    Attributes:
        status: Status of planning attempt.
        path: List of configurations forming the path (empty if failed).
        path_length: Total path length in configuration space.
        num_iterations: Number of iterations performed.
        planning_time: Time spent planning [s].
        num_nodes: Number of nodes in the tree/graph.
        num_collision_checks: Number of collision checks performed.
    """

    status: PlannerStatus
    path: list[np.ndarray] = field(default_factory=list)
    path_length: float = 0.0
    num_iterations: int = 0
    planning_time: float = 0.0
    num_nodes: int = 0
    num_collision_checks: int = 0

    def __post_init__(self) -> None:
        """Validate result."""
        if self.status == PlannerStatus.SUCCESS and len(self.path) == 0:
            raise ValueError("SUCCESS status requires non-empty path")
        if self.path_length < 0:
            raise ValueError("path_length cannot be negative")
        if self.num_iterations < 0:
            raise ValueError("num_iterations cannot be negative")
        if self.planning_time < 0:
            raise ValueError("planning_time cannot be negative")

    @property
    def success(self) -> bool:
        """Check if planning was successful."""
        return self.status == PlannerStatus.SUCCESS

    def get_path_array(self) -> np.ndarray:
        """Get path as numpy array.

        Returns:
            Array of shape (num_waypoints, n_dof).
        """
        if not self.path:
            return np.array([])
        return np.array(self.path)


class CollisionCheckerProtocol(Protocol):
    """Protocol for collision checkers used by planners."""

    def check_collision(self, q: np.ndarray) -> "CollisionResultProtocol":
        """Check if configuration is in collision."""
        ...

    def check_path_collision(
        self,
        q_start: np.ndarray,
        q_end: np.ndarray,
        num_samples: int = 10,
    ) -> tuple[bool, float | None]:
        """Check if path is collision-free."""
        ...


class CollisionResultProtocol(Protocol):
    """Protocol for collision check results."""

    in_collision: bool


class MotionPlanner(ABC):
    """Abstract base class for motion planners.

    Design by Contract:
        Preconditions:
            - collision_checker must implement CollisionCheckerProtocol
            - Configuration bounds must be valid (lower < upper)

        Postconditions:
            - plan() returns PlannerResult with valid status
            - If SUCCESS, path is collision-free

        Invariants:
            - Planner state is reset between plan() calls
    """

    def __init__(
        self,
        collision_checker: CollisionCheckerProtocol,
        config: PlannerConfig | None = None,
    ) -> None:
        """Initialize motion planner.

        Args:
            collision_checker: Collision checking interface.
            config: Planner configuration.
        """
        self._collision_checker = collision_checker
        self._config = config or PlannerConfig()
        self._lower_bounds: np.ndarray | None = None
        self._upper_bounds: np.ndarray | None = None
        self._rng = np.random.default_rng()

    def set_bounds(
        self,
        lower: np.ndarray,
        upper: np.ndarray,
    ) -> None:
        """Set configuration space bounds.

        Args:
            lower: Lower bounds for each DOF.
            upper: Upper bounds for each DOF.

        Raises:
            ValueError: If bounds are invalid.
        """
        lower = np.asarray(lower)
        upper = np.asarray(upper)

        if lower.shape != upper.shape:
            raise ValueError("Bounds must have same shape")
        if not np.all(lower < upper):
            raise ValueError("Lower bounds must be less than upper bounds")

        self._lower_bounds = lower
        self._upper_bounds = upper

    def set_seed(self, seed: int) -> None:
        """Set random seed for reproducibility.

        Args:
            seed: Random seed.
        """
        self._rng = np.random.default_rng(seed)

    @abstractmethod
    def plan(
        self,
        q_start: np.ndarray,
        q_goal: np.ndarray,
    ) -> PlannerResult:
        """Plan a path from start to goal.

        Args:
            q_start: Start configuration.
            q_goal: Goal configuration.

        Returns:
            PlannerResult with path and statistics.
        """
        ...

    def _sample_random(self) -> np.ndarray:
        """Sample random configuration within bounds.

        Returns:
            Random configuration.

        Raises:
            RuntimeError: If bounds not set.
        """
        if self._lower_bounds is None or self._upper_bounds is None:
            raise RuntimeError("Bounds must be set before sampling")
        return self._rng.uniform(self._lower_bounds, self._upper_bounds)

    def _sample_with_goal_bias(self, q_goal: np.ndarray) -> np.ndarray:
        """Sample random configuration with goal bias.

        Args:
            q_goal: Goal configuration.

        Returns:
            Sampled configuration (may be goal).
        """
        if self._rng.random() < self._config.goal_bias:
            return q_goal.copy()
        return self._sample_random()

    def _is_valid(self, q: np.ndarray) -> bool:
        """Check if configuration is valid (collision-free and in bounds).

        Args:
            q: Configuration to check.

        Returns:
            True if configuration is valid.
        """
        # Check bounds
        if self._lower_bounds is not None and self._upper_bounds is not None:
            if np.any(q < self._lower_bounds) or np.any(q > self._upper_bounds):
                return False

        # Check collision
        result = self._collision_checker.check_collision(q)
        return not result.in_collision

    def _steer(
        self,
        q_from: np.ndarray,
        q_to: np.ndarray,
        max_distance: float | None = None,
    ) -> np.ndarray:
        """Steer from one configuration toward another.

        Args:
            q_from: Starting configuration.
            q_to: Target configuration.
            max_distance: Maximum distance to travel.

        Returns:
            New configuration in direction of target.
        """
        max_distance = max_distance or self._config.step_size
        direction = q_to - q_from
        distance = np.linalg.norm(direction)

        if distance <= max_distance:
            return q_to.copy()

        return q_from + (direction / distance) * max_distance

    def _distance(self, q1: np.ndarray, q2: np.ndarray) -> float:
        """Compute distance between configurations.

        Args:
            q1: First configuration.
            q2: Second configuration.

        Returns:
            Euclidean distance.
        """
        return float(np.linalg.norm(q2 - q1))

    def _is_path_valid(
        self,
        q_from: np.ndarray,
        q_to: np.ndarray,
    ) -> bool:
        """Check if path between configurations is collision-free.

        Args:
            q_from: Start configuration.
            q_to: End configuration.

        Returns:
            True if path is collision-free.
        """
        is_free, _ = self._collision_checker.check_path_collision(
            q_from,
            q_to,
            self._config.collision_check_resolution,
        )
        return is_free

    def _compute_path_length(self, path: list[np.ndarray]) -> float:
        """Compute total path length.

        Args:
            path: List of configurations.

        Returns:
            Total path length.
        """
        if len(path) < 2:
            return 0.0

        total = 0.0
        for i in range(len(path) - 1):
            total += self._distance(path[i], path[i + 1])
        return total
