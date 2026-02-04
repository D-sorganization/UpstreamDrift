"""RRT* (Optimal RRT) motion planner.

This module implements the RRT* algorithm which extends RRT with
asymptotically optimal path planning via rewiring.

Reference:
    Karaman, S., & Frazzoli, E. (2011). Sampling-based algorithms
    for optimal motion planning. IJRR.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

from src.robotics.planning.motion.planner_base import (
    CollisionCheckerProtocol,
    MotionPlanner,
    PlannerConfig,
    PlannerResult,
    PlannerStatus,
)
from src.robotics.planning.motion.rrt import TreeNode


@dataclass
class RRTStarConfig(PlannerConfig):
    """Configuration for RRT* planner.

    Additional Attributes:
        rewire_radius: Radius for rewiring neighbors (None = auto).
        rewire_factor: Factor for computing rewire radius.
        use_kd_tree: Use KD-tree for nearest neighbor (future).
    """

    rewire_radius: float | None = None
    rewire_factor: float = 1.1
    use_kd_tree: bool = False

    def __post_init__(self) -> None:
        """Validate configuration."""
        super().__post_init__()
        if self.rewire_radius is not None and self.rewire_radius <= 0:
            raise ValueError("rewire_radius must be positive")
        if self.rewire_factor <= 0:
            raise ValueError("rewire_factor must be positive")


class RRTStarPlanner(MotionPlanner):
    """RRT* (optimal RRT) motion planner.

    RRT* extends basic RRT with:
    1. Near-neighbor search for finding potential parent nodes
    2. Rewiring to maintain asymptotically optimal paths

    Design by Contract:
        Preconditions:
            - Start and goal configurations must be valid
            - Bounds must be set before planning

        Postconditions:
            - If SUCCESS, returned path is collision-free
            - Path cost approaches optimal as iterations increase

        Invariants:
            - Tree maintains optimal-cost parent for each node

    Example:
        >>> planner = RRTStarPlanner(collision_checker)
        >>> planner.set_bounds(lower, upper)
        >>> result = planner.plan(q_start, q_goal)
        >>> print(f"Path length: {result.path_length}")
    """

    def __init__(
        self,
        collision_checker: CollisionCheckerProtocol,
        config: RRTStarConfig | None = None,
    ) -> None:
        """Initialize RRT* planner.

        Args:
            collision_checker: Collision checking interface.
            config: RRT* configuration.
        """
        super().__init__(collision_checker, config or RRTStarConfig())
        self._config: RRTStarConfig = self._config  # type: ignore
        self._nodes: list[TreeNode] = []
        self._num_collision_checks = 0
        self._dimension = 0

    def plan(
        self,
        q_start: np.ndarray,
        q_goal: np.ndarray,
    ) -> PlannerResult:
        """Plan an optimal path using RRT*.

        Args:
            q_start: Start configuration.
            q_goal: Goal configuration.

        Returns:
            PlannerResult with path and statistics.
        """
        q_start = np.asarray(q_start)
        q_goal = np.asarray(q_goal)
        self._dimension = len(q_start)

        # Reset state
        self._nodes = []
        self._num_collision_checks = 0
        start_time = time.perf_counter()

        # Validate start and goal
        if not self._is_valid(q_start):
            return PlannerResult(
                status=PlannerStatus.INVALID_START,
                planning_time=time.perf_counter() - start_time,
            )

        if not self._is_valid(q_goal):
            return PlannerResult(
                status=PlannerStatus.INVALID_GOAL,
                planning_time=time.perf_counter() - start_time,
            )

        # Initialize tree with start
        self._nodes.append(TreeNode(config=q_start.copy(), parent_idx=-1, cost=0.0))

        goal_idx = -1
        best_goal_cost = float("inf")
        iterations = 0

        while iterations < self._config.max_iterations:
            # Check time limit
            if time.perf_counter() - start_time > self._config.max_time:
                break

            iterations += 1

            # Sample random configuration with goal bias
            q_rand = self._sample_with_goal_bias(q_goal)

            # Find nearest node in tree
            nearest_idx = self._find_nearest(q_rand)
            q_nearest = self._nodes[nearest_idx].config

            # Steer toward random sample
            q_new = self._steer(q_nearest, q_rand)

            # Check if new configuration is valid
            self._num_collision_checks += 1
            if not self._is_valid(q_new):
                continue

            # Find near neighbors for RRT* rewiring
            near_indices = self._find_near(q_new)

            # Choose best parent from near neighbors
            best_parent_idx = self._choose_parent(q_new, near_indices)
            if best_parent_idx < 0:
                continue

            # Add new node to tree
            parent_node = self._nodes[best_parent_idx]
            new_cost = parent_node.cost + self._distance(parent_node.config, q_new)
            new_node = TreeNode(
                config=q_new.copy(),
                parent_idx=best_parent_idx,
                cost=new_cost,
            )
            new_idx = len(self._nodes)
            self._nodes.append(new_node)

            # Rewire tree
            self._rewire(new_idx, near_indices)

            # Check if goal reached
            dist_to_goal = self._distance(q_new, q_goal)
            if dist_to_goal <= self._config.goal_tolerance:
                # Try to connect to goal
                self._num_collision_checks += self._config.collision_check_resolution
                if self._is_path_valid(q_new, q_goal):
                    goal_cost = new_cost + dist_to_goal
                    if goal_cost < best_goal_cost:
                        # Update or add goal node
                        if goal_idx >= 0:
                            self._nodes[goal_idx].parent_idx = new_idx
                            self._nodes[goal_idx].cost = goal_cost
                        else:
                            goal_node = TreeNode(
                                config=q_goal.copy(),
                                parent_idx=new_idx,
                                cost=goal_cost,
                            )
                            goal_idx = len(self._nodes)
                            self._nodes.append(goal_node)
                        best_goal_cost = goal_cost

        planning_time = time.perf_counter() - start_time

        # Extract path if goal reached
        if goal_idx >= 0:
            path = self._extract_path(goal_idx)
            return PlannerResult(
                status=PlannerStatus.SUCCESS,
                path=path,
                path_length=self._compute_path_length(path),
                num_iterations=iterations,
                planning_time=planning_time,
                num_nodes=len(self._nodes),
                num_collision_checks=self._num_collision_checks,
            )

        status = (
            PlannerStatus.TIMEOUT
            if time.perf_counter() - start_time >= self._config.max_time
            else PlannerStatus.FAILURE
        )
        return PlannerResult(
            status=status,
            num_iterations=iterations,
            planning_time=planning_time,
            num_nodes=len(self._nodes),
            num_collision_checks=self._num_collision_checks,
        )

    def _compute_rewire_radius(self) -> float:
        """Compute the rewiring radius based on tree size.

        Uses the formula from RRT* paper:
        r = gamma * (log(n)/n)^(1/d)

        Returns:
            Rewiring radius.
        """
        if self._config.rewire_radius is not None:
            return self._config.rewire_radius

        n = len(self._nodes)
        if n < 2:
            return self._config.step_size * 2

        d = self._dimension
        # Compute volume of unit ball in d dimensions
        unit_ball_vol = (np.pi ** (d / 2)) / math.gamma(d / 2 + 1)

        # Get configuration space volume
        if self._lower_bounds is not None and self._upper_bounds is not None:
            space_vol = np.prod(self._upper_bounds - self._lower_bounds)
        else:
            space_vol = 1.0

        # Compute gamma
        gamma = self._config.rewire_factor * 2 * (1 + 1 / d) ** (1 / d) * (
            space_vol / unit_ball_vol
        ) ** (1 / d)

        # Compute radius
        radius = min(
            gamma * (np.log(n) / n) ** (1 / d),
            self._config.step_size * 3,
        )
        return max(radius, self._config.step_size)

    def _find_nearest(self, q: np.ndarray) -> int:
        """Find index of nearest node in tree.

        Args:
            q: Query configuration.

        Returns:
            Index of nearest node.
        """
        min_dist = float("inf")
        min_idx = 0

        for i, node in enumerate(self._nodes):
            dist = self._distance(node.config, q)
            if dist < min_dist:
                min_dist = dist
                min_idx = i

        return min_idx

    def _find_near(self, q: np.ndarray) -> list[int]:
        """Find all nodes within rewiring radius.

        Args:
            q: Query configuration.

        Returns:
            List of indices of near nodes.
        """
        radius = self._compute_rewire_radius()
        near_indices = []

        for i, node in enumerate(self._nodes):
            if self._distance(node.config, q) <= radius:
                near_indices.append(i)

        return near_indices

    def _choose_parent(
        self,
        q_new: np.ndarray,
        near_indices: list[int],
    ) -> int:
        """Choose best parent for new node from near neighbors.

        Args:
            q_new: New configuration.
            near_indices: Indices of near nodes.

        Returns:
            Index of best parent, or -1 if no valid parent.
        """
        best_cost = float("inf")
        best_idx = -1

        for idx in near_indices:
            node = self._nodes[idx]
            new_cost = node.cost + self._distance(node.config, q_new)

            if new_cost < best_cost:
                # Check if path is collision-free
                self._num_collision_checks += self._config.collision_check_resolution
                if self._is_path_valid(node.config, q_new):
                    best_cost = new_cost
                    best_idx = idx

        return best_idx

    def _rewire(self, new_idx: int, near_indices: list[int]) -> None:
        """Rewire tree to improve costs through new node.

        Args:
            new_idx: Index of newly added node.
            near_indices: Indices of near nodes to consider.
        """
        new_node = self._nodes[new_idx]

        for idx in near_indices:
            if idx == new_idx or idx == new_node.parent_idx:
                continue

            node = self._nodes[idx]
            new_cost = new_node.cost + self._distance(new_node.config, node.config)

            if new_cost < node.cost:
                # Check if path is collision-free
                self._num_collision_checks += self._config.collision_check_resolution
                if self._is_path_valid(new_node.config, node.config):
                    # Rewire: update parent and cost
                    node.parent_idx = new_idx
                    node.cost = new_cost
                    # Propagate cost updates to descendants
                    self._propagate_cost_update(idx)

    def _propagate_cost_update(self, start_idx: int) -> None:
        """Propagate cost updates to all descendants.

        Args:
            start_idx: Index of node whose cost was updated.
        """
        # Find all children and update their costs
        queue = [start_idx]
        while queue:
            current_idx = queue.pop(0)
            current_node = self._nodes[current_idx]

            for i, node in enumerate(self._nodes):
                if node.parent_idx == current_idx:
                    # Update child cost
                    node.cost = current_node.cost + self._distance(
                        current_node.config, node.config
                    )
                    queue.append(i)

    def _extract_path(self, goal_idx: int) -> list[np.ndarray]:
        """Extract path from tree by backtracking from goal.

        Args:
            goal_idx: Index of goal node.

        Returns:
            List of configurations from start to goal.
        """
        path = []
        idx = goal_idx

        while idx >= 0:
            path.append(self._nodes[idx].config.copy())
            idx = self._nodes[idx].parent_idx

        path.reverse()
        return path

    def get_tree_nodes(self) -> list[np.ndarray]:
        """Get all nodes in the tree (for visualization).

        Returns:
            List of configurations in tree.
        """
        return [node.config.copy() for node in self._nodes]

    def get_tree_edges(self) -> list[tuple[np.ndarray, np.ndarray]]:
        """Get all edges in the tree (for visualization).

        Returns:
            List of (parent, child) configuration pairs.
        """
        edges = []
        for node in self._nodes:
            if node.parent_idx >= 0:
                parent_config = self._nodes[node.parent_idx].config
                edges.append((parent_config.copy(), node.config.copy()))
        return edges

    def get_best_cost(self) -> float:
        """Get cost of best path found so far.

        Returns:
            Cost of best path, or inf if no path found.
        """
        # Find goal node (last node if goal was reached)
        for node in reversed(self._nodes):
            if node.parent_idx >= 0:
                return node.cost
        return float("inf")
