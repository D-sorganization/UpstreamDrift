"""RRT (Rapidly-exploring Random Trees) motion planner.

This module implements the basic RRT algorithm for single-query
motion planning in configuration space.

Reference:
    LaValle, S. M. (1998). Rapidly-exploring random trees:
    A new tool for path planning. Tech. Rep.
"""

from __future__ import annotations

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


@dataclass
class RRTConfig(PlannerConfig):
    """Configuration for RRT planner.

    Inherits all parameters from PlannerConfig.

    Additional Attributes:
        None currently - reserved for future RRT-specific options.
    """


@dataclass
class TreeNode:
    """Node in the RRT tree.

    Attributes:
        config: Configuration at this node.
        parent_idx: Index of parent node (-1 for root).
        cost: Cost from root to this node.
    """

    config: np.ndarray
    parent_idx: int = -1
    cost: float = 0.0


class RRTPlanner(MotionPlanner):
    """RRT motion planner.

    Design by Contract:
        Preconditions:
            - Start and goal configurations must be valid
            - Bounds must be set before planning

        Postconditions:
            - If SUCCESS, returned path is collision-free
            - Path starts at q_start and ends near q_goal

        Invariants:
            - Tree is reset between plan() calls

    Example:
        >>> planner = RRTPlanner(collision_checker)
        >>> planner.set_bounds(lower, upper)
        >>> result = planner.plan(q_start, q_goal)
        >>> if result.success:
        ...     for waypoint in result.path:
        ...         execute(waypoint)
    """

    def __init__(
        self,
        collision_checker: CollisionCheckerProtocol,
        config: RRTConfig | None = None,
    ) -> None:
        """Initialize RRT planner.

        Args:
            collision_checker: Collision checking interface.
            config: RRT configuration.
        """
        super().__init__(collision_checker, config or RRTConfig())
        self._nodes: list[TreeNode] = []
        self._num_collision_checks = 0

    def plan(
        self,
        q_start: np.ndarray,
        q_goal: np.ndarray,
    ) -> PlannerResult:
        """Plan a path using RRT.

        Args:
            q_start: Start configuration.
            q_goal: Goal configuration.

        Returns:
            PlannerResult with path and statistics.
        """
        q_start = np.asarray(q_start)
        q_goal = np.asarray(q_goal)

        self._nodes = []
        self._num_collision_checks = 0
        start_time = time.perf_counter()

        validation_result = self._validate_start_goal(q_start, q_goal, start_time)
        if validation_result is not None:
            return validation_result

        self._nodes.append(TreeNode(config=q_start.copy(), parent_idx=-1, cost=0.0))

        goal_idx = -1
        iterations = 0

        while iterations < self._config.max_iterations:
            if time.perf_counter() - start_time > self._config.max_time:
                return self._timeout_result(iterations, start_time)

            iterations += 1

            new_idx, new_cost = self._expand_tree(q_goal)
            if new_idx < 0:
                continue

            goal_idx = self._try_connect_goal(new_idx, new_cost, q_goal)
            if goal_idx >= 0:
                break

        return self._build_result(goal_idx, iterations, start_time)

    def _validate_start_goal(
        self,
        q_start: np.ndarray,
        q_goal: np.ndarray,
        start_time: float,
    ) -> PlannerResult | None:
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
        return None

    def _expand_tree(self, q_goal: np.ndarray) -> tuple[int, float]:
        q_rand = self._sample_with_goal_bias(q_goal)
        nearest_idx = self._find_nearest(q_rand)
        q_nearest = self._nodes[nearest_idx].config
        q_new = self._steer(q_nearest, q_rand)

        self._num_collision_checks += 1
        if not self._is_valid(q_new):
            return -1, 0.0

        self._num_collision_checks += self._config.collision_check_resolution
        if not self._is_path_valid(q_nearest, q_new):
            return -1, 0.0

        new_cost = self._nodes[nearest_idx].cost + self._distance(q_nearest, q_new)
        new_node = TreeNode(
            config=q_new.copy(),
            parent_idx=nearest_idx,
            cost=new_cost,
        )
        new_idx = len(self._nodes)
        self._nodes.append(new_node)
        return new_idx, new_cost

    def _try_connect_goal(
        self,
        new_idx: int,
        new_cost: float,
        q_goal: np.ndarray,
    ) -> int:
        q_new = self._nodes[new_idx].config
        if self._distance(q_new, q_goal) > self._config.goal_tolerance:
            return -1

        self._num_collision_checks += self._config.collision_check_resolution
        if not self._is_path_valid(q_new, q_goal):
            return -1

        goal_cost = new_cost + self._distance(q_new, q_goal)
        goal_node = TreeNode(
            config=q_goal.copy(),
            parent_idx=new_idx,
            cost=goal_cost,
        )
        goal_idx = len(self._nodes)
        self._nodes.append(goal_node)
        return goal_idx

    def _timeout_result(self, iterations: int, start_time: float) -> PlannerResult:
        return PlannerResult(
            status=PlannerStatus.TIMEOUT,
            num_iterations=iterations,
            planning_time=time.perf_counter() - start_time,
            num_nodes=len(self._nodes),
            num_collision_checks=self._num_collision_checks,
        )

    def _build_result(
        self,
        goal_idx: int,
        iterations: int,
        start_time: float,
    ) -> PlannerResult:
        planning_time = time.perf_counter() - start_time

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

        return PlannerResult(
            status=PlannerStatus.FAILURE,
            num_iterations=iterations,
            planning_time=planning_time,
            num_nodes=len(self._nodes),
            num_collision_checks=self._num_collision_checks,
        )

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
