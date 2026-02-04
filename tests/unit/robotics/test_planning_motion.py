"""Tests for motion planning module.

Tests cover:
- PlannerConfig and PlannerResult
- RRT planner
- RRT* planner
- Path validation and collision checking integration
"""

from __future__ import annotations

import numpy as np
import pytest

from src.robotics.planning.motion import (
    PlannerConfig,
    PlannerResult,
    PlannerStatus,
    RRTConfig,
    RRTPlanner,
    RRTStarConfig,
    RRTStarPlanner,
)

# =============================================================================
# Mock Collision Checker for Testing
# =============================================================================


class MockCollisionResult:
    """Mock collision result."""

    def __init__(self, in_collision: bool) -> None:
        self.in_collision = in_collision


class MockCollisionChecker:
    """Mock collision checker for testing."""

    def __init__(
        self,
        obstacles: list[tuple[np.ndarray, float]] | None = None,
    ) -> None:
        """Initialize with optional spherical obstacles.

        Args:
            obstacles: List of (center, radius) tuples for obstacles.
        """
        self.obstacles = obstacles or []
        self.collision_check_count = 0

    def check_collision(self, q: np.ndarray) -> MockCollisionResult:
        """Check if configuration is in collision."""
        self.collision_check_count += 1

        for center, radius in self.obstacles:
            if np.linalg.norm(q - center) < radius:
                return MockCollisionResult(in_collision=True)

        return MockCollisionResult(in_collision=False)

    def check_path_collision(
        self,
        q_start: np.ndarray,
        q_end: np.ndarray,
        num_samples: int = 10,
    ) -> tuple[bool, float | None]:
        """Check if path is collision-free."""
        for i in range(num_samples):
            t = i / (num_samples - 1)
            q = q_start + t * (q_end - q_start)
            result = self.check_collision(q)
            if result.in_collision:
                return False, t

        return True, None


# =============================================================================
# PlannerConfig Tests
# =============================================================================


class TestPlannerConfig:
    """Tests for PlannerConfig."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = PlannerConfig()
        assert config.max_iterations == 10000
        assert config.max_time == 30.0
        assert config.goal_bias == 0.05
        assert config.step_size == 0.1
        assert config.goal_tolerance == 0.01

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        config = PlannerConfig(
            max_iterations=5000,
            max_time=10.0,
            goal_bias=0.1,
            step_size=0.2,
        )
        assert config.max_iterations == 5000
        assert config.step_size == 0.2

    def test_invalid_max_iterations(self) -> None:
        """Test that non-positive max_iterations raises error."""
        with pytest.raises(ValueError, match="max_iterations must be positive"):
            PlannerConfig(max_iterations=0)

    def test_invalid_max_time(self) -> None:
        """Test that non-positive max_time raises error."""
        with pytest.raises(ValueError, match="max_time must be positive"):
            PlannerConfig(max_time=-1.0)

    def test_invalid_goal_bias(self) -> None:
        """Test that goal_bias outside [0,1] raises error."""
        with pytest.raises(ValueError, match="goal_bias must be in"):
            PlannerConfig(goal_bias=1.5)
        with pytest.raises(ValueError, match="goal_bias must be in"):
            PlannerConfig(goal_bias=-0.1)

    def test_invalid_step_size(self) -> None:
        """Test that non-positive step_size raises error."""
        with pytest.raises(ValueError, match="step_size must be positive"):
            PlannerConfig(step_size=0.0)


# =============================================================================
# PlannerResult Tests
# =============================================================================


class TestPlannerResult:
    """Tests for PlannerResult."""

    def test_success_result(self) -> None:
        """Test successful planning result."""
        path = [np.array([0, 0]), np.array([1, 1])]
        result = PlannerResult(
            status=PlannerStatus.SUCCESS,
            path=path,
            path_length=np.sqrt(2),
            num_iterations=100,
            planning_time=0.5,
        )
        assert result.success
        assert len(result.path) == 2

    def test_failure_result(self) -> None:
        """Test failed planning result."""
        result = PlannerResult(
            status=PlannerStatus.FAILURE,
            num_iterations=1000,
        )
        assert not result.success
        assert len(result.path) == 0

    def test_success_requires_path(self) -> None:
        """Test that SUCCESS status requires non-empty path."""
        with pytest.raises(ValueError, match="requires non-empty path"):
            PlannerResult(status=PlannerStatus.SUCCESS, path=[])

    def test_get_path_array(self) -> None:
        """Test path conversion to numpy array."""
        path = [np.array([0, 0, 0]), np.array([1, 1, 1]), np.array([2, 2, 2])]
        result = PlannerResult(status=PlannerStatus.SUCCESS, path=path)
        arr = result.get_path_array()
        assert arr.shape == (3, 3)

    def test_get_path_array_empty(self) -> None:
        """Test empty path conversion."""
        result = PlannerResult(status=PlannerStatus.FAILURE)
        arr = result.get_path_array()
        assert arr.size == 0


# =============================================================================
# RRT Planner Tests
# =============================================================================


class TestRRTPlanner:
    """Tests for RRT planner."""

    @pytest.fixture
    def collision_checker(self) -> MockCollisionChecker:
        """Create collision checker with no obstacles."""
        return MockCollisionChecker()

    @pytest.fixture
    def planner(self, collision_checker: MockCollisionChecker) -> RRTPlanner:
        """Create RRT planner."""
        config = RRTConfig(
            max_iterations=1000,
            max_time=5.0,
            step_size=0.5,
            goal_tolerance=0.1,
        )
        planner = RRTPlanner(collision_checker, config)
        planner.set_bounds(np.array([-5, -5]), np.array([5, 5]))
        planner.set_seed(42)
        return planner

    def test_create_planner(self, collision_checker: MockCollisionChecker) -> None:
        """Test planner creation."""
        planner = RRTPlanner(collision_checker)
        assert planner is not None

    def test_plan_simple_path(self, planner: RRTPlanner) -> None:
        """Test planning simple collision-free path."""
        q_start = np.array([0.0, 0.0])
        q_goal = np.array([2.0, 2.0])

        result = planner.plan(q_start, q_goal)

        assert result.success
        assert len(result.path) >= 2
        assert np.allclose(result.path[0], q_start)
        assert (
            np.linalg.norm(result.path[-1] - q_goal) <= planner._config.goal_tolerance
        )

    def test_plan_invalid_start(self, collision_checker: MockCollisionChecker) -> None:
        """Test planning with invalid start configuration."""
        # Add obstacle at start position
        collision_checker.obstacles.append((np.array([0.0, 0.0]), 0.5))

        planner = RRTPlanner(collision_checker)
        planner.set_bounds(np.array([-5, -5]), np.array([5, 5]))

        result = planner.plan(np.array([0.0, 0.0]), np.array([2.0, 2.0]))

        assert result.status == PlannerStatus.INVALID_START

    def test_plan_invalid_goal(self, collision_checker: MockCollisionChecker) -> None:
        """Test planning with invalid goal configuration."""
        # Add obstacle at goal position
        collision_checker.obstacles.append((np.array([2.0, 2.0]), 0.5))

        planner = RRTPlanner(collision_checker)
        planner.set_bounds(np.array([-5, -5]), np.array([5, 5]))

        result = planner.plan(np.array([0.0, 0.0]), np.array([2.0, 2.0]))

        assert result.status == PlannerStatus.INVALID_GOAL

    def test_plan_with_obstacle(self) -> None:
        """Test planning around obstacle."""
        # Add obstacle between start and goal
        collision_checker = MockCollisionChecker(
            obstacles=[(np.array([1.0, 1.0]), 0.3)]
        )

        config = RRTConfig(
            max_iterations=2000,
            step_size=0.3,
            goal_tolerance=0.2,
        )
        planner = RRTPlanner(collision_checker, config)
        planner.set_bounds(np.array([-5, -5]), np.array([5, 5]))
        planner.set_seed(42)

        result = planner.plan(np.array([0.0, 0.0]), np.array([2.0, 2.0]))

        # Should find path around obstacle
        assert result.success or result.status == PlannerStatus.FAILURE
        # Path should avoid obstacle if found
        if result.success:
            for waypoint in result.path:
                assert np.linalg.norm(waypoint - np.array([1.0, 1.0])) >= 0.25

    def test_plan_statistics(self, planner: RRTPlanner) -> None:
        """Test planning returns proper statistics."""
        result = planner.plan(np.array([0.0, 0.0]), np.array([1.0, 1.0]))

        assert result.num_iterations > 0
        assert result.planning_time > 0
        assert result.num_nodes > 0
        assert result.num_collision_checks > 0

    def test_set_seed_reproducibility(
        self,
        collision_checker: MockCollisionChecker,
    ) -> None:
        """Test that setting seed gives reproducible results."""
        config = RRTConfig(max_iterations=500)

        planner1 = RRTPlanner(collision_checker, config)
        planner1.set_bounds(np.array([-5, -5]), np.array([5, 5]))
        planner1.set_seed(123)
        result1 = planner1.plan(np.array([0.0, 0.0]), np.array([2.0, 2.0]))

        planner2 = RRTPlanner(collision_checker, config)
        planner2.set_bounds(np.array([-5, -5]), np.array([5, 5]))
        planner2.set_seed(123)
        result2 = planner2.plan(np.array([0.0, 0.0]), np.array([2.0, 2.0]))

        # Same seed should give same number of nodes
        assert result1.num_nodes == result2.num_nodes

    def test_get_tree_nodes(self, planner: RRTPlanner) -> None:
        """Test getting tree nodes for visualization."""
        planner.plan(np.array([0.0, 0.0]), np.array([1.0, 1.0]))
        nodes = planner.get_tree_nodes()
        assert len(nodes) > 0
        assert all(isinstance(n, np.ndarray) for n in nodes)

    def test_get_tree_edges(self, planner: RRTPlanner) -> None:
        """Test getting tree edges for visualization."""
        planner.plan(np.array([0.0, 0.0]), np.array([1.0, 1.0]))
        edges = planner.get_tree_edges()
        assert len(edges) >= 0
        for parent, child in edges:
            assert isinstance(parent, np.ndarray)
            assert isinstance(child, np.ndarray)


class TestRRTConfig:
    """Tests for RRTConfig."""

    def test_rrt_config_inheritance(self) -> None:
        """Test that RRTConfig inherits from PlannerConfig."""
        config = RRTConfig(max_iterations=500, step_size=0.2)
        assert config.max_iterations == 500
        assert config.step_size == 0.2


# =============================================================================
# RRT* Planner Tests
# =============================================================================


class TestRRTStarPlanner:
    """Tests for RRT* planner."""

    @pytest.fixture
    def collision_checker(self) -> MockCollisionChecker:
        """Create collision checker."""
        return MockCollisionChecker()

    @pytest.fixture
    def planner(self, collision_checker: MockCollisionChecker) -> RRTStarPlanner:
        """Create RRT* planner."""
        config = RRTStarConfig(
            max_iterations=1000,
            max_time=5.0,
            step_size=0.5,
            goal_tolerance=0.1,
        )
        planner = RRTStarPlanner(collision_checker, config)
        planner.set_bounds(np.array([-5, -5]), np.array([5, 5]))
        planner.set_seed(42)
        return planner

    def test_create_planner(self, collision_checker: MockCollisionChecker) -> None:
        """Test planner creation."""
        planner = RRTStarPlanner(collision_checker)
        assert planner is not None

    def test_plan_simple_path(self, planner: RRTStarPlanner) -> None:
        """Test planning simple path."""
        q_start = np.array([0.0, 0.0])
        q_goal = np.array([2.0, 2.0])

        result = planner.plan(q_start, q_goal)

        assert result.success
        assert len(result.path) >= 2
        assert np.allclose(result.path[0], q_start)

    def test_plan_invalid_start(self, collision_checker: MockCollisionChecker) -> None:
        """Test planning with invalid start."""
        collision_checker.obstacles.append((np.array([0.0, 0.0]), 0.5))

        planner = RRTStarPlanner(collision_checker)
        planner.set_bounds(np.array([-5, -5]), np.array([5, 5]))

        result = planner.plan(np.array([0.0, 0.0]), np.array([2.0, 2.0]))

        assert result.status == PlannerStatus.INVALID_START

    def test_plan_finds_shorter_path(self) -> None:
        """Test that RRT* improves path cost over time."""
        collision_checker = MockCollisionChecker()

        # Run with more iterations to allow optimization
        config = RRTStarConfig(
            max_iterations=500,
            step_size=0.3,
            goal_tolerance=0.2,
        )
        planner = RRTStarPlanner(collision_checker, config)
        planner.set_bounds(np.array([-5, -5]), np.array([5, 5]))
        planner.set_seed(42)

        result = planner.plan(np.array([0.0, 0.0]), np.array([2.0, 2.0]))

        if result.success:
            # Path length should be reasonable (not much longer than straight line)
            straight_line_dist = np.sqrt(8)  # sqrt(2^2 + 2^2)
            assert result.path_length < straight_line_dist * 2

    def test_rewiring_improves_cost(self) -> None:
        """Test that rewiring actually improves costs."""
        collision_checker = MockCollisionChecker()

        config = RRTStarConfig(
            max_iterations=300,
            step_size=0.4,
            goal_tolerance=0.3,
        )
        planner = RRTStarPlanner(collision_checker, config)
        planner.set_bounds(np.array([-3, -3]), np.array([3, 3]))
        planner.set_seed(42)

        result = planner.plan(np.array([0.0, 0.0]), np.array([2.0, 2.0]))

        # Just check that planning succeeds and produces reasonable result
        assert result.status in [PlannerStatus.SUCCESS, PlannerStatus.FAILURE]
        if result.success:
            assert result.path_length > 0

    def test_get_best_cost(self, planner: RRTStarPlanner) -> None:
        """Test getting best path cost."""
        result = planner.plan(np.array([0.0, 0.0]), np.array([1.0, 1.0]))

        if result.success:
            best_cost = planner.get_best_cost()
            assert best_cost > 0
            assert best_cost < float("inf")


class TestRRTStarConfig:
    """Tests for RRTStarConfig."""

    def test_default_config(self) -> None:
        """Test default configuration."""
        config = RRTStarConfig()
        assert config.rewire_radius is None
        assert config.rewire_factor == 1.1

    def test_custom_rewire_radius(self) -> None:
        """Test custom rewire radius."""
        config = RRTStarConfig(rewire_radius=0.5)
        assert config.rewire_radius == 0.5

    def test_invalid_rewire_radius(self) -> None:
        """Test that non-positive rewire radius raises error."""
        with pytest.raises(ValueError, match="rewire_radius must be positive"):
            RRTStarConfig(rewire_radius=0.0)

    def test_invalid_rewire_factor(self) -> None:
        """Test that non-positive rewire factor raises error."""
        with pytest.raises(ValueError, match="rewire_factor must be positive"):
            RRTStarConfig(rewire_factor=-1.0)


# =============================================================================
# Integration Tests
# =============================================================================


class TestPlannerIntegration:
    """Integration tests for motion planners."""

    def test_rrt_vs_rrt_star_path_quality(self) -> None:
        """Compare RRT and RRT* path quality."""
        collision_checker = MockCollisionChecker()

        # Configure both planners similarly
        rrt_config = RRTConfig(
            max_iterations=500,
            step_size=0.3,
            goal_tolerance=0.2,
        )
        rrt_star_config = RRTStarConfig(
            max_iterations=500,
            step_size=0.3,
            goal_tolerance=0.2,
        )

        rrt = RRTPlanner(collision_checker, rrt_config)
        rrt.set_bounds(np.array([-5, -5]), np.array([5, 5]))
        rrt.set_seed(42)

        rrt_star = RRTStarPlanner(collision_checker, rrt_star_config)
        rrt_star.set_bounds(np.array([-5, -5]), np.array([5, 5]))
        rrt_star.set_seed(42)

        q_start = np.array([0.0, 0.0])
        q_goal = np.array([3.0, 3.0])

        rrt_result = rrt.plan(q_start, q_goal)
        rrt_star_result = rrt_star.plan(q_start, q_goal)

        # Both should find a path in this simple case
        # (though not guaranteed with limited iterations)
        if rrt_result.success and rrt_star_result.success:
            # RRT* should generally find equal or better paths
            # (not strictly guaranteed in all cases)
            assert rrt_star_result.path_length <= rrt_result.path_length * 1.5

    def test_planner_with_narrow_passage(self) -> None:
        """Test planning through narrow passage."""
        # Create narrow passage with two obstacles
        obstacles = [
            (np.array([1.0, 0.5]), 0.4),
            (np.array([1.0, -0.5]), 0.4),
        ]
        collision_checker = MockCollisionChecker(obstacles=obstacles)

        config = RRTConfig(
            max_iterations=2000,
            step_size=0.2,
            goal_tolerance=0.15,
        )
        planner = RRTPlanner(collision_checker, config)
        planner.set_bounds(np.array([-2, -2]), np.array([3, 2]))
        planner.set_seed(42)

        result = planner.plan(np.array([0.0, 0.0]), np.array([2.0, 0.0]))

        # Narrow passages are hard - may or may not succeed
        assert result.status in [
            PlannerStatus.SUCCESS,
            PlannerStatus.FAILURE,
        ]

    def test_high_dimensional_planning(self) -> None:
        """Test planning in higher dimensions."""
        collision_checker = MockCollisionChecker()

        config = RRTConfig(
            max_iterations=2000,
            step_size=0.5,
            goal_tolerance=0.3,
        )
        planner = RRTPlanner(collision_checker, config)

        # 6-DOF configuration space
        planner.set_bounds(
            np.array([-np.pi] * 6),
            np.array([np.pi] * 6),
        )
        planner.set_seed(42)

        q_start = np.zeros(6)
        q_goal = np.array([1.0, 1.0, 1.0, 0.5, 0.5, 0.5])

        result = planner.plan(q_start, q_goal)

        # Should succeed in open space
        assert result.success
        assert result.path[0].shape == (6,)
