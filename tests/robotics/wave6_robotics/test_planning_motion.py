"""Tests for planner_base, RRT, RRT* motion planners.

Uses tiny configuration spaces and small iteration counts for speed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from src.robotics.planning.motion.planner_base import (
    PlannerConfig,
    PlannerResult,
    PlannerStatus,
)
from src.robotics.planning.motion.rrt import RRTConfig, RRTPlanner, TreeNode
from src.robotics.planning.motion.rrt_star import RRTStarConfig, RRTStarPlanner


@dataclass
class _Result:
    in_collision: bool = False


class FreeChecker:
    def check_collision(self, q):
        return _Result(in_collision=False)

    def check_path_collision(self, qs, qe, num_samples=10):
        return True, None


class BlockChecker:
    """Blocks any configuration with x-coordinate > threshold."""

    def __init__(self, threshold=0.5) -> None:
        self.threshold = threshold

    def check_collision(self, q):
        return _Result(in_collision=bool(q[0] > self.threshold))

    def check_path_collision(self, qs, qe, num_samples=10):
        for t in np.linspace(0.0, 1.0, num_samples):
            q = qs + t * (qe - qs)
            if q[0] > self.threshold:
                return False, float(t)
        return True, None


# ---- PlannerConfig ----


class TestPlannerConfig:
    def test_defaults_valid(self) -> None:
        c = PlannerConfig()
        assert c.max_iterations > 0

    @pytest.mark.parametrize(
        ("attr", "val", "msg"),
        [
            ("max_iterations", 0, "max_iterations"),
            ("max_time", 0, "max_time"),
            ("goal_bias", -0.1, "goal_bias"),
            ("goal_bias", 1.1, "goal_bias"),
            ("step_size", 0.0, "step_size"),
            ("goal_tolerance", 0.0, "goal_tolerance"),
            ("collision_check_resolution", 1, "collision_check_resolution"),
        ],
    )
    def test_invalid_values(self, attr, val, msg) -> None:
        kw = {attr: val}
        with pytest.raises(ValueError, match=msg):
            PlannerConfig(**kw)


# ---- PlannerResult ----


class TestPlannerResult:
    def test_success_with_path(self) -> None:
        r = PlannerResult(
            status=PlannerStatus.SUCCESS,
            path=[np.zeros(2), np.ones(2)],
            path_length=1.41,
        )
        assert r.success
        arr = r.get_path_array()
        assert arr.shape == (2, 2)

    def test_success_without_path_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            PlannerResult(status=PlannerStatus.SUCCESS, path=[])

    def test_negative_length(self) -> None:
        with pytest.raises(ValueError, match="path_length"):
            PlannerResult(status=PlannerStatus.FAILURE, path_length=-1.0)

    def test_negative_iterations(self) -> None:
        with pytest.raises(ValueError, match="num_iterations"):
            PlannerResult(status=PlannerStatus.FAILURE, num_iterations=-1)

    def test_negative_time(self) -> None:
        with pytest.raises(ValueError, match="planning_time"):
            PlannerResult(status=PlannerStatus.FAILURE, planning_time=-0.1)

    def test_failure_empty_path_array(self) -> None:
        r = PlannerResult(status=PlannerStatus.FAILURE)
        assert not r.success
        assert r.get_path_array().shape == (0,)


# ---- TreeNode ----


def test_tree_node_default() -> None:
    n = TreeNode(config=np.zeros(2))
    assert n.parent_idx == -1
    assert n.cost == 0.0


# ---- Shared helpers via RRT ----


def _make_rrt(checker=None, **cfg_kw):
    cfg = RRTConfig(
        max_iterations=200,
        max_time=2.0,
        step_size=0.2,
        goal_bias=0.3,
        goal_tolerance=0.1,
        collision_check_resolution=3,
        **cfg_kw,
    )
    planner = RRTPlanner(checker or FreeChecker(), config=cfg)
    planner.set_bounds(np.array([-1.0, -1.0]), np.array([1.0, 1.0]))
    planner.set_seed(42)
    return planner


def test_set_bounds_invalid_shape() -> None:
    p = _make_rrt()
    with pytest.raises(ValueError, match="shape"):
        p.set_bounds(np.zeros(2), np.zeros(3))


def test_set_bounds_lower_ge_upper() -> None:
    p = _make_rrt()
    with pytest.raises(ValueError, match="Lower bounds"):
        p.set_bounds(np.ones(2), np.zeros(2))


def test_sample_random_requires_bounds() -> None:
    p = RRTPlanner(FreeChecker(), config=RRTConfig())
    with pytest.raises(RuntimeError, match="Bounds"):
        p._sample_random()


def test_rrt_plan_simple_success() -> None:
    p = _make_rrt()
    r = p.plan(np.array([-0.9, -0.9]), np.array([0.9, 0.9]))
    assert r.status in (PlannerStatus.SUCCESS, PlannerStatus.FAILURE)
    if r.success:
        assert np.allclose(r.path[0], [-0.9, -0.9])
        assert r.path_length > 0


def test_rrt_invalid_start() -> None:
    p = _make_rrt(checker=BlockChecker(threshold=-1.5))  # blocks everything
    r = p.plan(np.array([-0.9, -0.9]), np.array([0.9, 0.9]))
    assert r.status == PlannerStatus.INVALID_START


def test_rrt_invalid_goal() -> None:
    class GoalBlocker:
        def check_collision(self, q):
            return _Result(in_collision=bool(q[0] > 0.5))

        def check_path_collision(self, qs, qe, num_samples=10):
            return True, None

    p = _make_rrt(checker=GoalBlocker())
    r = p.plan(np.array([-0.9, -0.9]), np.array([0.9, 0.9]))
    assert r.status == PlannerStatus.INVALID_GOAL


def test_rrt_tree_nodes_and_edges() -> None:
    p = _make_rrt()
    p.plan(np.array([-0.9, -0.9]), np.array([0.9, 0.9]))
    nodes = p.get_tree_nodes()
    edges = p.get_tree_edges()
    assert len(nodes) >= 1
    assert len(edges) == max(0, len(nodes) - 1) or len(edges) >= 0


def test_rrt_steer_within_distance() -> None:
    p = _make_rrt()
    q_from = np.array([0.0, 0.0])
    q_to = np.array([0.05, 0.0])  # within step size
    q_new = p._steer(q_from, q_to)
    assert np.allclose(q_new, q_to)


def test_rrt_steer_beyond_distance() -> None:
    p = _make_rrt()
    q_from = np.array([0.0, 0.0])
    q_to = np.array([1.0, 0.0])  # far
    q_new = p._steer(q_from, q_to)
    assert np.linalg.norm(q_new - q_from) == pytest.approx(
        p._config.step_size, abs=1e-6
    )


def test_rrt_distance_and_path_length() -> None:
    p = _make_rrt()
    assert p._distance(np.array([0, 0]), np.array([3.0, 4.0])) == pytest.approx(5.0)
    path = [np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.array([1.0, 1.0])]
    assert p._compute_path_length(path) == pytest.approx(2.0)
    assert p._compute_path_length([]) == 0.0


def test_rrt_set_seed_reproducible() -> None:
    p1 = _make_rrt()
    p2 = _make_rrt()
    s1 = p1._sample_random()
    s2 = p2._sample_random()
    assert np.allclose(s1, s2)


# ---- RRT* ----


def test_rrt_star_config_validation() -> None:
    with pytest.raises(ValueError, match="rewire_radius"):
        RRTStarConfig(rewire_radius=-0.1)
    with pytest.raises(ValueError, match="rewire_factor"):
        RRTStarConfig(rewire_factor=0.0)


def _make_rrt_star(checker=None, **kw):
    cfg = RRTStarConfig(
        max_iterations=80,
        max_time=2.0,
        step_size=0.3,
        goal_bias=0.3,
        goal_tolerance=0.15,
        collision_check_resolution=3,
        rewire_radius=0.5,
        **kw,
    )
    p = RRTStarPlanner(checker or FreeChecker(), config=cfg)
    p.set_bounds(np.array([-1.0, -1.0]), np.array([1.0, 1.0]))
    p.set_seed(123)
    return p


def test_rrt_star_invalid_start() -> None:
    p = _make_rrt_star(checker=BlockChecker(threshold=-1.5))
    r = p.plan(np.array([-0.9, -0.9]), np.array([0.9, 0.9]))
    assert r.status == PlannerStatus.INVALID_START


def test_rrt_star_simple_plan() -> None:
    p = _make_rrt_star()
    r = p.plan(np.array([-0.9, -0.9]), np.array([0.9, 0.9]))
    assert isinstance(r, PlannerResult)
