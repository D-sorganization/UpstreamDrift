"""Comprehensive tests for deployment.safety.collision."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from src.deployment.realtime import RobotState
from src.deployment.safety.collision import (
    CollisionAvoidance,
    HumanState,
    Obstacle,
    ObstacleType,
)


def _state(n: int = 3) -> RobotState:
    return RobotState(
        timestamp=0.0,
        joint_positions=np.zeros(n),
        joint_velocities=np.zeros(n),
        joint_torques=np.zeros(n),
    )


class TestObstacle:
    def test_sphere_distance(self) -> None:
        ob = Obstacle(
            name="s",
            obstacle_type=ObstacleType.SPHERE,
            position=np.zeros(3),
            dimensions=np.array([1.0]),
            inflation=0.0,
        )
        assert ob.get_distance(np.array([2.0, 0, 0])) == pytest.approx(1.0)

    def test_box_distance(self) -> None:
        ob = Obstacle(
            name="b",
            obstacle_type=ObstacleType.BOX,
            position=np.zeros(3),
            dimensions=np.array([2.0, 2.0, 2.0]),
            inflation=0.0,
        )
        assert ob.get_distance(np.array([2.0, 0.0, 0.0])) == pytest.approx(1.0)
        assert ob.get_distance(np.zeros(3)) <= 0  # inside

    def test_human_obstacle_uses_box_math(self) -> None:
        ob = Obstacle(
            name="h",
            obstacle_type=ObstacleType.HUMAN,
            position=np.zeros(3),
            dimensions=np.array([1.0, 1.0, 1.0]),
            inflation=0.0,
        )
        assert ob.get_distance(np.array([1.0, 0, 0])) == pytest.approx(0.5)

    def test_cylinder_distance(self) -> None:
        ob = Obstacle(
            name="c",
            obstacle_type=ObstacleType.CYLINDER,
            position=np.zeros(3),
            dimensions=np.array([1.0, 2.0]),
            inflation=0.0,
        )
        # Outside along radial
        assert ob.get_distance(np.array([2.0, 0, 0])) == pytest.approx(1.0)
        # Outside along z
        assert ob.get_distance(np.array([0.0, 0, 2.0])) == pytest.approx(1.0)
        # Corner case
        d = ob.get_distance(np.array([2.0, 0, 2.0]))
        assert d > 0

    def test_dynamic_returns_zero(self) -> None:
        ob = Obstacle(
            name="d",
            obstacle_type=ObstacleType.DYNAMIC,
            position=np.zeros(3),
            dimensions=np.array([1.0]),
        )
        assert ob.get_distance(np.array([5.0, 0, 0])) == 0.0

    def test_gradient_points_outward(self) -> None:
        ob = Obstacle(
            name="s",
            obstacle_type=ObstacleType.SPHERE,
            position=np.zeros(3),
            dimensions=np.array([1.0]),
            inflation=0.0,
        )
        g = ob.get_gradient(np.array([2.0, 0.0, 0.0]))
        assert g[0] > 0.9
        np.testing.assert_array_almost_equal(g[1:], [0.0, 0.0], decimal=3)


class TestHumanState:
    def test_to_obstacle(self) -> None:
        h = HumanState(position=np.array([1.0, 2.0, 3.0]))
        ob = h.to_obstacle()
        assert ob.obstacle_type == ObstacleType.HUMAN
        assert ob.inflation == 0.3


class TestCollisionAvoidance:
    def _ca(self) -> CollisionAvoidance:
        model = MagicMock()
        del model.get_link_positions  # force fallback
        del model.set_joint_positions
        return CollisionAvoidance(model, safety_distance=0.1)

    def test_add_remove_obstacle(self) -> None:
        ca = self._ca()
        ob = Obstacle(
            name="x",
            obstacle_type=ObstacleType.SPHERE,
            position=np.zeros(3),
            dimensions=np.array([0.1]),
        )
        ca.add_obstacle(ob)
        assert ca.remove_obstacle("x") is True
        assert ca.remove_obstacle("x") is False

    def test_clear_obstacles(self) -> None:
        ca = self._ca()
        ca.add_obstacle(
            Obstacle("a", ObstacleType.SPHERE, np.zeros(3), np.array([0.1]))
        )
        ca.clear_obstacles()
        assert ca._obstacles == []

    def test_update_human(self) -> None:
        ca = self._ca()
        h = HumanState(position=np.array([1.0, 0, 0]))
        ca.update_human_position(h)
        assert ca._human_state is h

    def test_get_link_positions_fallback(self) -> None:
        ca = self._ca()
        pos = ca.get_link_positions(_state(n=3))
        assert len(pos) == 3
        assert "link_0" in pos

    def test_get_link_positions_via_model(self) -> None:
        model = MagicMock()
        model.get_link_positions.return_value = {"link_0": np.zeros(3)}
        ca = CollisionAvoidance(model)
        result = ca.get_link_positions(_state())
        assert "link_0" in result

    def test_repulsive_field_no_obstacles(self) -> None:
        ca = self._ca()
        r = ca.compute_repulsive_field(_state())
        assert np.all(r == 0)

    def test_repulsive_field_with_close_obstacle(self) -> None:
        ca = self._ca()
        ca.add_obstacle(
            Obstacle(
                "s",
                ObstacleType.SPHERE,
                position=np.array([0.0, 0.0, 0.15]),
                dimensions=np.array([0.05]),
                inflation=0.0,
            )
        )
        r = ca.compute_repulsive_field(_state(n=3))
        assert r.shape == (3,)

    def test_repulsive_field_inside_obstacle(self) -> None:
        ca = self._ca()
        ca.add_obstacle(
            Obstacle(
                "s",
                ObstacleType.SPHERE,
                position=np.array([0.0, 0.0, 0.1]),
                dimensions=np.array([1.0]),
                inflation=0.0,
            )
        )
        r = ca.compute_repulsive_field(_state(n=3))
        assert r.shape == (3,)

    def test_path_clearance_clear(self) -> None:
        ca = self._ca()
        traj = np.zeros((5, 3))
        clear, dist = ca.check_path_clearance(traj)
        assert clear is True

    def test_path_clearance_blocked(self) -> None:
        ca = self._ca()
        ca.add_obstacle(
            Obstacle(
                "s",
                ObstacleType.SPHERE,
                position=np.array([0.0, 0.0, 0.2]),
                dimensions=np.array([1.0]),
                inflation=0.0,
            )
        )
        traj = np.zeros((3, 3))
        clear, _ = ca.check_path_clearance(traj, min_distance=0.5)
        assert clear is False

    def test_safe_velocity_no_obstacle(self) -> None:
        ca = self._ca()
        assert ca.get_safe_velocity_scaling(_state()) == 1.0

    def test_safe_velocity_inside(self) -> None:
        ca = self._ca()
        ca.add_obstacle(
            Obstacle(
                "s",
                ObstacleType.SPHERE,
                position=np.array([0, 0, 0.1]),
                dimensions=np.array([1.0]),
                inflation=0.0,
            )
        )
        assert ca.get_safe_velocity_scaling(_state(n=3)) == 0.0

    def test_safe_velocity_far(self) -> None:
        ca = self._ca()
        ca.add_obstacle(
            Obstacle(
                "s",
                ObstacleType.SPHERE,
                position=np.array([10.0, 0, 0]),
                dimensions=np.array([0.01]),
                inflation=0.0,
            )
        )
        assert ca.get_safe_velocity_scaling(_state(n=3)) == 1.0

    def test_safe_velocity_mid_range(self) -> None:
        ca = self._ca()
        # Fallback link_0 at z=0.1; obstacle far enough that
        # safety_distance < dist < repulsion_distance.
        ca.add_obstacle(
            Obstacle(
                "s",
                ObstacleType.SPHERE,
                position=np.array([0.5, 0.0, 0.1]),
                dimensions=np.array([0.05]),
                inflation=0.0,
            )
        )
        v = ca.get_safe_velocity_scaling(_state(n=1))
        assert 0.0 < v < 1.0

    def test_min_distance_no_obstacle(self) -> None:
        ca = self._ca()
        assert ca.get_minimum_distance(_state()) == float("inf")

    def test_min_distance_with_obstacle(self) -> None:
        ca = self._ca()
        ca.add_obstacle(
            Obstacle(
                "s",
                ObstacleType.SPHERE,
                position=np.array([1.0, 0, 0]),
                dimensions=np.array([0.1]),
                inflation=0.0,
            )
        )
        d = ca.get_minimum_distance(_state(n=3))
        assert d > 0
        assert d < float("inf")

    def test_with_human(self) -> None:
        ca = self._ca()
        ca.update_human_position(HumanState(position=np.array([0.5, 0.0, 0.0])))
        d = ca.get_minimum_distance(_state(n=3))
        assert d < float("inf")
