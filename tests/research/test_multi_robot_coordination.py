"""Smoke tests for research multi-robot coordination module."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
from src.research.multi_robot.coordination import (
    CooperativeManipulation,
    FormationConfig,
    FormationController,
)


class TestFormationConfig:
    """Smoke tests for FormationConfig."""

    def test_line_formation(self) -> None:
        config = FormationConfig.line_formation(n_robots=4, spacing=1.5)
        assert config.name == "line"
        assert config.positions.shape == (4, 3)
        np.testing.assert_allclose(config.positions[0], [0, 0, 0])
        np.testing.assert_allclose(config.positions[1], [0, 1.5, 0])

    def test_circle_formation(self) -> None:
        config = FormationConfig.circle_formation(n_robots=3, radius=2.0)
        assert config.name == "circle"
        assert config.positions.shape == (3, 3)

    def test_wedge_formation(self) -> None:
        config = FormationConfig.wedge_formation(n_robots=5)
        assert config.name == "wedge"
        assert config.positions.shape == (5, 3)
        np.testing.assert_allclose(config.positions[0], [0, 0, 0])


class TestFormationController:
    """Smoke tests for FormationController."""

    def test_multi_robot_coordination_construction(self) -> None:
        config = FormationConfig.line_formation(3)
        fc = FormationController(robots=["r0", "r1", "r2"], formation=config)
        assert len(fc.robots) == 3

    def test_compute_formation_control(self) -> None:
        config = FormationConfig.line_formation(2)
        fc = FormationController(robots=["r0", "r1"], formation=config)
        leader_pose = np.array([0.0, 0.0, 0.0])
        positions = {
            "r0": np.array([0.1, 0.0, 0.0]),
            "r1": np.array([0.0, 0.5, 0.0]),
        }
        commands = fc.compute_formation_control(leader_pose, positions)
        assert "r0" in commands
        assert "r1" in commands
        assert commands["r0"].shape == (3,)

    def test_get_formation_error(self) -> None:
        config = FormationConfig.line_formation(2, spacing=1.0)
        fc = FormationController(robots=["r0", "r1"], formation=config)
        leader_pose = np.array([0.0, 0.0, 0.0])
        positions = {
            "r0": np.array([0.0, 0.0, 0.0]),
            "r1": np.array([0.0, 1.0, 0.0]),
        }
        error = fc.get_formation_error(leader_pose, positions)
        assert error >= 0.0

    def test_set_gains(self) -> None:
        config = FormationConfig.line_formation(2)
        fc = FormationController(robots=["r0", "r1"], formation=config)
        fc.set_gains(position=3.0, velocity=2.0, heading=0.5)


class TestCooperativeManipulation:
    """Smoke tests for CooperativeManipulation."""

    def test_multi_robot_coordination_construction(self) -> None:
        robots = [MagicMock(), MagicMock()]
        cm = CooperativeManipulation(robots)
        assert cm.n_robots == 2

    def test_set_grasp_points(self) -> None:
        robots = [MagicMock(), MagicMock()]
        cm = CooperativeManipulation(robots)
        grasp_points = [np.array([0.1, 0.0, 0.0]), np.array([-0.1, 0.0, 0.0])]
        cm.set_grasp_points(grasp_points)

    def test_compute_grasp_matrix(self) -> None:
        robots = [MagicMock(), MagicMock()]
        cm = CooperativeManipulation(robots)
        grasp_points = [np.array([0.1, 0.0, 0.0]), np.array([-0.1, 0.0, 0.0])]
        cm.set_grasp_points(grasp_points)
        pose = np.array([0, 0, 0, 1, 0, 0, 0], dtype=float)
        G = cm.compute_grasp_matrix(pose)
        assert G.shape == (6, 6)

    def test_check_force_closure(self) -> None:
        robots = [MagicMock(), MagicMock()]
        cm = CooperativeManipulation(robots)
        grasp_points = [np.array([0.1, 0.0, 0.0]), np.array([-0.1, 0.0, 0.0])]
        cm.set_grasp_points(grasp_points)
        pose = np.array([0, 0, 0, 1, 0, 0, 0], dtype=float)
        has_closure, quality = cm.check_force_closure(pose)
        assert (
            has_closure is True
            or has_closure is False
            or isinstance(has_closure, (bool, np.bool_))
        )
        assert isinstance(quality, (float, np.floating))
