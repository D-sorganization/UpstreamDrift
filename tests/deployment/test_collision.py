"""Tests for collision avoidance module."""

from unittest.mock import MagicMock

import numpy as np
import pytest
from src.deployment.realtime.state import RobotState
from src.deployment.safety.collision import (
    CollisionAvoidance,
    HumanState,
    Obstacle,
    ObstacleType,
)


@pytest.fixture
def mock_sim() -> MagicMock:
    """Mock simulation engine."""
    sim = MagicMock()
    return sim


def test_obstacle_distance_box() -> None:
    """Test distance computation for box obstacle."""
    box = Obstacle(
        name="box_1",
        obstacle_type=ObstacleType.BOX,
        position=np.array([1.0, 1.0, 1.0]),
        dimensions=np.array([0.5, 0.5, 0.5]),
        inflation=0.1,
    )

    # Point outside
    dist1 = box.get_distance(np.array([2.0, 1.0, 1.0]))
    # surface is at 1.25, point is at 2.0 -> dist = 0.75 - inflation(0.1) = 0.65
    assert np.isclose(dist1, 0.65)

    # Point inside (negative distance)
    dist2 = box.get_distance(np.array([1.0, 1.0, 1.0]))
    assert dist2 < 0


def test_obstacle_distance_cylinder() -> None:
    """Test distance computation for cylinder obstacle."""
    cyl = Obstacle(
        name="cyl_1",
        obstacle_type=ObstacleType.CYLINDER,
        position=np.array([0.0, 0.0, 0.0]),
        dimensions=np.array([0.5, 2.0]),  # r=0.5, h=2.0
        inflation=0.0,
    )

    # Point outside on xy plane
    dist1 = cyl.get_distance(np.array([1.0, 0.0, 0.0]))
    assert np.isclose(dist1, 0.5)

    # Point outside on z axis
    dist2 = cyl.get_distance(np.array([0.0, 0.0, 2.0]))
    assert np.isclose(dist2, 1.0)

    # Unknown type
    unk_obs = Obstacle(
        name="unk",
        obstacle_type=ObstacleType.DYNAMIC,
        position=np.array([0.0, 0.0, 0.0]),
        dimensions=np.array([1.0, 1.0, 1.0]),
    )
    assert unk_obs.get_distance(np.array([1.0, 1.0, 1.0])) == 0.0


def test_obstacle_gradient() -> None:
    """Test gradient computation."""
    sphere = Obstacle(
        name="sphere_1",
        obstacle_type=ObstacleType.SPHERE,
        position=np.array([0.0, 0.0, 0.0]),
        dimensions=np.array([1.0]),
        inflation=0.0,
    )

    point = np.array([2.0, 0.0, 0.0])
    grad = sphere.get_gradient(point)
    assert np.allclose(grad, np.array([1.0, 0.0, 0.0]))

    point2 = np.array([0.0, 2.0, 0.0])
    grad2 = sphere.get_gradient(point2)
    assert np.allclose(grad2, np.array([0.0, 1.0, 0.0]))


def test_human_state() -> None:
    """Test human state obstacle conversion."""
    human = HumanState(position=np.array([1.0, 2.0, 0.0]))
    obs = human.to_obstacle()
    assert obs.obstacle_type == ObstacleType.HUMAN
    assert np.allclose(obs.position, np.array([1.0, 2.0, 0.0]))
    assert obs.inflation == 0.3


def test_collision_avoidance_obstacles(mock_sim: MagicMock) -> None:
    """Test collision avoidance obstacle management."""
    avoid = CollisionAvoidance(robot_model=mock_sim)

    box = Obstacle(
        name="box_1",
        obstacle_type=ObstacleType.BOX,
        position=np.array([1.0, 0.0, 0.0]),
        dimensions=np.array([0.5, 0.5, 0.5]),
    )
    avoid.add_obstacle(box)
    assert len(avoid._obstacles) == 1

    # Remove existing
    assert avoid.remove_obstacle("box_1") is True
    assert len(avoid._obstacles) == 0

    # Remove non-existing
    assert avoid.remove_obstacle("box_2") is False


def test_collision_avoidance_computation(mock_sim: MagicMock) -> None:
    """Test collision avoidance algorithms."""
    avoid = CollisionAvoidance(robot_model=mock_sim)

    mock_sim.set_joint_positions = MagicMock()
    mock_sim.get_link_positions = MagicMock()
    # Mock links near obstacle
    mock_sim.get_link_positions.return_value = {"link_0": np.array([0.8, 0.0, 0.0])}

    box = Obstacle(
        name="box_1",
        obstacle_type=ObstacleType.BOX,
        position=np.array([1.0, 0.0, 0.0]),
        dimensions=np.array([0.2, 0.2, 0.2]),
        inflation=0.0,
    )
    avoid.add_obstacle(box)

    state = RobotState(
        timestamp=0.0,
        joint_positions=np.zeros(7),
        joint_velocities=np.zeros(7),
        joint_torques=np.zeros(7),
    )

    # Test checking path clearance
    traj = np.zeros((2, 7))
    is_clear, min_dist = avoid.check_path_clearance(traj, min_distance=0.5)
    # dist = 1.0 - 0.8 = 0.2 - 0.1 (half dim) = 0.1
    # 0.1 < 0.5, so it should not be clear
    assert not is_clear
    assert min_dist < 0.5

    # Test safe velocity scaling
    scale = avoid.get_safe_velocity_scaling(state)
    assert scale < 1.0

    # Test repulsion
    rep = avoid.compute_repulsive_field(state)
    assert rep is not None
    assert len(rep) == 7

    # Add a human
    human = HumanState(position=np.array([0.85, 0.0, 0.0]))  # very close
    avoid.update_human_position(human)
    scale_with_human = avoid.get_safe_velocity_scaling(state)
    assert scale_with_human == 0.0  # because it's inside inflation


def _single_link_repulsion(n_joints: int) -> np.ndarray:
    """Drive ``compute_repulsive_field`` with one link inside one obstacle.

    Uses a model that lacks ``get_link_positions`` so the fallback kinematic
    chain produces exactly ``n_joints`` links at z = 0.1*(i+1). A sphere centred
    at the first link guarantees a single in-collision contact with a known
    (max) magnitude, isolating the magnitude-scaling/averaging code path.
    """
    model = MagicMock()
    del model.get_link_positions  # force fallback chain
    del model.set_joint_positions
    avoid = CollisionAvoidance(robot_model=model)
    # Tighten the repulsion radius so only the single nearest link interacts,
    # giving a clean single-contact magnitude to assert against.
    avoid._repulsion_distance = 0.03
    # Fallback links sit on the z-axis at z = 0.1*(i+1). Place the sphere
    # offset along +x from link_0 so its distance gradient has a non-zero x
    # component (which survives the gradient[:n_joints] slice for low DOF) and
    # link_0 is inside it (distance <= 0 -> max repulsion).
    sphere = Obstacle(
        name="s",
        obstacle_type=ObstacleType.SPHERE,
        position=np.array([0.06, 0.0, 0.1]),
        dimensions=np.array([0.08]),
        inflation=0.0,
    )
    avoid.add_obstacle(sphere)
    state = RobotState(
        timestamp=0.0,
        joint_positions=np.zeros(n_joints),
        joint_velocities=np.zeros(n_joints),
        joint_torques=np.zeros(n_joints),
    )
    return avoid.compute_repulsive_field(state)


def test_repulsion_magnitude_scaled_for_low_dof() -> None:
    """Issue #6640 F1: <=3-DOF branch must apply magnitude/len scaling.

    Before the precedence fix, ``* magnitude / len(...)`` bound only to the
    ``else`` branch, so a <=3-DOF robot received the raw unit gradient
    (norm ~1) with no scaling. With the fix the repulsion is scaled by
    ``max_repulsion / n_links``.
    """
    rep = _single_link_repulsion(n_joints=2)
    assert len(rep) == 2
    norm = float(np.linalg.norm(rep))
    # max_repulsion (10.0) / n_links (2) = 5.0 (only link_0 is in collision).
    assert np.isclose(norm, 5.0), norm
    # Sanity: the buggy unscaled value would have had norm ~1.0.
    assert norm > 1.5


def test_repulsion_consistent_across_dof_branches() -> None:
    """Both the <=3 and >3 DOF branches scale by max_repulsion / n_links."""
    rep7 = _single_link_repulsion(n_joints=7)
    norm7 = float(np.linalg.norm(rep7))
    # Only link_0 in collision: max_repulsion (10.0) / 7 links.
    assert np.isclose(norm7, 10.0 / 7), norm7
    # Padding joints (index >= 3) carry no repulsion.
    assert np.allclose(rep7[3:], 0.0)


def test_collision_avoidance_no_obstacles(mock_sim: MagicMock) -> None:
    """Test behavior with no obstacles."""
    avoid = CollisionAvoidance(robot_model=mock_sim)
    state = RobotState(
        timestamp=0.0,
        joint_positions=np.zeros(7),
        joint_velocities=np.zeros(7),
        joint_torques=np.zeros(7),
    )

    # Fallback to no model get_link_positions
    avoid.model = MagicMock()
    del avoid.model.get_link_positions  # force fallback

    is_clear, min_dist = avoid.check_path_clearance(np.zeros((1, 7)))
    assert is_clear
    assert min_dist == float("inf")

    scale = avoid.get_safe_velocity_scaling(state)
    assert scale == 1.0

    dist = avoid.get_minimum_distance(state)
    assert dist == float("inf")

    rep = avoid.compute_repulsive_field(state)
    assert np.allclose(rep, np.zeros(7))
