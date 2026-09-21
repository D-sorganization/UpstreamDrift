"""Unit tests for render_humanoid_overlay module."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from src.engines.physics_engines.mujoco.python.motion_matching.render_humanoid_overlay import (
    HumanoidSkeletalTopology,
    HumanoidTrajectoryData,
    render_humanoid_frame,
    render_humanoid_video,
)


@pytest.mark.unit
def test_humanoid_skeletal_topology_filters_correctly() -> None:
    topology = HumanoidSkeletalTopology()
    available = ["pelvis", "lower_torso", "upper_torso", "head", "left_shoulder"]
    filtered = topology.filter_bones_for_bodies(available)

    assert ("pelvis", "lower_torso") in filtered
    assert ("lower_torso", "upper_torso") in filtered
    assert ("upper_torso", "head") in filtered
    assert ("upper_torso", "left_shoulder") in filtered
    assert ("left_shoulder", "left_elbow") not in filtered


@pytest.mark.unit
def test_humanoid_trajectory_data_validation() -> None:
    times = np.array([0.0, 0.02, 0.04])
    bodies = ["pelvis", "club", "clubhead"]
    pos = np.zeros((3, 3, 3))

    data = HumanoidTrajectoryData(
        times=times,
        body_names=bodies,
        body_positions=pos,
        grip_index=1,
        clubhead_index=2,
    )

    assert data.frame_count == 3
    assert data.grip_points.shape == (3, 3)
    assert data.clubhead_points.shape == (3, 3)

    with pytest.raises(ValueError, match="times must be a non-empty 1D array"):
        HumanoidTrajectoryData(
            times=np.array([]),
            body_names=bodies,
            body_positions=pos,
            grip_index=1,
            clubhead_index=2,
        )

    with pytest.raises(IndexError, match="grip_index out of range"):
        HumanoidTrajectoryData(
            times=times,
            body_names=bodies,
            body_positions=pos,
            grip_index=5,
            clubhead_index=2,
        )


@pytest.mark.unit
def test_render_humanoid_frame_executes(tmp_path: pytest.TempPathFactory) -> None:
    times = np.array([0.0, 0.02])
    bodies = ["pelvis", "lower_torso", "club", "clubhead"]
    pos = np.zeros((2, 4, 3))

    data = HumanoidTrajectoryData(
        times=times,
        body_names=bodies,
        body_positions=pos,
        grip_index=2,
        clubhead_index=3,
    )

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    render_humanoid_frame(data, 0, ax)
    plt.close(fig)


@pytest.mark.unit
def test_render_humanoid_video_gif(tmp_path: pytest.TempPathFactory) -> None:
    times = np.array([0.0, 0.02, 0.04])
    bodies = ["pelvis", "lower_torso", "club", "clubhead"]
    pos = np.zeros((3, 4, 3))

    data = HumanoidTrajectoryData(
        times=times,
        body_names=bodies,
        body_positions=pos,
        grip_index=2,
        clubhead_index=3,
    )

    out_gif = tmp_path / "test_humanoid.gif"
    out_path = render_humanoid_video(data, out_gif, fps=10, format="gif")
    assert out_path.exists()
    assert out_path.stat().st_size > 0
