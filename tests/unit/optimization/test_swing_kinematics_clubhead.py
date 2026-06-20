"""Parity tests for the vectorized compute_clubhead_trajectory (issue #7714).

The per-frame Python loop was replaced with vectorized numpy. This test
pins the vectorized output to an independent per-frame reference
implementation so the optimization is provably numerically identical.
"""

import numpy as np
import pytest

pytestmark = pytest.mark.unit

from src.shared.python.optimization._swing_kinematics import (
    compute_clubhead_trajectory,
)
from src.shared.python.optimization._swing_models import ClubModel, GolferModel


def _reference_clubhead_trajectory(
    joint_angles: dict[str, np.ndarray],
    time: np.ndarray,
    golfer: GolferModel,
    club: ClubModel,
) -> tuple[np.ndarray, np.ndarray]:
    """Original per-frame loop, kept as the parity reference."""
    n_frames = len(time)
    position = np.zeros((n_frames, 3))
    velocity = np.zeros((n_frames, 3))

    arm_length = golfer.arm_length
    club_length = club.total_length

    for i in range(n_frames):
        trunk_rot = joint_angles.get("trunk_rotation", np.zeros(n_frames))[i]
        shoulder_h = joint_angles.get("shoulder_horizontal", np.zeros(n_frames))[i]
        wrist = joint_angles.get("wrist_cock", np.zeros(n_frames))[i]

        total_angle = trunk_rot + shoulder_h + wrist

        position[i, 0] = (arm_length + club_length) * np.sin(total_angle)
        position[i, 1] = 0
        position[i, 2] = (arm_length + club_length) * np.cos(total_angle) - club_length

    dt = time[1] - time[0] if len(time) > 1 else 0.001
    for dim in range(3):
        velocity[:, dim] = np.gradient(position[:, dim], dt)

    return position, velocity


def test_vectorized_matches_reference_multiframe() -> None:
    rng = np.random.default_rng(7714)
    n_frames = 50
    time = np.linspace(0.0, 0.3, n_frames)
    joint_angles = {
        "trunk_rotation": rng.uniform(-1.5, 1.5, n_frames),
        "shoulder_horizontal": rng.uniform(-2.0, 2.0, n_frames),
        "shoulder_vertical": rng.uniform(-1.0, 1.0, n_frames),
        "elbow_flexion": rng.uniform(0.0, 2.4, n_frames),
        "wrist_cock": rng.uniform(-1.2, 1.2, n_frames),
    }
    golfer = GolferModel()
    club = ClubModel()

    pos, vel = compute_clubhead_trajectory(joint_angles, time, golfer, club)
    ref_pos, ref_vel = _reference_clubhead_trajectory(joint_angles, time, golfer, club)

    np.testing.assert_allclose(pos, ref_pos, atol=1e-10, rtol=0.0)
    np.testing.assert_allclose(vel, ref_vel, atol=1e-10, rtol=0.0)


def test_vectorized_matches_reference_missing_keys() -> None:
    """Defaulted (missing) joints must still match the reference."""
    n_frames = 12
    time = np.linspace(0.0, 0.2, n_frames)
    joint_angles = {"trunk_rotation": np.linspace(-1.0, 1.0, n_frames)}
    golfer = GolferModel()
    club = ClubModel()

    pos, vel = compute_clubhead_trajectory(joint_angles, time, golfer, club)
    ref_pos, ref_vel = _reference_clubhead_trajectory(joint_angles, time, golfer, club)

    np.testing.assert_allclose(pos, ref_pos, atol=1e-10, rtol=0.0)
    np.testing.assert_allclose(vel, ref_vel, atol=1e-10, rtol=0.0)
    # Y component is identically zero.
    assert np.all(pos[:, 1] == 0.0)
