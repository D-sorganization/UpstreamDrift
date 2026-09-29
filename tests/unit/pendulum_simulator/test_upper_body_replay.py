"""TDD contracts for constrained upper-body golfer replay (TB-06 #10591)."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.pendulum_simulator.physics_golfer import GolferParams, N_DOF
from src.shared.python.pendulum_simulator.upper_body_replay import (
    UpperBodyBernsteinTorqueProfile,
    UpperBodyBernsteinReplayTarget,
    UpperBodyCaptureFrame,
    UpperBodyMarkerAttachment,
    UpperBodyReplayTarget,
    evaluate_replay_against_body_target,
    project_replay_markers,
    replay_upper_body_target,
    replay_upper_body_bernstein_target,
)
from src.shared.python.motion_matching.body_target import BodyTarget
from src.shared.python.motion_matching.club_target import SourceProvenance

pytestmark = pytest.mark.unit


def _params() -> GolferParams:
    return GolferParams(
        m_hub=0.01,
        m_r_upper=2.0,
        m_r_fore=1.5,
        m_l_upper=2.0,
        m_l_fore=1.5,
        m_club=0.3,
        L_hub=0.05,
        L_r_upper=0.35,
        L_r_fore=0.30,
        L_l_upper=0.35,
        L_l_fore=0.30,
        L_club=1.0,
        d_rs=0.2,
        d_ls=0.2,
        grip_right=0.2,
        grip_left=0.25,
    )


def test_replay_projects_initial_state_and_reports_constraint_diagnostics() -> None:
    target = UpperBodyReplayTarget(
        times=np.array([0.0, 0.02, 0.04]),
        initial_state=np.zeros(2 * N_DOF),
        torques=np.zeros((3, 7)),
        params=_params(),
    )

    replay = replay_upper_body_target(target)

    assert np.array_equal(replay.times, target.times)
    assert replay.states.shape[1] == 2 * N_DOF
    assert replay.actuator_torques.shape[1] == N_DOF - 1
    assert replay.constraint_residual_m >= 0.0
    assert replay.reaction_forces.shape[1] == 4
    assert replay.kinematic_points["club_tip"].shape == (len(replay.times), 2)
    assert np.isfinite(replay.states).all()


def test_replay_rejects_nonmonotonic_clock() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        UpperBodyReplayTarget(
            times=np.array([0.0, 0.02, 0.02]),
            initial_state=np.zeros(2 * N_DOF),
            torques=np.zeros((3, 7)),
            params=_params(),
        )


def test_replay_rejects_nonuniform_clock_that_cannot_be_replayed_exactly() -> None:
    with pytest.raises(ValueError, match="uniformly sampled"):
        UpperBodyReplayTarget(
            times=np.array([0.0, 0.02, 0.05]),
            initial_state=np.zeros(2 * N_DOF),
            torques=np.zeros((3, 7)),
            params=_params(),
        )


def test_replay_rejects_clock_not_normalized_to_its_capture_start() -> None:
    with pytest.raises(ValueError, match="start at zero"):
        UpperBodyReplayTarget(
            times=np.array([1.0, 1.02, 1.04]),
            initial_state=np.zeros(2 * N_DOF),
            torques=np.zeros((3, 7)),
            params=_params(),
        )


def test_bernstein_torque_profile_preserves_endpoints_and_bounds() -> None:
    control_points = np.array(
        [
            np.linspace(-2.0 - actuator, 3.0 + actuator, 7)
            for actuator in range(N_DOF - 1)
        ]
    )
    profile = UpperBodyBernsteinTorqueProfile(
        control_points=control_points,
        duration_s=0.04,
    )

    assert np.allclose(profile.torque_at(0.0), control_points[:, 0])
    assert np.allclose(profile.torque_at(0.04), control_points[:, -1])
    for time_s in (-0.01, 0.01, 0.02, 0.03, 0.05):
        torque = np.asarray(profile.torque_at(time_s))
        assert np.all(torque >= control_points.min(axis=1))
        assert np.all(torque <= control_points.max(axis=1))


def test_bernstein_torque_profile_rejects_wrong_actuator_shape() -> None:
    with pytest.raises(ValueError, match=r"shape \(7, 7\)"):
        UpperBodyBernsteinTorqueProfile(
            control_points=np.zeros((N_DOF - 2, 7)),
            duration_s=0.04,
        )


def test_bernstein_replay_rejects_profile_with_wrong_horizon() -> None:
    profile = UpperBodyBernsteinTorqueProfile(
        control_points=np.zeros((N_DOF - 1, 7)),
        duration_s=0.03,
    )

    with pytest.raises(ValueError, match="duration_s must match"):
        UpperBodyBernsteinReplayTarget(
            times=np.array([0.0, 0.02, 0.04]),
            initial_state=np.zeros(2 * N_DOF),
            torque_profile=profile,
            params=_params(),
        )


def test_bernstein_replay_uses_continuous_bounded_control_profile() -> None:
    control_points = np.array(
        [np.linspace(0.0, 0.02 * (actuator + 1), 7) for actuator in range(N_DOF - 1)]
    )
    target = UpperBodyBernsteinReplayTarget(
        times=np.array([0.0, 0.02, 0.04]),
        initial_state=np.zeros(2 * N_DOF),
        torque_profile=UpperBodyBernsteinTorqueProfile(
            control_points=control_points,
            duration_s=0.04,
        ),
        params=_params(),
    )

    replay = replay_upper_body_bernstein_target(target)

    assert np.allclose(replay.actuator_torques[0], control_points[:, 0])
    assert np.allclose(replay.actuator_torques[-1], control_points[:, -1])
    assert np.all(replay.actuator_torques >= control_points.min(axis=1))
    assert np.all(replay.actuator_torques <= control_points.max(axis=1))
    assert replay.reaction_forces.shape[1] == 4


def test_marker_projection_requires_explicit_attachments_and_capture_frame() -> None:
    target = UpperBodyReplayTarget(
        times=np.array([0.0, 0.02, 0.04]),
        initial_state=np.zeros(2 * N_DOF),
        torques=np.zeros((3, 7)),
        params=_params(),
    )

    replay = replay_upper_body_target(target)
    projected = project_replay_markers(
        replay,
        attachments=(
            UpperBodyMarkerAttachment("right_shoulder", "rs", np.zeros(2)),
            UpperBodyMarkerAttachment("right_elbow", "re", np.zeros(2)),
            UpperBodyMarkerAttachment("clubhead", "club_tip", np.zeros(2)),
        ),
        capture_frame=UpperBodyCaptureFrame(
            origin_m=np.zeros(3),
            plane_basis=np.array([[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]]),
        ),
    )

    assert projected.shape == (len(replay.times), 3, 3)
    assert np.allclose(projected[:, 0, 1], 0.0)


def test_marker_attachment_rejects_nonplanar_offset() -> None:
    with pytest.raises(ValueError, match=r"shape \(2,\)"):
        UpperBodyMarkerAttachment("clubhead", "club_tip", np.zeros(3))


def test_replay_evaluation_uses_declared_body_target_clock_and_markers() -> None:
    target = UpperBodyReplayTarget(
        times=np.array([0.0, 0.02, 0.04]),
        initial_state=np.zeros(2 * N_DOF),
        torques=np.zeros((3, 7)),
        params=_params(),
    )
    replay = replay_upper_body_target(target)
    attachments = (
        UpperBodyMarkerAttachment("right_shoulder", "rs", np.zeros(2)),
        UpperBodyMarkerAttachment("right_elbow", "re", np.zeros(2)),
        UpperBodyMarkerAttachment("clubhead", "club_tip", np.zeros(2)),
    )
    capture_frame = UpperBodyCaptureFrame(
        origin_m=np.zeros(3),
        plane_basis=np.array([[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]]),
    )
    observed = project_replay_markers(replay, attachments, capture_frame)
    body_target = BodyTarget(
        time=replay.times,
        marker_xyz=observed,
        marker_names=tuple(attachment.label for attachment in attachments),
        impact_idx=1,
        events=(),
        source=SourceProvenance("known.c3d", "c3d", "test", "test", "0" * 64),
    )

    metrics = evaluate_replay_against_body_target(
        replay, body_target, attachments, capture_frame
    )

    assert metrics.whole_marker_rmse_m == pytest.approx(0.0)
    assert metrics.n_valid == observed.shape[0] * observed.shape[1]
