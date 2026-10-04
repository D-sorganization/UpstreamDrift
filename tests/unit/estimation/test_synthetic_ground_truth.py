from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.estimation.residuals import project_pinhole
from src.shared.python.estimation.synthetic_fixtures import (
    make_fixture_cameras,
    make_two_link_trajectory,
)
from src.shared.python.estimation.synthetic_ground_truth import (
    NoiseModel,
    ObservationPolicy,
    SkeletonRigForwardModel,
    SyntheticCamera,
    SyntheticObservationRig,
    project_world_point,
)
from src.shared.python.motion_pipeline import (
    CameraExtrinsics,
    CameraIntrinsics,
)


def test_fixture_cameras_are_proper_rotations() -> None:
    """Positive: every fixture extrinsic validates and is a proper rotation."""
    for _camera_id, _intrinsics, extrinsics in make_fixture_cameras():
        rotation = np.asarray(extrinsics.rotation, dtype=np.float64)
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-9)
        assert np.isclose(np.linalg.det(rotation), 1.0, atol=1e-9)


def test_cam1_uses_negative_fifteen_degree_yaw() -> None:
    """cam1 must encode a −15° rotation about Y (R_y(-15°)), not +15°.

    Regression guard for review feedback on merged PR #7069 (issue #7071):
    the original non-orthonormal fixture used a −15° world-to-camera rotation
    ([cos, 0, -sin] / [0,1,0] / [sin, 0, cos]).  The fix must preserve the
    same geometric orientation, not mirror it.

    For R_y(-15°):  R[0][2] = sin(-15°) < 0  and  R[2][0] = -sin(-15°) > 0.
    """
    cameras = make_fixture_cameras()
    _cam_id, _intrinsics, cam1_extrinsics = cameras[1]
    rotation = np.asarray(cam1_extrinsics.rotation, dtype=np.float64)
    assert rotation[0][2] < 0, (
        f"cam1 R[0][2]={rotation[0][2]:.6f}: expected negative (−sin 15°). "
        "The fixture must use −15° yaw, not +15°."
    )
    assert rotation[2][0] > 0, (
        f"cam1 R[2][0]={rotation[2][0]:.6f}: expected positive (+sin 15°). "
        "The fixture must use −15° yaw, not +15°."
    )


def test_camera_extrinsics_rejects_non_orthonormal_rotation() -> None:
    """Negative: the legacy non-orthonormal fixture matrix is rejected."""
    with pytest.raises(ValueError, match="orthonormal"):
        CameraExtrinsics(
            rotation=[
                [0.9659, 0.0, -0.2588],
                [0.0, 1.0, 0.0],
                [0.2588, 0.0, 0.9659],
            ],
            translation=[0.2, 0.0, 0.0],
        )


def test_project_world_point_matches_pinhole_with_nonzero_k3() -> None:
    """project_world_point must equal project_pinhole's 5-term radial model."""
    intrinsics = CameraIntrinsics(
        fx=800.0, fy=820.0, cx=640.0, cy=480.0, k1=0.12, k2=-0.05, k3=0.03
    )
    camera = SyntheticCamera("cam", intrinsics, CameraExtrinsics())
    point = np.array([0.2, -0.1, 2.0])

    x_px, y_px, _depth = project_world_point(camera, point)

    camera_matrix = np.array(
        [
            [intrinsics.fx, 0.0, intrinsics.cx],
            [0.0, intrinsics.fy, intrinsics.cy],
            [0.0, 0.0, 1.0],
        ]
    )
    distortion = np.array(
        [intrinsics.k1, intrinsics.k2, intrinsics.p1, intrinsics.p2, intrinsics.k3]
    )
    expected = project_pinhole(
        point.reshape(1, 3),
        camera_matrix,
        distortion=distortion,
    )[0]

    np.testing.assert_allclose([x_px, y_px], expected, atol=1e-9)


def test_project_world_point_unchanged_for_zero_k3() -> None:
    """The 5-term model collapses to the 4-term model when k3 == 0."""
    intrinsics = CameraIntrinsics(
        fx=800.0, fy=820.0, cx=640.0, cy=480.0, k1=0.12, k2=-0.05
    )
    camera = SyntheticCamera("cam", intrinsics, CameraExtrinsics())
    point = np.array([0.2, -0.1, 2.0])

    x_px, y_px, _depth = project_world_point(camera, point)

    camera_matrix = np.array(
        [
            [intrinsics.fx, 0.0, intrinsics.cx],
            [0.0, intrinsics.fy, intrinsics.cy],
            [0.0, 0.0, 1.0],
        ]
    )
    distortion = np.array([intrinsics.k1, intrinsics.k2, intrinsics.p1, intrinsics.p2])
    expected = project_pinhole(
        point.reshape(1, 3),
        camera_matrix,
        distortion=distortion,
    )[0]

    np.testing.assert_allclose([x_px, y_px], expected, atol=1e-9)


def test_project_world_point_uses_intrinsics_and_depth() -> None:
    camera_id, intrinsics, extrinsics = make_fixture_cameras()[0]
    camera = SyntheticCamera(camera_id, intrinsics, extrinsics)

    x_px, y_px, depth = project_world_point(camera, np.array([0.2, -0.1, 2.0]))

    assert x_px == 720.0
    assert y_px == 440.0
    assert depth == 2.0


def test_synthetic_rig_generates_reproducible_multi_camera_observations() -> None:
    trajectory = make_two_link_trajectory(n_frames=4)
    cameras = tuple(SyntheticCamera(*payload) for payload in make_fixture_cameras())
    forward_model = SkeletonRigForwardModel(trajectory.skeleton)
    rig = SyntheticObservationRig(cameras, forward_model, seed=123)

    result = rig.generate(trajectory)

    assert set(result.observations_by_camera) == {"cam0", "cam1"}
    assert result.ground_truth_markers.num_frames == 4
    assert len(result.projection_records) == 2 * 4 * trajectory.skeleton.num_joints
    cam0_frame0 = result.observations_by_camera["cam0"].frames[0]
    assert [point.name for point in cam0_frame0.keypoints] == [
        "root",
        "elbow",
        "wrist",
    ]
    assert all(point.confidence == 1.0 for point in cam0_frame0.keypoints)


def test_noise_is_seeded_and_applied_in_pixel_space() -> None:
    trajectory = make_two_link_trajectory(n_frames=2)
    cameras = tuple(SyntheticCamera(*payload) for payload in make_fixture_cameras()[:1])
    forward_model = SkeletonRigForwardModel(trajectory.skeleton)
    clean = SyntheticObservationRig(cameras, forward_model, seed=7).generate(trajectory)
    noisy_a = SyntheticObservationRig(
        cameras, forward_model, noise=NoiseModel(sigma_px=0.25), seed=7
    ).generate(trajectory)
    noisy_b = SyntheticObservationRig(
        cameras, forward_model, noise=NoiseModel(sigma_px=0.25), seed=7
    ).generate(trajectory)

    clean_x = clean.observations_by_camera["cam0"].frames[0].keypoints[1].x
    noisy_x = noisy_a.observations_by_camera["cam0"].frames[0].keypoints[1].x
    repeat_x = noisy_b.observations_by_camera["cam0"].frames[0].keypoints[1].x

    assert noisy_x != clean_x
    assert noisy_x == repeat_x


def test_occlusion_and_dropout_lower_visibility_confidence() -> None:
    trajectory = make_two_link_trajectory(n_frames=1)
    cameras = tuple(SyntheticCamera(*payload) for payload in make_fixture_cameras()[:1])
    forward_model = SkeletonRigForwardModel(trajectory.skeleton)
    policy = ObservationPolicy(
        dropout_probability=0.0,
        occlusion_radius_m=0.05,
        occluder_centers_m=((0.0, 0.0, 3.0),),
    )
    rig = SyntheticObservationRig(cameras, forward_model, policy=policy, seed=4)

    result = rig.generate(trajectory)

    root_record = next(
        record for record in result.projection_records if record.name == "root"
    )
    root_keypoint = result.observations_by_camera["cam0"].frames[0].keypoints[0]
    assert root_record.occluded is True
    assert root_record.dropped is False
    assert root_keypoint.confidence == 0.0


def test_dropout_removes_keypoints_but_keeps_projection_records() -> None:
    trajectory = make_two_link_trajectory(n_frames=1)
    cameras = tuple(SyntheticCamera(*payload) for payload in make_fixture_cameras()[:1])
    forward_model = SkeletonRigForwardModel(trajectory.skeleton)
    policy = ObservationPolicy(dropout_probability=1.0)
    rig = SyntheticObservationRig(cameras, forward_model, policy=policy, seed=1)

    result = rig.generate(trajectory)

    assert all(record.dropped for record in result.projection_records)
    frame = result.observations_by_camera["cam0"].frames[0]
    assert len(frame.keypoints) == 1
    assert frame.keypoints[0].name == "empty"
    assert frame.keypoints[0].confidence == 0.0


@pytest.mark.unit
def test_shared_fixed_base_pendulum_fixture_specification() -> None:
    """Verify deterministic fixed-base pendulum initial state, controls, sampling, and truth."""
    from src.shared.python.estimation.synthetic_fixtures import (
        make_fixed_base_pendulum_fixture,
    )

    n_frames = 12
    fps = 100.0
    pendulum = make_fixed_base_pendulum_fixture(
        n_frames=n_frames, fps=fps, theta_0=0.15
    )

    assert pendulum.units == "m"
    assert pendulum.sampling_rate_hz == 100.0
    assert pendulum.initial_state == (0.15, 0.0)
    assert len(pendulum.frames) == n_frames
    assert np.all(pendulum.controls == 0.0)
    assert "Harmonic linear oscillator" in pendulum.exact_truth_derivation

    # Ensure camera projection works with the pendulum skeleton
    cameras = tuple(SyntheticCamera(*p) for p in make_fixture_cameras()[:1])
    forward_model = SkeletonRigForwardModel(pendulum.skeleton)
    rig = SyntheticObservationRig(cameras, forward_model, seed=42)
    obs = rig.generate(pendulum.trajectory)
    assert len(obs.projection_records) == n_frames * pendulum.skeleton.num_joints


@pytest.mark.unit
def test_shared_underactuated_analytic_fixture_specification() -> None:
    """Verify underactuated analytic fixture passive root, controls, and dynamic coupling."""
    from src.shared.python.estimation.synthetic_fixtures import (
        make_underactuated_analytic_fixture,
    )

    n_frames = 15
    fps = 60.0
    fixture = make_underactuated_analytic_fixture(n_frames=n_frames, fps=fps)

    assert fixture.units == "m"
    assert fixture.sampling_rate_hz == 60.0
    assert fixture.is_underactuated is True
    assert fixture.unactuated_dofs == (0,)
    assert fixture.degree_of_underactuation == 1
    assert np.all(fixture.controls[:, 0] == 0.0)
    assert np.any(fixture.controls[:, 1] != 0.0)
    assert "Joint 0 passive" in fixture.exact_truth_derivation


@pytest.mark.unit
def test_shared_native_stance_fixture_specification() -> None:
    """Verify native stance fixture equilibrium contact force GRF_z = m*g and holding torques."""
    from src.shared.python.estimation.synthetic_fixtures import (
        make_native_stance_fixture,
    )

    mass = 80.0
    g = 9.81
    stance = make_native_stance_fixture(
        n_frames=8, fps=50.0, mass_kg=mass, gravity_m_s2=g
    )

    assert stance.units == "m"
    assert stance.sampling_rate_hz == 50.0
    assert stance.is_stance is True
    assert np.isclose(stance.ground_reaction_force_z, mass * g, atol=1e-5)
    assert stance.ground_reaction_force_xy == (0.0, 0.0)
    assert len(stance.frames) == 8
    assert "GRF_z = mass*g" in stance.exact_truth_derivation
