from __future__ import annotations

import numpy as np

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
