"""Tests for reference scene registration, synchronization, and projection (#9865)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.motion_capture.reconstruct.cameras import (
    PinholeCamera,
    intrinsics_from_fov,
    look_at,
)
from src.motion_capture.reference.model import (
    ReferenceMotion,
    ReferenceSource,
    ReferenceVideo,
)
from src.motion_capture.reference.registration import (
    EventAnchors,
    ReferenceRegistration,
    ReferenceTransform,
    TimeMapping,
    project_reference_to_camera,
    sample_reference_motion,
    transform_reference_motion,
)
from src.shared.python.pose_estimation.observations import (
    CameraCalibration,
    CameraExtrinsics,
    CameraIntrinsics,
)

pytestmark = pytest.mark.unit


def sample_motion() -> ReferenceMotion:
    return ReferenceMotion.model_validate(
        {
            "title": "Pro Driver Benchmark",
            "source": ReferenceSource(path="pro.c3d", sha256="c" * 64, format="c3d"),
            "source_units": "m",
            "source_axes": ("+X", "+Y", "+Z"),
            "source_names": ("pelvis", "lead_wrist", "clubhead"),
            "joint_names": ("pelvis", "wrist", "clubhead"),
            "edges": ((0, 1), (1, 2)),
            "time_s": (0.0, 0.5, 1.0, 1.5, 2.0),
            "points_m": (
                # frame 0: address
                ((0.0, 0.0, 1.0), (0.2, 0.0, 1.0), (0.5, 0.0, 0.0)),
                # frame 1: top of backswing
                ((0.0, 0.0, 1.0), (-0.2, 0.0, 1.6), (-0.4, 0.0, 1.8)),
                # frame 2: impact (wrist unobserved / occluded gap)
                ((0.0, 0.0, 1.0), None, (0.5, 0.0, 0.0)),
                # frame 3: finish
                ((0.0, 0.0, 1.0), (-0.3, 0.0, 1.5), (-0.5, 0.0, 1.6)),
                # frame 4: post-finish
                ((0.0, 0.0, 1.0), (-0.2, 0.0, 1.2), (-0.2, 0.0, 0.5)),
            ),
        }
    )


def two_camera_rig() -> tuple[PinholeCamera, PinholeCamera]:
    size = (1920, 1080)
    # Face-on camera: positioned at world (0, 1, 3.5), looking at (0, 1, 0)
    cam_fo_pos = np.array([0.0, 1.0, 3.5])
    cam_fo = PinholeCamera(
        camera_id="face_on",
        matrix=intrinsics_from_fov(size[0], size[1], 60.0),
        rotation_world_from_camera=look_at(cam_fo_pos, np.array([0.0, 1.0, 0.0])),
        translation_world_from_camera_m=cam_fo_pos,
        image_size_px=size,
    )
    # Down-the-line camera: positioned at world (-3.5, 1.0, 0.0), looking at (0, 1, 0)
    cam_dtl_pos = np.array([-3.5, 1.0, 0.0])
    cam_dtl = PinholeCamera(
        camera_id="down_the_line",
        matrix=intrinsics_from_fov(size[0], size[1], 60.0),
        rotation_world_from_camera=look_at(cam_dtl_pos, np.array([0.0, 1.0, 0.0])),
        translation_world_from_camera_m=cam_dtl_pos,
        image_size_px=size,
    )
    return cam_fo, cam_dtl


def test_reference_transform_applies_rotation_scale_and_translation() -> None:
    # 90 deg rotation around canonical Z, scale 2.0, translation (1, 2, 3)
    rot_z90 = (
        (0.0, -1.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
    )
    tf = ReferenceTransform(
        rotation=rot_z90,
        translation_m=(1.0, 2.0, 3.0),
        scale=2.0,
    )
    pts = np.array([[1.0, 0.0, 0.5]])
    # R @ [1, 0, 0.5] = [0, 1, 0.5]
    # s * R @ p = [0, 2, 1]
    # + t = [1, 4, 4]
    transformed = tf.apply(pts)
    np.testing.assert_allclose(transformed, [[1.0, 4.0, 4.0]], atol=1e-9)


def test_time_mapping_offset_and_event_anchors() -> None:
    # Test 1: Simple linear offset
    tm_simple = TimeMapping(offset_s=10.0, rate_scale=1.0)
    assert tm_simple.reference_to_scene(1.5) == pytest.approx(11.5)
    assert tm_simple.scene_to_reference(11.5) == pytest.approx(1.5)

    # Test 2: Event anchors piecewise warping
    # Reference events at t = 0.0 (address), 0.5 (top), 1.0 (impact), 1.5 (finish)
    # Scene events at t = 5.0, 6.0, 6.8, 8.0
    anchors = EventAnchors(
        reference={"address": 0.0, "top": 0.5, "impact": 1.0, "finish": 1.5},
        scene={"address": 5.0, "top": 6.0, "impact": 6.8, "finish": 8.0},
    )
    tm_warped = TimeMapping(event_anchors=anchors)
    # Midpoint between address and top in ref: 0.25 -> scene: 5.5
    assert tm_warped.reference_to_scene(0.25) == pytest.approx(5.5)
    # Midpoint between top and impact in ref: 0.75 -> scene: 6.4
    assert tm_warped.reference_to_scene(0.75) == pytest.approx(6.4)
    # Inverse mapping
    assert tm_warped.scene_to_reference(6.4) == pytest.approx(0.75)


def test_sample_reference_motion_preserves_missing_joint_masks_across_gaps() -> None:
    motion = sample_motion()
    reg = ReferenceRegistration(
        reference_id=motion.id,
        calibration_id="rig_01",
        max_gap_s=0.5,
        transform=ReferenceTransform(),
        time_mapping=TimeMapping(offset_s=0.0),
    )
    # Sample at scene times: 0.0, 0.5, 0.75, 1.0, 1.25, 1.5
    scene_times = np.array([0.0, 0.5, 0.75, 1.0, 1.25, 1.5])
    sampled_pts, valid_mask = sample_reference_motion(motion, reg, scene_times)

    # Joint 1 (wrist) is observed at 0.5 and missing at 1.0.
    # At t = 0.5: valid
    assert bool(valid_mask[1, 1]) is True
    # At t = 0.75 (between 0.5 and 1.0): one bound is None -> MUST be masked as invalid!
    assert bool(valid_mask[2, 1]) is False
    # At t = 1.0: missing -> invalid
    assert bool(valid_mask[3, 1]) is False
    # At t = 1.25 (between 1.0 and 1.5): one bound is None -> invalid!
    assert bool(valid_mask[4, 1]) is False
    # At t = 1.5: observed -> valid
    assert bool(valid_mask[5, 1]) is True

    # Joint 0 (pelvis) is observed at all times
    assert np.all(valid_mask[:, 0])


def test_camera_projection_with_distortion_and_clipping() -> None:
    cam_fo, cam_dtl = two_camera_rig()
    motion = sample_motion()
    reg = ReferenceRegistration(
        reference_id=motion.id,
        calibration_id="rig_01",
        max_gap_s=0.5,
        transform=ReferenceTransform(translation_m=(0.0, 0.0, 0.0)),
        time_mapping=TimeMapping(offset_s=0.0),
    )

    # Known point at (0, 1, 0) in world coordinates (the target)
    pts_world = np.array(
        [[[0.0, 1.0, 0.0], [0.0, 1.0, 5.0]]]
    )  # second point behind camera
    valid = np.array([[True, True]])

    # Pure pinhole projection
    px, vis = project_reference_to_camera(pts_world, valid, cam_fo)
    # Point 0 is at target centre: (1920/2, 1080/2) = (960, 540)
    assert bool(vis[0, 0]) is True
    np.testing.assert_allclose(px[0, 0], [960.0, 540.0], atol=1e-3)
    # Point 1 is behind camera: clipped
    assert bool(vis[0, 1]) is False

    # Projection with Brown-Conrady distortion
    dist_k = (-0.15, 0.05, 0.001, -0.001)
    calib = CameraCalibration(
        camera_id=cam_fo.camera_id,
        intrinsics=CameraIntrinsics(matrix=cam_fo.matrix, distortion=np.array(dist_k)),
        extrinsics=CameraExtrinsics(
            rotation_world_from_camera=cam_fo.rotation_world_from_camera,
            translation_world_from_camera_m=cam_fo.translation_world_from_camera_m,
        ),
        image_size_px=cam_fo.image_size_px,
    )
    px_dist, vis_dist = project_reference_to_camera(pts_world, valid, calib)
    # Center pixel unaffected by radial distortion
    np.testing.assert_allclose(px_dist[0, 0], [960.0, 540.0], atol=1e-2)


def test_serialization_round_trip_yields_identical_pixels_and_times() -> None:
    cam_fo, _ = two_camera_rig()
    motion = sample_motion()
    rot = (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )
    reg = ReferenceRegistration(
        reference_id=motion.id,
        calibration_id="rig_01",
        max_gap_s=0.5,
        transform=ReferenceTransform(
            rotation=rot,
            translation_m=(0.1, 0.2, 0.3),
            scale=1.05,
            body_size_normalized=True,
        ),
        time_mapping=TimeMapping(
            offset_s=0.25,
            rate_scale=1.02,
        ),
        assumption_labels=("proportional_limb_scale", "strobe_aligned"),
        is_calibrated=True,
    )

    serialized = reg.model_dump_json(indent=2)
    deserialized = ReferenceRegistration.model_validate_json(serialized)
    assert deserialized == reg

    # Sample and project with both
    times = np.array([0.25, 0.75, 1.25])
    pts_1, mask_1 = sample_reference_motion(motion, reg, times)
    pts_2, mask_2 = sample_reference_motion(motion, deserialized, times)
    np.testing.assert_allclose(pts_1, pts_2)
    assert np.array_equal(mask_1, mask_2)

    px_1, vis_1 = project_reference_to_camera(pts_1, mask_1, cam_fo)
    px_2, vis_2 = project_reference_to_camera(pts_2, mask_2, cam_fo)
    np.testing.assert_allclose(px_1, px_2)
    assert np.array_equal(vis_1, vis_2)


def test_2d_expert_video_registration_uses_manual_image_transform_without_3d_claims() -> (
    None
):
    video = ReferenceVideo.model_validate(
        {
            "title": "PGA Tour Slowmo",
            "source": ReferenceSource(path="pga.mp4", sha256="d" * 64, format="video"),
            "width": 1920,
            "height": 1080,
            "frames": 300,
            "fps": 60.0,
        }
    )
    # Affine 2D homography (3x3): scale and translation in pixel coordinates
    h_2d = (
        (0.9, 0.0, 50.0),
        (0.0, 0.9, 30.0),
        (0.0, 0.0, 1.0),
    )
    reg_2d = ReferenceRegistration(
        reference_id=video.id,
        calibration_id="uncalibrated_2d",
        transform=ReferenceTransform(),
        time_mapping=TimeMapping(offset_s=1.2),
        image_transform_2d=h_2d,
        assumption_labels=("manual_2d_fit", "no_3d_claim"),
        is_calibrated=False,
    )
    assert reg_2d.is_calibrated is False
    assert reg_2d.image_transform_2d is not None
    assert "no_3d_claim" in reg_2d.assumption_labels


def test_reference_track_generates_overlay_track() -> None:
    from src.motion_capture.reconstruct.overlay3d import reference_track

    cam_fo, _ = two_camera_rig()
    motion = sample_motion()
    reg = ReferenceRegistration(
        reference_id=motion.id,
        calibration_id="rig_01",
        max_gap_s=0.5,
        transform=ReferenceTransform(translation_m=(0.0, 0.0, 0.0)),
        time_mapping=TimeMapping(offset_s=0.0),
    )
    scene_times = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    track = reference_track(reg, motion, cam_fo, scene_times, (0, 255, 0))

    assert track.kind == "reference"
    assert track.frames == 5
    assert track.px.shape == (5, 3, 2)
    assert track.names == ("pelvis", "wrist", "clubhead")
    assert track.edges == ((0, 1), (1, 2))
    # At t = 1.0 (frame 2), wrist was missing in sample_motion -> visible must be False
    assert bool(track.visible[2, 1]) is False
    # At frame 0, pelvis is at target center -> (960, 540)
    assert bool(track.visible[0, 0]) is True
    np.testing.assert_allclose(track.px[0, 0], [960.0, 540.0], atol=1e-2)


def test_event_anchors_rejects_non_monotonic_or_unbounded_rates() -> None:
    # Repeated reference timestamps
    with pytest.raises(ValueError, match="strictly monotonic"):
        EventAnchors(
            reference={"address": 0.0, "impact": 0.0},
            scene={"address": 1.0, "impact": 2.0},
        )

    # Reversed reference timestamps
    with pytest.raises(ValueError, match="strictly monotonic"):
        EventAnchors(
            reference={"address": 1.0, "impact": 0.5},
            scene={"address": 1.0, "impact": 2.0},
        )

    # Reversed scene timestamps
    with pytest.raises(ValueError, match="strictly monotonic"):
        EventAnchors(
            reference={"address": 0.0, "impact": 1.0},
            scene={"address": 2.0, "impact": 1.0},
        )

    # Unbounded rate (dt_scene / dt_ref > 100.0)
    with pytest.raises(ValueError, match="bounded within"):
        EventAnchors(
            reference={"address": 0.0, "impact": 0.01},
            scene={"address": 0.0, "impact": 2.0},  # rate = 200.0
        )

    # Unbounded rate (dt_scene / dt_ref < 0.01)
    with pytest.raises(ValueError, match="bounded within"):
        EventAnchors(
            reference={"address": 0.0, "impact": 10.0},
            scene={"address": 0.0, "impact": 0.05},  # rate = 0.005
        )


def test_single_anchor_alignment_and_scalar_array_round_trip() -> None:
    # Exactly one anchor must align that event as a pure offset
    anchors = EventAnchors(
        reference={"impact": 1.2},
        scene={"impact": 5.7},
    )
    tm = TimeMapping(event_anchors=anchors, rate_scale=1.5)
    # At t_ref = 1.2, t_scene MUST be exactly 5.7
    assert tm.reference_to_scene(1.2) == pytest.approx(5.7)
    assert tm.scene_to_reference(5.7) == pytest.approx(1.2)

    # Test arbitrary scalars
    t_ref_val = 2.0
    t_scene_val = tm.reference_to_scene(t_ref_val)
    assert tm.scene_to_reference(t_scene_val) == pytest.approx(t_ref_val)

    # Test array round trip
    ref_arr = np.array([-1.0, 0.0, 1.2, 3.5, 10.0])
    scene_arr = tm.reference_to_scene(ref_arr)
    assert isinstance(scene_arr, np.ndarray)
    np.testing.assert_allclose(tm.scene_to_reference(scene_arr), ref_arr)


def test_multi_anchor_piecewise_warping_round_trip_and_extrapolation() -> None:
    anchors = EventAnchors(
        reference={"address": 0.0, "top": 0.5, "impact": 1.0, "finish": 1.5},
        scene={"address": 5.0, "top": 5.8, "impact": 6.5, "finish": 8.0},
    )
    tm = TimeMapping(event_anchors=anchors)

    # Endpoints exact match
    assert tm.reference_to_scene(0.0) == pytest.approx(5.0)
    assert tm.reference_to_scene(0.5) == pytest.approx(5.8)
    assert tm.reference_to_scene(1.0) == pytest.approx(6.5)
    assert tm.reference_to_scene(1.5) == pytest.approx(8.0)

    # Inverses exact match
    assert tm.scene_to_reference(5.0) == pytest.approx(0.0)
    assert tm.scene_to_reference(5.8) == pytest.approx(0.5)
    assert tm.scene_to_reference(6.5) == pytest.approx(1.0)
    assert tm.scene_to_reference(8.0) == pytest.approx(1.5)

    # Arrays including interior, endpoints, and extrapolation outside
    ref_points = np.array([-0.5, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.5])
    scene_points = tm.reference_to_scene(ref_points)
    recovered_ref = tm.scene_to_reference(scene_points)
    np.testing.assert_allclose(recovered_ref, ref_points, atol=1e-12)


def test_sample_reference_motion_refuses_interpolation_beyond_max_gap() -> None:
    motion = sample_motion()
    reg = ReferenceRegistration(
        reference_id=motion.id,
        calibration_id="rig_01",
        transform=ReferenceTransform(),
        time_mapping=TimeMapping(offset_s=0.0),
    )
    # Frames in sample_motion are at 0.0, 0.5, 1.0, 1.5, 2.0 (interval is 0.5s)
    # With max_gap_s=0.4, gap of 0.5s between frames must refuse interpolation!
    t_eval = np.array([0.25])
    pts, valid = sample_reference_motion(motion, reg, t_eval, max_gap_s=0.4)
    assert not valid[0].any()

    # With default max_gap_s=0.5 (dt = 0.5 <= max_gap_s), interpolation is permitted
    pts_allowed, valid_allowed = sample_reference_motion(
        motion, reg, t_eval, max_gap_s=0.5
    )
    assert bool(valid_allowed[0, 0]) is True  # pelvis is observed


def test_reference_registration_geometry_fingerprints_and_stale_guards() -> None:
    motion = sample_motion()
    # Cannot be calibrated with uncalibrated calibration_id
    with pytest.raises(ValueError, match="is_calibrated cannot be True"):
        ReferenceRegistration(
            reference_id=motion.id,
            calibration_id="uncalibrated",
            is_calibrated=True,
        )

    with pytest.raises(ValueError, match="is_calibrated cannot be True"):
        ReferenceRegistration(
            reference_id=motion.id,
            calibration_id="uncalibrated_2d",
            is_calibrated=True,
        )

    # Valid binding with fingerprints
    reg = ReferenceRegistration(
        reference_id=motion.id,
        calibration_id="rig_calib_2026",
        asset_fingerprint="sha256:abc12345",
        camera_fingerprint="sha256:fed98765",
        is_calibrated=True,
    )
    assert reg.asset_fingerprint == "sha256:abc12345"
    assert reg.camera_fingerprint == "sha256:fed98765"
    assert reg.is_calibrated is True


def test_projection_across_two_cameras_with_distortion_and_clipping() -> None:
    cam_fo, cam_dtl = two_camera_rig()
    motion = sample_motion()
    reg = ReferenceRegistration(
        reference_id=motion.id,
        calibration_id="rig_01",
        transform=ReferenceTransform(),
        time_mapping=TimeMapping(offset_s=0.0),
    )

    # Point at target (0, 1, 0)
    pts_world = np.array([[[0.0, 1.0, 0.0]]])
    valid = np.array([[True]])

    # Face-on camera (at [0, 1, 3.5] looking at [0, 1, 0])
    px_fo, vis_fo = project_reference_to_camera(pts_world, valid, cam_fo)
    assert bool(vis_fo[0, 0]) is True
    np.testing.assert_allclose(px_fo[0, 0], [960.0, 540.0], atol=1e-3)

    # Down-the-line camera (at [-3.5, 1.0, 0.0] looking at [0, 1, 0])
    px_dtl, vis_dtl = project_reference_to_camera(pts_world, valid, cam_dtl)
    assert bool(vis_dtl[0, 0]) is True
    np.testing.assert_allclose(px_dtl[0, 0], [960.0, 540.0], atol=1e-3)

    # Add Brown-Conrady distortion to DTL camera
    dist_k = (-0.1, 0.02, 0.0, 0.0)
    calib_dtl = CameraCalibration(
        camera_id=cam_dtl.camera_id,
        intrinsics=CameraIntrinsics(matrix=cam_dtl.matrix, distortion=np.array(dist_k)),
        extrinsics=CameraExtrinsics(
            rotation_world_from_camera=cam_dtl.rotation_world_from_camera,
            translation_world_from_camera_m=cam_dtl.translation_world_from_camera_m,
        ),
        image_size_px=cam_dtl.image_size_px,
    )
    px_dtl_dist, vis_dtl_dist = project_reference_to_camera(pts_world, valid, calib_dtl)
    assert bool(vis_dtl_dist[0, 0]) is True
    np.testing.assert_allclose(px_dtl_dist[0, 0], [960.0, 540.0], atol=1e-2)
