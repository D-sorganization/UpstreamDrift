"""Unit tests for OpenCV projected segment mesh renderer (FTO-26, #11311)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from src.motion_capture.reconstruct.cameras import (
    PinholeCamera,
    intrinsics_from_fov,
    look_at,
)
from src.shared.python.body_part_viz.axial_loads import AxialLoadFrame
from src.shared.python.body_part_viz.force_colors import ForceColorScale
from src.shared.python.force_overlay.conversions import SegmentAxis
from src.shared.python.force_overlay.renderers.opencv_glyphs import PinholeProjector
from src.shared.python.force_overlay.renderers.opencv_segments import (
    SegmentDrawReceipt,
    SegmentPose,
    SegmentShading,
    draw_segment_meshes_on_frame,
    segment_poses_from_axes,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def synthetic_camera() -> PinholeCamera:
    """Camera at (0, 0, 5) looking down -z at target (0, 0, 0) with up (0, 1, 0).

    In ADR-0041:
    pos = (0, 0, 5), tgt = (0, 0, 0).
    Image: 1920x1080, FOV 60 deg.
    Principal point: (960, 540).
    Depth to origin is 5.0m.
    """
    w, h = 1920, 1080
    k = intrinsics_from_fov(w, h, 60.0)
    pos = np.array([0.0, 0.0, 5.0])
    tgt = np.array([0.0, 0.0, 0.0])
    r = look_at(pos, tgt, up=np.array([0.0, 1.0, 0.0]))
    return PinholeCamera(
        camera_id="cam_test",
        matrix=k,
        rotation_world_from_camera=r,
        translation_world_from_camera_m=pos,
        image_size_px=(w, h),
    )


@pytest.fixture
def synthetic_frame() -> np.ndarray:
    """Blank grey image 1920x1080 BGR."""
    return np.full((1080, 1920, 3), 200, dtype=np.uint8)


def test_capsule_bounding_box_within_2px(
    synthetic_camera: PinholeCamera, synthetic_frame: np.ndarray
) -> None:
    """A capsule in front of the camera fills a region with expected bounding box (+/- 2 px)."""
    projector = PinholeProjector(synthetic_camera)
    proximal = (-0.5, 0.0, 0.0)
    distal = (0.5, 0.0, 0.0)
    radius_m = 0.1

    axes = [SegmentAxis("capsule_seg", "joint", proximal, distal)]
    poses = segment_poses_from_axes(axes, radius_m=radius_m)
    assert len(poses) == 1

    shading = SegmentShading(ambient=0.4, opacity=1.0)
    receipt = draw_segment_meshes_on_frame(
        synthetic_frame,
        poses,
        projector,
        shading=shading,
    )
    assert receipt.triangles_drawn > 0

    drawn = receipt.frame
    assert drawn is not None

    diff = np.any(drawn != synthetic_frame, axis=2)
    y_idx, x_idx = np.where(diff)
    assert len(x_idx) > 0

    actual_min_x, actual_max_x = int(x_idx.min()), int(x_idx.max())
    actual_min_y, actual_max_y = int(y_idx.min()), int(y_idx.max())

    # Analytical projection of capsule bounding box:
    # Capsule along x from proximal - radius to distal + radius:
    # x in [-0.6, 0.6], y in [-0.1, 0.1], z in [-0.1, 0.1] (depth 5.0m).
    # Project extreme 3D points:
    test_points = np.array(
        [
            [-0.6, 0.0, 0.0],
            [0.6, 0.0, 0.0],
            [0.0, -0.1, 0.0],
            [0.0, 0.1, 0.0],
        ],
        dtype=float,
    )
    px, valid = projector.project(test_points)
    assert np.all(valid)

    exp_min_x = min(px[0, 0], px[1, 0])
    exp_max_x = max(px[0, 0], px[1, 0])
    exp_min_y = min(px[2, 1], px[3, 1])
    exp_max_y = max(px[2, 1], px[3, 1])

    assert abs(actual_min_x - exp_min_x) <= 2
    assert abs(actual_max_x - exp_max_x) <= 2
    assert abs(actual_min_y - exp_min_y) <= 2
    assert abs(actual_max_y - exp_max_y) <= 2


def test_overlapping_capsules_nearer_wins(
    synthetic_camera: PinholeCamera, synthetic_frame: np.ndarray
) -> None:
    """Two overlapping capsules: the nearer one's color wins in the overlap (painter's order)."""
    projector = PinholeProjector(synthetic_camera)

    # Farther capsule at z = 0.0 (depth 5.0m from camera at z = 5.0), blue base color (#0000ff)
    far_axis = SegmentAxis("far", "j1", (-0.4, 0.0, 0.0), (0.4, 0.0, 0.0))
    far_poses = segment_poses_from_axes([far_axis], radius_m=0.15)
    far_pose = SegmentPose(
        name=far_poses[0].name,
        mesh_id=far_poses[0].mesh_id,
        T_world_segment=far_poses[0].T_world_segment,
        scale=far_poses[0].scale,
        base_color="#0000ff",
    )

    # Nearer capsule at z = 1.0 (depth 4.0m from camera at z = 5.0), red base color (#ff0000)
    near_axis = SegmentAxis("near", "j2", (0.0, -0.4, 1.0), (0.0, 0.4, 1.0))
    near_poses = segment_poses_from_axes([near_axis], radius_m=0.15)
    near_pose = SegmentPose(
        name=near_poses[0].name,
        mesh_id=near_poses[0].mesh_id,
        T_world_segment=near_poses[0].T_world_segment,
        scale=near_poses[0].scale,
        base_color="#ff0000",
    )

    # Render both together; painter's algorithm sorts far to near, so near is drawn on top
    shading = SegmentShading(ambient=0.5, opacity=1.0)
    receipt = draw_segment_meshes_on_frame(
        synthetic_frame,
        [far_pose, near_pose],
        projector,
        shading=shading,
    )
    drawn = receipt.frame
    assert drawn is not None

    # Center pixel (960, 540) is in the overlap of both capsules
    # In BGR: Red has high R (channel 2), Blue has high B (channel 0)
    bgr_at_center = drawn[540, 960]
    b, g, r = int(bgr_at_center[0]), int(bgr_at_center[1]), int(bgr_at_center[2])
    assert r > b, (
        f"Expected nearer red capsule to win over farther blue capsule, got BGR={bgr_at_center}"
    )


def test_loads_tension_blue_compression_red(
    synthetic_camera: PinholeCamera, synthetic_frame: np.ndarray
) -> None:
    """With loads: a tension segment's hue is blue-ish and compression is red-ish."""
    projector = PinholeProjector(synthetic_camera)

    axis_tension = SegmentAxis("seg_tension", "j1", (-0.5, 0.2, 0.0), (-0.1, 0.2, 0.0))
    axis_comp = SegmentAxis("seg_comp", "j2", (0.1, 0.2, 0.0), (0.5, 0.2, 0.0))
    poses = segment_poses_from_axes([axis_tension, axis_comp], radius_m=0.08)

    color_scale = ForceColorScale(
        enabled=True,
        tension_limit_n=1000.0,
        compression_limit_n=1000.0,
        deadband_n=0.0,
        tension_color="#0000ff",
        compression_color="#ff0000",
        neutral_color="#ffffff",
    )

    # Verify ForceColorScale full saturation before shading
    assert color_scale.color(1500.0, "#808080") == "#0000ff"
    assert color_scale.color(-1500.0, "#808080") == "#ff0000"

    loads = AxialLoadFrame(
        time_s=0.0,
        values_n={"seg_tension": 1500.0, "seg_comp": -1500.0},
        source="test",
    )

    # Use a black frame for clear hue detection
    black_frame = np.zeros_like(synthetic_frame)
    shading = SegmentShading(ambient=0.5, opacity=1.0)
    receipt = draw_segment_meshes_on_frame(
        black_frame,
        poses,
        projector,
        shading=shading,
        loads=loads,
        color_scale=color_scale,
    )
    drawn = receipt.frame
    assert drawn is not None

    # Sample center of tension segment (left side, u ~ 780, v ~ 450)
    p_tension, _ = projector.project(np.array([[-0.3, 0.2, 0.0]]))
    u_t, v_t = int(round(p_tension[0, 0])), int(round(p_tension[0, 1]))
    bgr_tension = drawn[v_t, u_t]
    assert bgr_tension[0] > bgr_tension[2], (
        f"Tension should be blue-ish, got BGR={bgr_tension}"
    )

    # Sample center of compression segment (right side, u ~ 1140, v ~ 450)
    p_comp, _ = projector.project(np.array([[0.3, 0.2, 0.0]]))
    u_c, v_c = int(round(p_comp[0, 0])), int(round(p_comp[0, 1]))
    bgr_comp = drawn[v_c, u_c]
    assert bgr_comp[2] > bgr_comp[0], (
        f"Compression should be red-ish, got BGR={bgr_comp}"
    )


def test_opacity_zero_leaves_frame_unchanged(
    synthetic_camera: PinholeCamera, synthetic_frame: np.ndarray
) -> None:
    """Opacity 0 leaves the frame unchanged (pixel equality)."""
    projector = PinholeProjector(synthetic_camera)
    axis = SegmentAxis("seg", "j", (-0.3, 0.0, 0.0), (0.3, 0.0, 0.0))
    poses = segment_poses_from_axes([axis], radius_m=0.1)

    receipt = draw_segment_meshes_on_frame(
        synthetic_frame,
        poses,
        projector,
        shading=SegmentShading(opacity=0.0),
    )
    np.testing.assert_array_equal(receipt.frame, synthetic_frame)


def test_segment_behind_camera_culled_and_counted(
    synthetic_camera: PinholeCamera, synthetic_frame: np.ndarray
) -> None:
    """A segment behind the camera is culled and counted in receipt."""
    projector = PinholeProjector(synthetic_camera)
    # Camera is at (0, 0, 5) looking down -z at (0, 0, 0).
    # z = 10 is behind the camera.
    axis = SegmentAxis("behind", "j", (-0.3, 0.0, 10.0), (0.3, 0.0, 10.0))
    poses = segment_poses_from_axes([axis], radius_m=0.1)

    receipt = draw_segment_meshes_on_frame(
        synthetic_frame,
        poses,
        projector,
    )
    assert receipt.triangles_culled > 0
    assert receipt.triangles_drawn == 0
    np.testing.assert_array_equal(receipt.frame, synthetic_frame)


def test_segment_poses_from_axes_aligns_endpoints() -> None:
    """segment_poses_from_axes aligns the capsule axis with proximal -> distal endpoints."""
    proximal = (1.0, 2.0, 3.0)
    distal = (1.0, 5.0, 3.0)
    axes = [SegmentAxis("tibia", "knee", proximal, distal)]
    poses = segment_poses_from_axes(axes, radius_m=0.05)
    assert len(poses) == 1

    pose = poses[0]
    assert pose.name == "tibia"
    assert pose.mesh_id == "capsule"
    assert np.isclose(pose.scale[0], 3.0)  # Length
    assert np.isclose(pose.scale[1], 0.05)  # Radius
    assert np.isclose(pose.scale[2], 0.05)

    T = pose.T_world_segment
    # Local origin maps to proximal
    p0 = T[:3, :3] @ np.array([0.0, 0.0, 0.0]) + T[:3, 3]
    np.testing.assert_allclose(p0, proximal, atol=1e-6)

    # Local length along x maps to distal
    p1 = T[:3, :3] @ np.array([pose.scale[0], 0.0, 0.0]) + T[:3, 3]
    np.testing.assert_allclose(p1, distal, atol=1e-6)


def test_segment_pose_validation() -> None:
    """Design by Contract checks on SegmentPose inputs."""
    # Empty name
    with pytest.raises(ValueError, match="name"):
        SegmentPose(name="", mesh_id="capsule")

    # Empty mesh_id
    with pytest.raises(ValueError, match="mesh_id"):
        SegmentPose(name="seg", mesh_id="")

    # Non-orthonormal matrix
    bad_rot = np.eye(4)
    bad_rot[0, 0] = 2.0
    with pytest.raises(ValueError, match="orthonormal"):
        SegmentPose(name="seg", T_world_segment=bad_rot)

    # Non-positive scale
    with pytest.raises(ValueError, match="scale"):
        SegmentPose(name="seg", scale=(1.0, 0.0, 1.0))

    # Negative radius in segment_poses_from_axes
    axis = SegmentAxis("s", "j", (0, 0, 0), (1, 0, 0))
    with pytest.raises(ValueError, match="radius_m"):
        segment_poses_from_axes([axis], radius_m=-0.1)


def test_synthetic_demo_still_three_segment_arm(
    synthetic_camera: PinholeCamera, tmp_path: Path
) -> None:
    """Create synthetic demo still: gradient background, 3-segment arm, mixed loads, FTO-8 arrows."""
    # 1. Gradient background 1920x1080
    y = np.linspace(30, 80, 1080, dtype=np.uint8)[:, None, None]
    x = np.linspace(40, 90, 1920, dtype=np.uint8)[None, :, None]
    gradient_frame = np.broadcast_to(y + x, (1080, 1920, 3)).copy()

    # 2. Three-segment arm
    shoulder = (-0.4, 0.2, 0.0)
    elbow = (0.0, 0.35, 0.0)
    wrist = (0.35, 0.1, 0.0)
    hand = (0.55, -0.1, 0.0)

    axes = [
        SegmentAxis("upper_arm", "shoulder_j", shoulder, elbow),
        SegmentAxis("forearm", "elbow_j", elbow, wrist),
        SegmentAxis("hand", "wrist_j", wrist, hand),
    ]
    poses = segment_poses_from_axes(axes, radius_m=0.06)
    assert len(poses) == 3

    # Mixed tension and compression loads
    loads = AxialLoadFrame(
        time_s=0.0,
        values_n={"upper_arm": 1200.0, "forearm": -1200.0, "hand": 0.0},
        source="simscape",
    )
    color_scale = ForceColorScale(
        enabled=True,
        tension_limit_n=1000.0,
        compression_limit_n=1000.0,
        tension_color="#0000ff",
        compression_color="#ff0000",
        neutral_color="#ffffff",
    )

    projector = PinholeProjector(synthetic_camera, world_frame="adr0041_world")
    receipt = draw_segment_meshes_on_frame(
        gradient_frame,
        poses,
        projector,
        shading=SegmentShading(ambient=0.35, opacity=0.75),
        loads=loads,
        color_scale=color_scale,
    )
    assert receipt.triangles_drawn > 0
    assert receipt.render_time_ms > 0
    drawn = receipt.frame
    assert drawn is not None

    # 3. Add FTO-8 arrows
    from src.shared.python.force_overlay.contracts import (
        ForceTorqueFrame,
        OverlayWrench,
        WrenchKind,
    )
    from src.shared.python.force_overlay.glyphs import ForceGlyphStyle, build_glyphs
    from src.shared.python.force_overlay.renderers.opencv_glyphs import (
        draw_glyphs_on_frame,
    )

    wrench = OverlayWrench(
        kind=WrenchKind.JOINT_REACTION,
        label="reaction:elbow",
        body="forearm",
        point_m=elbow,
        force_n=(0.0, 500.0, 0.0),
        source="test",
    )
    ft_frame = ForceTorqueFrame(
        time_s=0.0,
        engine="simscape",
        world_frame="adr0041_world",
        wrenches=(wrench,),
    )
    glyphs = build_glyphs(ft_frame, ForceGlyphStyle())
    glyph_receipt = draw_glyphs_on_frame(
        drawn, glyphs, projector, world_frame="adr0041_world", inplace=True
    )
    assert glyph_receipt.drawn == 1

    out_file = tmp_path / "fto26_demo_still.png"
    cv2.imwrite(str(out_file), drawn)
    assert out_file.exists()
