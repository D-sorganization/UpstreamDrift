"""Unit tests for OpenCV segment mesh renderer (FTO-26, #11311).

Tests the engine-agnostic model-on-footage layer: filled, shaded segment volumes
drawn on video frames and coloured by tension/compression loads.
"""

from __future__ import annotations

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
    """Blank uniform grey image 1920x1080 BGR."""
    return np.full((1080, 1920, 3), 200, dtype=np.uint8)


def test_capsule_bounding_box(
    synthetic_camera: PinholeCamera, synthetic_frame: np.ndarray
) -> None:
    """A capsule in front of the camera fills a region with the expected bounding box (± 2 px)."""
    # Create an axis along world X from (-0.5, 0, 0) to (0.5, 0, 0) with radius 0.1m at depth 5m (z=0)
    axis = SegmentAxis(
        segment="forearm",
        joint_label="joint_reaction:elbow",
        proximal_m=(-0.5, 0.0, 0.0),
        distal_m=(0.5, 0.0, 0.0),
    )
    radius_m = 0.1
    poses = segment_poses_from_axes([axis], radius_m=radius_m)
    assert len(poses) == 1
    pose = poses[0]

    projector = PinholeProjector(synthetic_camera)
    shading = SegmentShading(opacity=1.0)
    scale = ForceColorScale(enabled=False)

    drawn, receipt = draw_segment_meshes_on_frame(
        synthetic_frame,
        [pose],
        projector,
        shading=shading,
        loads=None,
        color_scale=scale,
    )

    assert receipt.triangles_drawn > 0
    assert receipt.segments_rendered == 1

    # Find modified pixels
    diff = np.any(drawn != synthetic_frame, axis=-1)
    assert diff.any()
    y_idxs, x_idxs = np.where(diff)
    actual_ymin, actual_ymax = int(y_idxs.min()), int(y_idxs.max())
    actual_xmin, actual_xmax = int(x_idxs.min()), int(x_idxs.max())

    # Analytical projection of the capsule bounds:
    # Cylinder + hemisphere caps along X from (-0.5 - 0.1) = -0.6 to (0.5 + 0.1) = 0.6
    # In Y from -0.1 to +0.1. All at Z=0 (depth 5.0m from camera at (0,0,5))
    extreme_pts = np.array(
        [
            [-0.6, 0.0, 0.0],
            [0.6, 0.0, 0.0],
            [0.0, -0.1, 0.0],
            [0.0, 0.1, 0.0],
        ]
    )
    px_extremes, _ = synthetic_camera.project(extreme_pts)
    expected_xmin = int(round(px_extremes[0, 0]))
    expected_xmax = int(round(px_extremes[1, 0]))
    expected_ymin = int(round(px_extremes[3, 1]))  # Y is up in world, down in image
    expected_ymax = int(round(px_extremes[2, 1]))

    assert abs(actual_xmin - expected_xmin) <= 2
    assert abs(actual_xmax - expected_xmax) <= 2
    assert abs(actual_ymin - expected_ymin) <= 2
    assert abs(actual_ymax - expected_ymax) <= 2


def test_overlapping_capsules_painter_order(
    synthetic_camera: PinholeCamera, synthetic_frame: np.ndarray
) -> None:
    """Two overlapping capsules: the nearer one's colour wins in the overlap (painter's order)."""
    # Far capsule at z=0 (depth 5m), colored Red (#ff0000)
    axis_far = SegmentAxis(
        segment="far_seg",
        joint_label="joint_reaction:far",
        proximal_m=(-0.3, 0.0, 0.0),
        distal_m=(0.3, 0.0, 0.0),
    )
    # Near capsule at z=1 (depth 4m), colored Blue (#0000ff)
    axis_near = SegmentAxis(
        segment="near_seg",
        joint_label="joint_reaction:near",
        proximal_m=(0.0, -0.3, 1.0),
        distal_m=(0.0, 0.3, 1.0),
    )
    poses_far = segment_poses_from_axes([axis_far], radius_m=0.15)
    poses_near = segment_poses_from_axes([axis_near], radius_m=0.15)
    pose_far = SegmentPose(
        name=poses_far[0].name,
        mesh_id=poses_far[0].mesh_id,
        T_world_segment=poses_far[0].T_world_segment,
        scale=poses_far[0].scale,
        base_color="#ff0000",
    )
    pose_near = SegmentPose(
        name=poses_near[0].name,
        mesh_id=poses_near[0].mesh_id,
        T_world_segment=poses_near[0].T_world_segment,
        scale=poses_near[0].scale,
        base_color="#0000ff",
    )

    projector = PinholeProjector(synthetic_camera)
    shading = SegmentShading(opacity=1.0)
    scale = ForceColorScale(enabled=False)

    # Pass in both orders to verify depth sorting governs, not input list order
    drawn_1, _ = draw_segment_meshes_on_frame(
        synthetic_frame,
        [pose_far, pose_near],
        projector,
        shading=shading,
        loads=None,
        color_scale=scale,
    )
    drawn_2, _ = draw_segment_meshes_on_frame(
        synthetic_frame,
        [pose_near, pose_far],
        projector,
        shading=shading,
        loads=None,
        color_scale=scale,
    )

    # Center pixel (960, 540) is in the overlap of both capsules
    cx, cy = 960, 540
    # Blue is BGR (255, 0, 0), Red is BGR (0, 0, 255)
    color_1 = drawn_1[cy, cx]
    color_2 = drawn_2[cy, cx]
    # Nearer capsule is Blue: B channel must be much greater than R channel
    assert color_1[0] > color_1[2]
    assert color_2[0] > color_2[2]
    assert np.array_equal(drawn_1[cy, cx], drawn_2[cy, cx])


def test_loads_color_tension_and_compression(
    synthetic_camera: PinholeCamera, synthetic_frame: np.ndarray
) -> None:
    """With loads: a tension segment's hue is blue-ish and a compression segment's is red-ish."""
    axis_tension = SegmentAxis(
        segment="tension_seg",
        joint_label="joint_reaction:t",
        proximal_m=(-0.5, 0.2, 0.0),
        distal_m=(-0.1, 0.2, 0.0),
    )
    axis_comp = SegmentAxis(
        segment="comp_seg",
        joint_label="joint_reaction:c",
        proximal_m=(0.1, 0.2, 0.0),
        distal_m=(0.5, 0.2, 0.0),
    )
    poses = segment_poses_from_axes([axis_tension, axis_comp], radius_m=0.08)

    loads = AxialLoadFrame(
        time_s=0.0,
        values_n={"tension_seg": 500.0, "comp_seg": -500.0},
        source="test",
    )
    scale = ForceColorScale(
        enabled=True,
        tension_limit_n=500.0,
        compression_limit_n=500.0,
        tension_color="#0000ff",
        compression_color="#ff0000",
    )
    projector = PinholeProjector(synthetic_camera)
    shading = SegmentShading(opacity=1.0)

    drawn, receipt = draw_segment_meshes_on_frame(
        synthetic_frame,
        poses,
        projector,
        shading=shading,
        loads=loads,
        color_scale=scale,
    )
    assert receipt.segments_without_loads == 0

    # Project centers of both segments
    px_tension, _ = synthetic_camera.project(np.array([[-0.3, 0.2, 0.0]]))
    px_comp, _ = synthetic_camera.project(np.array([[0.3, 0.2, 0.0]]))

    t_color = drawn[int(round(px_tension[0, 1])), int(round(px_tension[0, 0]))]
    c_color = drawn[int(round(px_comp[0, 1])), int(round(px_comp[0, 0]))]

    # Tension is blue-ish: BGR -> B > R
    assert t_color[0] > 100
    assert t_color[2] < 50
    # Compression is red-ish: BGR -> R > B
    assert c_color[2] > 100
    assert c_color[0] < 50


def test_opacity_zero_leaves_frame_unchanged(
    synthetic_camera: PinholeCamera, synthetic_frame: np.ndarray
) -> None:
    """Opacity 0 leaves the frame unchanged."""
    axis = SegmentAxis(
        segment="arm",
        joint_label="joint_reaction:shoulder",
        proximal_m=(-0.2, 0.0, 0.0),
        distal_m=(0.2, 0.0, 0.0),
    )
    poses = segment_poses_from_axes([axis], radius_m=0.08)
    projector = PinholeProjector(synthetic_camera)
    shading = SegmentShading(opacity=0.0)
    scale = ForceColorScale(enabled=False)

    drawn, receipt = draw_segment_meshes_on_frame(
        synthetic_frame,
        poses,
        projector,
        shading=shading,
        loads=None,
        color_scale=scale,
    )
    assert np.array_equal(drawn, synthetic_frame)


def test_segment_behind_camera_culled(
    synthetic_camera: PinholeCamera, synthetic_frame: np.ndarray
) -> None:
    """A segment behind the camera is culled and counted."""
    # Camera is at (0, 0, 5) looking down -Z. Behind camera is Z > 5.
    axis_behind = SegmentAxis(
        segment="behind",
        joint_label="joint_reaction:back",
        proximal_m=(0.0, 0.0, 10.0),
        distal_m=(0.0, 0.5, 10.0),
    )
    poses = segment_poses_from_axes([axis_behind], radius_m=0.1)
    projector = PinholeProjector(synthetic_camera)
    shading = SegmentShading(opacity=1.0)
    scale = ForceColorScale(enabled=False)

    drawn, receipt = draw_segment_meshes_on_frame(
        synthetic_frame,
        poses,
        projector,
        shading=shading,
        loads=None,
        color_scale=scale,
    )
    assert receipt.triangles_drawn == 0
    assert receipt.triangles_culled > 0
    assert np.array_equal(drawn, synthetic_frame)


def test_segment_poses_from_axes_alignment() -> None:
    """segment_poses_from_axes aligns the capsule axis with proximal→distal."""
    proximal = (1.0, 2.0, 3.0)
    distal = (1.0, 6.0, 3.0)  # Along +Y with length 4.0
    axis = SegmentAxis(
        segment="thigh",
        joint_label="joint_reaction:hip",
        proximal_m=proximal,
        distal_m=distal,
    )
    poses = segment_poses_from_axes([axis], radius_m=0.08)
    assert len(poses) == 1
    pose = poses[0]

    assert pose.name == "thigh"
    assert pose.mesh_id == "capsule"
    assert np.isclose(pose.scale[0], 4.0)
    assert np.isclose(pose.scale[1], 0.08)

    T = pose.T_world_segment
    # Proximal point in local capsule is origin (0, 0, 0)
    p_hom = T @ np.array([0.0, 0.0, 0.0, 1.0])
    np.testing.assert_allclose(p_hom[:3], proximal, atol=1e-6)

    # Distal point in local capsule is (length, 0, 0)
    d_hom = T @ np.array([4.0, 0.0, 0.0, 1.0])
    np.testing.assert_allclose(d_hom[:3], distal, atol=1e-6)

    # Local X axis is aligned with proximal -> distal
    R = T[:3, :3]
    expected_axis = np.array([0.0, 1.0, 0.0])
    np.testing.assert_allclose(R[:, 0], expected_axis, atol=1e-6)


def test_validation_contracts() -> None:
    """DbC assertions: SegmentPose, SegmentShading, and segment_poses_from_axes validations."""
    with pytest.raises(ValueError):
        # Empty name
        SegmentPose(name="", mesh_id="capsule", T_world_segment=np.eye(4))
    with pytest.raises(ValueError):
        # Non-orthonormal matrix
        T_bad = np.eye(4)
        T_bad[0, 1] = 2.0
        SegmentPose(name="arm", mesh_id="capsule", T_world_segment=T_bad)
    with pytest.raises(ValueError):
        # Negative scale
        SegmentPose(
            name="arm",
            mesh_id="capsule",
            T_world_segment=np.eye(4),
            scale=(-1.0, 1.0, 1.0),
        )
    with pytest.raises(ValueError):
        # Invalid ambient
        SegmentShading(ambient=-0.1)
    with pytest.raises(ValueError):
        # Invalid opacity
        SegmentShading(opacity=1.5)
    with pytest.raises(ValueError):
        # Non-positive radius
        axis = SegmentAxis(
            segment="arm", joint_label="j", proximal_m=(0, 0, 0), distal_m=(1, 0, 0)
        )
        segment_poses_from_axes([axis], radius_m=0.0)


def test_triangle_budget_capping(
    synthetic_camera: PinholeCamera, synthetic_frame: np.ndarray
) -> None:
    """Triangle budget limits drawn triangles and counts culled."""
    axis = SegmentAxis(
        segment="arm", joint_label="j", proximal_m=(-0.5, 0, 0), distal_m=(0.5, 0, 0)
    )
    poses = segment_poses_from_axes([axis], radius_m=0.1)
    projector = PinholeProjector(synthetic_camera)
    # Very small budget of 20 triangles
    shading = SegmentShading(max_triangles=20)
    _, receipt = draw_segment_meshes_on_frame(
        synthetic_frame,
        poses,
        projector,
        shading=shading,
        loads=None,
        color_scale=ForceColorScale(enabled=False),
    )
    assert receipt.triangles_drawn <= 20
    assert receipt.render_time_ms >= 0.0


def test_synthetic_demo_still_generation(synthetic_camera: PinholeCamera) -> None:
    """Generate the acceptance demo still: gradient bg, 3-segment arm (tension/comp), FTO-8 arrows."""
    from pathlib import Path
    from src.shared.python.force_overlay.glyphs import ArrowGlyph, GlyphSet, LegendSpec
    from src.shared.python.force_overlay.renderers.opencv_glyphs import (
        draw_glyphs_on_frame,
    )

    # 1. Gradient background 1920x1080 BGR
    w, h = 1920, 1080
    y_vals = np.linspace(30, 90, h, dtype=np.uint8).reshape(h, 1)
    channel = np.repeat(y_vals, w, axis=1)
    gradient_frame = np.stack([channel, channel, channel], axis=-1)

    # 2. Three-segment arm: upper arm, forearm, hand
    p_shoulder = (-0.6, 0.4, 0.0)
    p_elbow = (-0.1, 0.2, 0.0)
    p_wrist = (0.3, -0.1, 0.0)
    p_hand = (0.5, -0.2, 0.0)

    axis_upper = SegmentAxis(
        segment="upper_arm",
        joint_label="joint_reaction:shoulder",
        proximal_m=p_shoulder,
        distal_m=p_elbow,
    )
    axis_forearm = SegmentAxis(
        segment="forearm",
        joint_label="joint_reaction:elbow",
        proximal_m=p_elbow,
        distal_m=p_wrist,
    )
    axis_hand = SegmentAxis(
        segment="hand",
        joint_label="joint_reaction:wrist",
        proximal_m=p_wrist,
        distal_m=p_hand,
    )

    poses_arm = segment_poses_from_axes(
        [axis_upper, axis_forearm, axis_hand], radius_m=0.07
    )

    # Upper arm in tension (+800 N), Forearm in compression (-700 N), Hand neutral (0 N)
    loads = AxialLoadFrame(
        time_s=0.0,
        values_n={"upper_arm": 800.0, "forearm": -700.0, "hand": 0.0},
        source="synthetic_demo",
    )
    color_scale = ForceColorScale(
        enabled=True,
        tension_limit_n=1000.0,
        compression_limit_n=1000.0,
        tension_color="#0055ff",
        compression_color="#ff2200",
    )
    projector = PinholeProjector(synthetic_camera)
    shading = SegmentShading(ambient=0.4, opacity=0.75)

    composite, receipt = draw_segment_meshes_on_frame(
        gradient_frame,
        poses_arm,
        projector,
        shading=shading,
        loads=loads,
        color_scale=color_scale,
    )
    assert receipt.triangles_drawn > 0
    assert receipt.segments_rendered == 3

    # 3. Add FTO-8 force arrows at shoulder and wrist
    from src.shared.python.force_overlay.contracts import (
        ForceTorqueFrame,
        OverlayWrench,
        WrenchKind,
    )
    from src.shared.python.force_overlay.glyphs import ForceGlyphStyle, build_glyphs

    ft_frame = ForceTorqueFrame(
        time_s=0.0,
        engine="synthetic",
        wrenches=(
            OverlayWrench(
                kind=WrenchKind.JOINT_REACTION,
                label="joint_reaction:shoulder",
                body="upper_arm",
                point_m=p_shoulder,
                force_n=(-120.0, 250.0, 0.0),
                source="synthetic_demo",
            ),
            OverlayWrench(
                kind=WrenchKind.EXTERNAL,
                label="external:wrist",
                body="hand",
                point_m=p_wrist,
                force_n=(100.0, -200.0, 0.0),
                source="synthetic_demo",
            ),
        ),
    )
    glyph_set = build_glyphs(ft_frame, ForceGlyphStyle())
    final_image, _ = draw_glyphs_on_frame(composite, glyph_set, projector)
    assert final_image.shape == (h, w, 3)

    # Save to docs/development/fto_26_demo_still.png (or tmp_path)
    out_path = Path("docs/development/fto_26_demo_still.png").resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img_bytes = np.ascontiguousarray(final_image, dtype=np.uint8)
    success = cv2.imwrite(str(out_path), img_bytes)
    assert success is True
    assert out_path.is_file()
    assert out_path.stat().st_size > 1000

    # Also save to repo docs/development/ if directory exists
    repo_docs = Path(__file__).resolve().parents[3] / "docs" / "development"
    if repo_docs.is_dir():
        cv2.imwrite(str(repo_docs / "fto_26_demo_still.png"), img_bytes)
