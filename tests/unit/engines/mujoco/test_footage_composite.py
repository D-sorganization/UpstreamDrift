"""Unit tests for calibrated MuJoCo mesh render composited onto source footage (FTO-27, #11312).

Verifies:
1. mujoco_camera_from_pinhole reprojection accuracy (<= 1 px) with off-centre principal point.
2. OpenCV camera forward mapping to MuJoCo -z (axis flip).
3. Alpha segmentation: background pixels untouched, model pixels composited.
4. Lens distortion policy: undistorting footage frame ensures arrows and mesh agree within 1.5 px.
5. Contract validation for camera intrinsics, registration matrix, and frame dimensions.
6. Integration with glyphs (FTO-6) and axial loads (force color shading).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
import pytest

if TYPE_CHECKING:
    import mujoco

mujoco = pytest.importorskip("mujoco")

from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.footage_composite import (
    CompositeOptions,
    FootageCompositeReceipt,
    MjCameraSpec,
    apply_camera_spec_to_scene,
    composite_model_on_frame,
    mujoco_camera_from_pinhole,
    registration_to_world_from_mj,
)
from src.motion_capture.reconstruct.cameras import PinholeCamera, look_at
from src.motion_capture.reference.registration import (
    ReferenceRegistration,
    ReferenceTransform,
)
from src.shared.python.body_part_viz import AxialLoadFrame
from src.shared.python.body_part_viz.force_colors import ForceColorScale
from src.shared.python.force_overlay.contracts import WrenchKind
from src.shared.python.force_overlay.glyphs import (
    ArrowGlyph,
    GlyphSet,
    LegendSpec,
)


def _make_sphere_model(points_mj: list[tuple[float, float, float]]) -> mujoco.MjModel:
    """Build an MJCF model with small target spheres."""
    xml = """
    <mujoco>
      <visual>
        <global offwidth="1000" offheight="1000"/>
      </visual>
      <worldbody>
        <light pos="0 0 5"/>
    """
    for i, (x, y, z) in enumerate(points_mj):
        xml += f'    <geom name="target_{i}" type="sphere" size="0.015" pos="{x} {y} {z}" rgba="0.9 0.1 0.1 1"/>\n'
    xml += """
      </worldbody>
    </mujoco>
    """
    return mujoco.MjModel.from_xml_string(xml)


@pytest.mark.unit
@pytest.mark.requires_gl
def test_mujoco_camera_from_pinhole_synthetic_reprojection() -> None:
    """A synthetic camera with off-centre principal point projects to the same pixels (<= 1 px)."""
    # 1. Pinhole camera with deliberately off-centre principal point
    w_px, h_px = 640, 480
    fx, fy = 750.0, 750.0
    cx, cy = 350.0, 210.0  # off-centre: w/2=320, h/2=240
    k_mat = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=float)

    cam_pos_world = np.array([0.2, 1.5, -2.5], dtype=float)
    lookat_world = np.array([0.0, 0.9, 0.0], dtype=float)
    r_w2c = look_at(cam_pos_world, lookat_world, up=np.array([0.0, 1.0, 0.0]))

    camera = PinholeCamera(
        camera_id="test_cam",
        matrix=k_mat,
        rotation_world_from_camera=r_w2c,
        translation_world_from_camera_m=cam_pos_world,
        image_size_px=(w_px, h_px),
    )

    # 2. Canonical to ADR-0041 world transform
    c_eff = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]], dtype=float)
    world_from_mj = np.eye(4, dtype=float)
    world_from_mj[:3, :3] = c_eff
    world_from_mj[:3, 3] = np.array([0.1, 0.05, -0.2], dtype=float)

    # 3. Model targets in MuJoCo coordinates
    targets_mj = [
        (0.0, 0.0, 0.9),
        (0.2, -0.1, 1.1),
        (-0.15, 0.2, 0.7),
        (0.1, 0.15, 0.85),
    ]
    model = _make_sphere_model(targets_mj)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    # 4. Map camera to MuJoCo
    spec = mujoco_camera_from_pinhole(camera, world_from_mj)
    assert isinstance(spec, MjCameraSpec)
    assert spec.render_w >= w_px
    assert spec.render_h >= h_px
    assert len(spec.crop) == 4

    # 5. Render with segmentation
    renderer = mujoco.Renderer(model, width=spec.render_w, height=spec.render_h)
    renderer.update_scene(data)
    apply_camera_spec_to_scene(renderer.scene, spec)

    renderer.enable_segmentation_rendering()
    seg = renderer.render()
    renderer.disable_segmentation_rendering()

    # Crop to original image rectangle
    x0, y0, x1, y1 = spec.crop
    cropped_seg = seg[y0:y1, x0:x1]
    assert cropped_seg.shape[:2] == (h_px, w_px)

    # 6. Compare centroids for all targets
    for i, pt_mj in enumerate(targets_mj):
        pt_world = (world_from_mj @ np.append(pt_mj, 1.0))[:3]
        px_pin, in_front = camera.project(pt_world)
        assert in_front[0]
        u_pin, v_pin = px_pin[0]

        geom_mask = cropped_seg[:, :, 0] == i
        assert np.any(geom_mask), f"Geom {i} not rendered in frame"
        coords = np.argwhere(geom_mask)
        v_render = float(coords[:, 0].mean())
        u_render = float(coords[:, 1].mean())

        err = float(np.hypot(u_render - u_pin, v_render - v_pin))
        assert err <= 1.0, (
            f"Target {i} reprojection error {err:.3f} px exceeds 1.0 px budget"
        )


@pytest.mark.unit
def test_axis_flip_maps_camera_forward_to_mujoco_minus_z() -> None:
    """The axis flip maps camera forward to MuJoCo -z."""
    w_px, h_px = 640, 480
    k_mat = np.array(
        [[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]], dtype=float
    )
    # Identity rotation: OpenCV camera looking along world +Z
    camera = PinholeCamera(
        camera_id="axis_cam",
        matrix=k_mat,
        rotation_world_from_camera=np.eye(3, dtype=float),
        translation_world_from_camera_m=np.zeros(3, dtype=float),
        image_size_px=(w_px, h_px),
    )
    world_from_mj = np.eye(4, dtype=float)

    spec = mujoco_camera_from_pinhole(camera, world_from_mj)

    # In OpenCV, forward is +Z [0, 0, 1].
    # In MuJoCo, forward viewing direction is -(x x y) = y x x.
    x_axis = spec.xyaxes[:3]
    y_axis = spec.xyaxes[3:]
    forward_mj = np.cross(y_axis, x_axis)

    # Camera forward in MuJoCo coordinates must be along +Z (since R_world_from_camera is eye and world_from_mj is eye)
    # And in the camera's local GL frame, looking is along -Z.
    np.testing.assert_allclose(forward_mj, [0.0, 0.0, 1.0], atol=1e-6)
    # The up vector in OpenCV is +Y down, which flips to -Y in world (so y_axis is [0, -1, 0])
    np.testing.assert_allclose(x_axis, [1.0, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(y_axis, [0.0, -1.0, 0.0], atol=1e-6)


@pytest.mark.unit
@pytest.mark.requires_gl
def test_alpha_composite_background_untouched() -> None:
    """Alpha: background pixels are untouched in the composite; model pixels change."""
    w_px, h_px = 320, 240
    frame_bgr = np.full((h_px, w_px, 3), 128, dtype=np.uint8)
    # Add a distinctive background test pattern
    frame_bgr[0:10, 0:10] = [200, 50, 50]
    frame_bgr[-10:, -10:] = [50, 200, 50]

    k_mat = np.array(
        [[400.0, 0.0, 160.0], [0.0, 400.0, 120.0], [0.0, 0.0, 1.0]], dtype=float
    )
    cam_pos = np.array([0.0, 0.0, -1.5], dtype=float)
    target_pos = np.array([0.0, 0.0, 0.0], dtype=float)
    camera = PinholeCamera(
        camera_id="cam",
        matrix=k_mat,
        rotation_world_from_camera=look_at(
            cam_pos, target_pos, up=np.array([0.0, 1.0, 0.0])
        ),
        translation_world_from_camera_m=cam_pos,
        image_size_px=(w_px, h_px),
    )

    model = _make_sphere_model([(0.0, 0.0, 0.0)])
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    world_from_mj = np.eye(4, dtype=float)
    opts = CompositeOptions(opacity=0.85, feather_px=1.0)

    out, receipt = composite_model_on_frame(
        frame_bgr=frame_bgr,
        model=model,
        data=data,
        camera=camera,
        registration=world_from_mj,
        glyphs=None,
        loads=None,
        opts=opts,
    )

    assert isinstance(receipt, FootageCompositeReceipt)
    assert receipt.model_pixels > 0

    # Corners far from the sphere must remain completely untouched
    assert np.array_equal(out[0, 0], frame_bgr[0, 0])
    assert np.array_equal(out[-1, -1], frame_bgr[-1, -1])
    assert np.array_equal(out[0, -1], frame_bgr[0, -1])
    assert np.array_equal(out[-1, 0], frame_bgr[-1, 0])

    # Center pixels must be modified by the model render
    center_px = (h_px // 2, w_px // 2)
    assert not np.array_equal(out[center_px], frame_bgr[center_px])


@pytest.mark.unit
@pytest.mark.requires_gl
def test_distortion_policy_mesh_and_arrows_agree() -> None:
    """Distortion policy: with non-zero k1, arrows and mesh agree at marker point within 1.5 px."""
    w_px, h_px = 640, 480
    fx, fy = 650.0, 650.0
    cx, cy = 320.0, 240.0
    k_mat = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=float)
    distortion = np.array([-0.18, 0.06, 0.0, 0.0, 0.0], dtype=float)

    cam_pos = np.array([0.0, 1.2, -2.2], dtype=float)
    lookat_pt = np.array([0.0, 0.8, 0.0], dtype=float)
    camera = PinholeCamera(
        camera_id="dist_cam",
        matrix=k_mat,
        rotation_world_from_camera=look_at(
            cam_pos, lookat_pt, up=np.array([0.0, 1.0, 0.0])
        ),
        translation_world_from_camera_m=cam_pos,
        image_size_px=(w_px, h_px),
        distortion=distortion,
    )

    marker_mj = (0.25, 0.1, 0.9)
    model = _make_sphere_model([marker_mj])
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    world_from_mj = np.eye(4, dtype=float)
    pt_world = (world_from_mj @ np.append(marker_mj, 1.0))[:3]

    # Arrow glyph rooted at marker
    arrow = ArrowGlyph(
        label="test_arrow",
        kind=WrenchKind.CONTACT,
        tail_m=marker_mj,
        tip_m=(marker_mj[0] + 0.1, marker_mj[1] + 0.1, marker_mj[2] + 0.1),
        head_base_m=(marker_mj[0] + 0.08, marker_mj[1] + 0.08, marker_mj[2] + 0.08),
        shaft_radius_m=0.01,
        head_radius_m=0.02,
        rgba=(0.0, 1.0, 0.0, 1.0),
        magnitude=50.0,
        units="N",
        clamped=False,
    )
    glyphs = GlyphSet(
        time_s=0.0,
        arrows=(arrow,),
        torque_arcs=(),
        legend=LegendSpec(),
    )

    frame_bgr = np.zeros((h_px, w_px, 3), dtype=np.uint8)
    out, receipt = composite_model_on_frame(
        frame_bgr=frame_bgr,
        model=model,
        data=data,
        camera=camera,
        registration=world_from_mj,
        glyphs=glyphs,
        loads=None,
        opts=CompositeOptions(undistort_footage=True),
    )

    assert receipt.frame_undistorted is True

    # Mesh location in rectilinear projection
    rectilinear_cam = PinholeCamera(
        camera_id="rect_cam",
        matrix=k_mat,
        rotation_world_from_camera=camera.rotation_world_from_camera,
        translation_world_from_camera_m=camera.translation_world_from_camera_m,
        image_size_px=(w_px, h_px),
        distortion=None,
    )
    px_arrow_start, _ = rectilinear_cam.project(pt_world)
    u_arrow, v_arrow = px_arrow_start[0]

    # Find the sphere in the rendered output (red channel high, green low)
    red_mask = (out[:, :, 2] > 100) & (out[:, :, 1] < 50)
    assert np.any(red_mask), "Sphere mesh not visible in composite output"
    red_coords = np.argwhere(red_mask)
    v_mesh = float(red_coords[:, 0].mean())
    u_mesh = float(red_coords[:, 1].mean())

    err = float(np.hypot(u_mesh - u_arrow, v_mesh - v_arrow))
    assert err <= 1.5, f"Mesh and arrow root disagree by {err:.3f} px (budget: 1.5 px)"


@pytest.mark.unit
def test_contract_validation_errors() -> None:
    """DbC preconditions reject invalid cameras, shapes, and inputs."""
    w_px, h_px = 640, 480
    k_mat = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
    camera = PinholeCamera(
        camera_id="cam",
        matrix=k_mat,
        rotation_world_from_camera=np.eye(3),
        translation_world_from_camera_m=np.zeros(3),
        image_size_px=(w_px, h_px),
    )

    # Invalid world_from_mj shape
    with pytest.raises(ValueError, match="world_from_mj"):
        mujoco_camera_from_pinhole(camera, np.eye(3))

    # Invalid world_from_mj determinant (singular or zero scale)
    bad_mat = np.zeros((4, 4))
    with pytest.raises(ValueError, match="singular"):
        mujoco_camera_from_pinhole(camera, bad_mat)

    # Invalid frame dimensions in composite_model_on_frame
    model = _make_sphere_model([(0.0, 0.0, 0.0)])
    data = mujoco.MjData(model)
    with pytest.raises(ValueError, match="frame_bgr dimensions"):
        composite_model_on_frame(
            frame_bgr=np.zeros((100, 100, 3), dtype=np.uint8),
            model=model,
            data=data,
            camera=camera,
            registration=np.eye(4),
        )


@pytest.mark.unit
@pytest.mark.requires_gl
def test_composite_with_registration_loads_and_glyphs() -> None:
    """Test compositing with ReferenceRegistration, AxialLoadFrame, and GlyphSet."""
    from uuid import uuid4

    w_px, h_px = 320, 240
    k_mat = np.array([[300.0, 0.0, 160.0], [0.0, 300.0, 120.0], [0.0, 0.0, 1.0]])
    cam_pos = np.array([0.0, 1.0, -2.0], dtype=float)
    target_pos = np.array([0.0, 0.5, 0.0], dtype=float)
    camera = PinholeCamera(
        camera_id="cam_reg",
        matrix=k_mat,
        rotation_world_from_camera=look_at(
            cam_pos, target_pos, up=np.array([0.0, 1.0, 0.0])
        ),
        translation_world_from_camera_m=cam_pos,
        image_size_px=(w_px, h_px),
    )

    xml = """
    <mujoco>
      <visual>
        <global offwidth="500" offheight="500"/>
      </visual>
      <worldbody>
        <light pos="0 0 5"/>
        <body name="body_test" pos="0 0 0.5">
          <geom name="geom_test" type="sphere" size="0.05" rgba="0.5 0.5 0.5 1"/>
        </body>
      </worldbody>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    reg = ReferenceRegistration(
        reference_id=str(uuid4()),
        calibration_id="test_calib",
        transform=ReferenceTransform(scale=1.0),
        mirror_lateral=False,
    )

    arrow = ArrowGlyph(
        label="contact_arrow",
        kind=WrenchKind.CONTACT,
        tail_m=(0.0, 0.0, 0.5),
        tip_m=(0.0, 0.0, 0.7),
        head_base_m=(0.0, 0.0, 0.65),
        shaft_radius_m=0.01,
        head_radius_m=0.02,
        rgba=(0.0, 1.0, 0.0, 1.0),
        magnitude=100.0,
        units="N",
        clamped=False,
    )
    glyphs = GlyphSet(
        time_s=0.0,
        arrows=(arrow,),
        torque_arcs=(),
        legend=LegendSpec(),
    )
    loads = AxialLoadFrame(
        time_s=0.0,
        values_n={"body_test": 500.0},
        source="test_sim",
    )
    scale = ForceColorScale(enabled=True)
    opts = CompositeOptions(color_scale=scale)

    frame_bgr = np.full((h_px, w_px, 3), 64, dtype=np.uint8)
    out, receipt = composite_model_on_frame(
        frame_bgr=frame_bgr,
        model=model,
        data=data,
        camera=camera,
        registration=reg,
        glyphs=glyphs,
        loads=loads,
        opts=opts,
    )

    assert receipt.glyphs_drawn == 1
    assert receipt.model_pixels > 0
    assert not np.array_equal(out, frame_bgr)
