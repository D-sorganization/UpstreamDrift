"""Calibrated MuJoCo mesh render composited onto source footage (FTO-27, #11312).

Provides:
- `mujoco_camera_from_pinhole`: maps calibrated PinholeCamera to MjCameraSpec.
- `composite_model_on_frame`: renders MuJoCo mesh and force glyphs, composited onto footage.
- `apply_camera_spec_to_scene`: applies MjCameraSpec to MjvScene camera buffers.
- `registration_to_world_from_mj`: converts ReferenceRegistration to 4x4 world_from_mj matrix.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

from src.motion_capture.reconstruct.cameras import PinholeCamera
from src.shared.python.body_part_viz import AxialLoadFrame
from src.shared.python.body_part_viz.force_colors import ForceColorScale
from src.shared.python.body_part_viz.mujoco_force_colors import (
    apply_mujoco_scene_colors,
)
from src.shared.python.force_overlay.glyphs import GlyphSet
from src.shared.python.logging_pkg.logging_config import get_logger

from .force_glyphs import add_glyphs_to_scene

if TYPE_CHECKING:
    import mujoco

    from src.motion_capture.reference.registration import ReferenceRegistration
    from src.shared.python.pose_estimation.observations import CameraCalibration

__all__ = [
    "CompositeOptions",
    "FootageCompositeReceipt",
    "MjCameraSpec",
    "apply_camera_spec_to_scene",
    "composite_model_on_frame",
    "mujoco_camera_from_pinhole",
    "registration_to_world_from_mj",
]

logger = get_logger(__name__)

_CANONICAL_TO_ADR0041 = np.array(
    [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]], dtype=float
)
_CANONICAL_TO_ADR0041_MIRRORED = np.array(
    [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]], dtype=float
)


@dataclass(frozen=True)
class MjCameraSpec:
    """MuJoCo camera rendering specification derived from a calibrated pinhole camera."""

    pos: tuple[float, float, float]
    xyaxes: tuple[float, float, float, float, float, float]
    fovy_deg: float
    render_w: int
    render_h: int
    crop: tuple[int, int, int, int]


@dataclass(frozen=True)
class CompositeOptions:
    """Configuration options for model-on-footage compositing."""

    opacity: float = 1.0
    feather_px: float = 1.0
    color_scale: ForceColorScale | None = None
    undistort_footage: bool = True
    athlete_mask: np.ndarray | None = None


@dataclass(frozen=True)
class FootageCompositeReceipt:
    """Execution receipt documenting model-on-footage compositing."""

    camera_spec: MjCameraSpec
    glyphs_drawn: int
    glyphs_dropped: int
    model_pixels: int
    frame_undistorted: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "pos": list(self.camera_spec.pos),
            "xyaxes": list(self.camera_spec.xyaxes),
            "fovy_deg": self.camera_spec.fovy_deg,
            "render_w": self.camera_spec.render_w,
            "render_h": self.camera_spec.render_h,
            "crop": list(self.camera_spec.crop),
            "glyphs_drawn": self.glyphs_drawn,
            "glyphs_dropped": self.glyphs_dropped,
            "model_pixels": self.model_pixels,
            "frame_undistorted": self.frame_undistorted,
        }

    def __iter__(self) -> Any:
        yield self


def registration_to_world_from_mj(
    registration: ReferenceRegistration | np.ndarray | None,
) -> np.ndarray:
    """Convert ReferenceRegistration or array to 4x4 world_from_mj transformation matrix."""
    if registration is None:
        mat = np.eye(4, dtype=float)
        mat[:3, :3] = _CANONICAL_TO_ADR0041
        return mat

    if isinstance(registration, np.ndarray):
        arr = np.asarray(registration, dtype=float)
        if arr.shape != (4, 4):
            raise ValueError(
                f"world_from_mj matrix must have shape (4, 4), got {arr.shape}"
            )
        if not np.all(np.isfinite(arr)):
            raise ValueError("world_from_mj matrix contains non-finite entries")
        return arr

    # ReferenceRegistration object
    c_eff = (
        _CANONICAL_TO_ADR0041_MIRRORED
        if registration.mirror_lateral
        else _CANONICAL_TO_ADR0041
    )
    rot = np.asarray(registration.transform.rotation, dtype=float)
    scale = float(registration.transform.scale)
    trans = np.asarray(registration.transform.translation_m, dtype=float)

    mat = np.eye(4, dtype=float)
    mat[:3, :3] = scale * (rot @ c_eff)
    mat[:3, 3] = trans
    return mat


def _resolve_camera(camera: PinholeCamera | CameraCalibration) -> PinholeCamera:
    """Extract PinholeCamera from either PinholeCamera or CameraCalibration."""
    if isinstance(camera, PinholeCamera):
        return camera
    return PinholeCamera.from_calibration(camera)


def _compute_render_dimensions_and_crop(
    w_px: int, h_px: int, cx: float, cy: float
) -> tuple[int, int, tuple[int, int, int, int]]:
    """Compute enlarged frame dimensions and crop rectangle to center (cx, cy)."""
    render_w = int(np.ceil(2.0 * max(cx, float(w_px) - cx)))
    render_h = int(np.ceil(2.0 * max(cy, float(h_px) - cy)))
    x0 = int(round(render_w / 2.0 - cx))
    y0 = int(round(render_h / 2.0 - cy))
    crop = (x0, y0, x0 + w_px, y0 + h_px)
    return render_w, render_h, crop


def mujoco_camera_from_pinhole(
    camera: PinholeCamera | CameraCalibration,
    world_from_mj: np.ndarray,
) -> MjCameraSpec:
    """Map a calibrated PinholeCamera into an MjCameraSpec for MuJoCo scene rendering.

    Computes:
    - Camera position and xyaxes in MuJoCo model coordinates.
    - OpenGL/MuJoCo axis flip (OpenCV +y down, +z forward -> MuJoCo +y up, -z forward).
    - Enlarged render dimensions and crop rectangle centering the principal point (cx, cy).
    - Vertical field of view (fovy) matching the enlarged render height and focal length.
    """
    cam = _resolve_camera(camera)
    w_mat = np.asarray(world_from_mj, dtype=float)
    if w_mat.shape != (4, 4):
        raise ValueError(f"world_from_mj must be a 4x4 matrix, got {w_mat.shape}")
    if not np.all(np.isfinite(w_mat)):
        raise ValueError("world_from_mj must contain only finite numbers")

    det_rot = float(np.linalg.det(w_mat[:3, :3]))
    if abs(det_rot) < 1e-8:
        raise ValueError("world_from_mj linear component is singular")

    mj_from_world = np.linalg.inv(w_mat)

    t_world_from_cam = np.eye(4, dtype=float)
    t_world_from_cam[:3, :3] = cam.rotation_world_from_camera
    t_world_from_cam[:3, 3] = cam.translation_world_from_camera_m

    t_mj_from_cam_cv = mj_from_world @ t_world_from_cam
    pos_mj = t_mj_from_cam_cv[:3, 3]
    r_mj_from_cam_cv = t_mj_from_cam_cv[:3, :3]

    # Basis change: OpenCV (x right, y down, z forward) to MuJoCo (x right, y up, -z forward)
    axis_flip = np.diag([1.0, -1.0, -1.0])
    r_mj_from_cam_gl = r_mj_from_cam_cv @ axis_flip

    x_axis = r_mj_from_cam_gl[:, 0]
    y_axis = r_mj_from_cam_gl[:, 1]
    norm_x = np.linalg.norm(x_axis)
    norm_y = np.linalg.norm(y_axis)
    if norm_x > 1e-8:
        x_axis = x_axis / norm_x
    if norm_y > 1e-8:
        y_axis = y_axis / norm_y

    w_px, h_px = cam.image_size_px
    cx = float(cam.matrix[0, 2])
    cy = float(cam.matrix[1, 2])
    fy = float(cam.matrix[1, 1])
    if fy <= 0:
        raise ValueError(f"Focal length fy must be positive, got {fy}")

    render_w, render_h, crop = _compute_render_dimensions_and_crop(w_px, h_px, cx, cy)
    fovy_rad = 2.0 * np.arctan((render_h / 2.0) / fy)
    fovy_deg = float(np.rad2deg(fovy_rad))

    xyaxes = (
        float(x_axis[0]),
        float(x_axis[1]),
        float(x_axis[2]),
        float(y_axis[0]),
        float(y_axis[1]),
        float(y_axis[2]),
    )
    pos_tuple = (float(pos_mj[0]), float(pos_mj[1]), float(pos_mj[2]))

    return MjCameraSpec(
        pos=pos_tuple,
        xyaxes=xyaxes,
        fovy_deg=fovy_deg,
        render_w=render_w,
        render_h=render_h,
        crop=crop,
    )


def apply_camera_spec_to_scene(scene: mujoco.MjvScene, spec: MjCameraSpec) -> None:
    """Apply an MjCameraSpec to the MuJoCo MjvScene camera buffers."""
    x_axis = np.asarray(spec.xyaxes[:3], dtype=float)
    y_axis = np.asarray(spec.xyaxes[3:], dtype=float)
    forward = np.cross(y_axis, x_axis)
    pos = np.asarray(spec.pos, dtype=float)

    for i in (0, 1):
        cam = scene.camera[i]
        cam.pos[:] = pos
        cam.forward[:] = forward
        cam.up[:] = y_axis
        top = cam.frustum_near * np.tan(np.deg2rad(spec.fovy_deg) / 2.0)
        cam.frustum_top = top
        cam.frustum_bottom = -top
        cam.frustum_center = 0.0
        cam.frustum_width = 0.0

    if scene.nlight > 0:
        scene.lights[0].pos[:] = pos
        scene.lights[0].dir[:] = forward


def _render_scene_and_segmentation(
    renderer: mujoco.Renderer,
    crop: tuple[int, int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Render segmentation and RGB buffers and crop to image bounds."""
    renderer.enable_segmentation_rendering()
    seg = renderer.render()
    renderer.disable_segmentation_rendering()
    rgb = renderer.render()

    x0, y0, x1, y1 = crop
    cropped_seg = seg[y0:y1, x0:x1]
    cropped_rgb = rgb[y0:y1, x0:x1]
    return cropped_seg, cropped_rgb


def _compute_feathered_alpha(
    cropped_seg: np.ndarray,
    opacity: float,
    feather_px: float,
    athlete_mask: np.ndarray | None,
) -> np.ndarray:
    """Generate feathered alpha channel from segmentation buffer."""
    raw_mask = (cropped_seg[:, :, 0] != -1).astype(np.float32)

    if athlete_mask is not None:
        raw_mask = raw_mask * (athlete_mask > 0).astype(np.float32)

    if feather_px > 0.0:
        ksize = int(2 * round(feather_px) + 1)
        feathered = cv2.GaussianBlur(raw_mask, (ksize, ksize), 0)
    else:
        feathered = raw_mask

    alpha = np.clip(feathered, 0.0, 1.0) * float(np.clip(opacity, 0.0, 1.0))
    return alpha


def _apply_alpha_composite(
    frame_bgr: np.ndarray,
    render_rgb: np.ndarray,
    alpha: np.ndarray,
) -> np.ndarray:
    """Blend cropped RGB render over destination BGR frame with alpha."""
    render_bgr = cv2.cvtColor(render_rgb, cv2.COLOR_RGB2BGR)
    active = alpha > 1e-4

    out = frame_bgr.copy()
    if np.any(active):
        alpha_act = alpha[active, None]
        blended = (
            frame_bgr[active].astype(float) * (1.0 - alpha_act)
            + render_bgr[active].astype(float) * alpha_act
        )
        out[active] = np.clip(blended, 0.0, 255.0).astype(np.uint8)
    return out


def composite_model_on_frame(
    frame_bgr: np.ndarray,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    camera: PinholeCamera | CameraCalibration,
    registration: ReferenceRegistration | np.ndarray | None = None,
    glyphs: GlyphSet | None = None,
    loads: AxialLoadFrame | None = None,
    opts: CompositeOptions | None = None,
) -> tuple[np.ndarray, FootageCompositeReceipt]:
    """Render calibrated MuJoCo model meshes and force glyphs composited onto footage.

    Args:
        frame_bgr: Source video footage frame (H, W, 3) in BGR uint8 format.
        model: Compiled MuJoCo MjModel.
        data: Synchronized MuJoCo MjData state.
        camera: Calibrated PinholeCamera or CameraCalibration.
        registration: ReferenceRegistration or 4x4 world_from_mj transform.
        glyphs: Optional GlyphSet of force/torque vectors (FTO-6).
        loads: Optional AxialLoadFrame for segment tension/compression shading.
        opts: CompositeOptions controlling opacity, feathering, and distortion.

    Returns:
        tuple[np.ndarray, FootageCompositeReceipt]: Composited BGR frame and receipt.
    """
    import mujoco

    cam = _resolve_camera(camera)
    w_px, h_px = cam.image_size_px

    if not isinstance(frame_bgr, np.ndarray) or frame_bgr.ndim != 3:
        raise ValueError("frame_bgr must be a 3D numpy array (H, W, 3)")
    if (frame_bgr.shape[1], frame_bgr.shape[0]) != (w_px, h_px):
        raise ValueError(
            f"frame_bgr dimensions ({frame_bgr.shape[1]}, {frame_bgr.shape[0]}) "
            f"must match camera image_size_px ({w_px}, {h_px})"
        )

    options = opts or CompositeOptions()
    world_from_mj = registration_to_world_from_mj(registration)
    spec = mujoco_camera_from_pinhole(cam, world_from_mj)

    renderer = mujoco.Renderer(model, width=spec.render_w, height=spec.render_h)
    renderer.update_scene(data)
    apply_camera_spec_to_scene(renderer.scene, spec)

    if loads is not None:
        scale = options.color_scale or ForceColorScale(enabled=True)
        apply_mujoco_scene_colors(model, renderer.scene, loads, scale)

    glyphs_drawn = 0
    glyphs_dropped = 0
    if glyphs is not None:
        glyph_receipt = add_glyphs_to_scene(renderer.scene, glyphs)
        glyphs_drawn = glyph_receipt.added
        glyphs_dropped = glyph_receipt.dropped

    cropped_seg, cropped_rgb = _render_scene_and_segmentation(renderer, spec.crop)

    alpha = _compute_feathered_alpha(
        cropped_seg, options.opacity, options.feather_px, options.athlete_mask
    )
    model_pixels = int(np.sum(cropped_seg[:, :, 0] != -1))

    frame_undistorted = False
    dest_frame = frame_bgr
    if (
        options.undistort_footage
        and cam.distortion is not None
        and np.any(cam.distortion != 0)
    ):
        dest_frame = cv2.undistort(frame_bgr, cam.matrix, cam.distortion)
        frame_undistorted = True

    out = _apply_alpha_composite(dest_frame, cropped_rgb, alpha)

    receipt = FootageCompositeReceipt(
        camera_spec=spec,
        glyphs_drawn=glyphs_drawn,
        glyphs_dropped=glyphs_dropped,
        model_pixels=model_pixels,
        frame_undistorted=frame_undistorted,
    )
    return out, receipt
