"""Silhouette rendering, analytic projection, and valid-pixel residual losses (ST-05).

This module implements:
- `PinholeCameraModel`: Slotted frozen camera model with distortion, cropping, and mirroring.
- `project_point_to_pixel()`: Analytic 3D world landmark projection into 2D pixel coordinates.
- `AnalyticSilhouetteRenderer`: Reference renderer fulfilling the `SilhouetteRenderer` protocol.
- `SilhouetteLossResult`: Evaluation record for valid-pixel aware silhouette residuals.
- `compute_silhouette_loss()`: Objective loss comparing candidate render results with observed masks.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math

from ._validation import (
    check_id,
    check_pos_int,
    check_strict_float,
)
from .contracts import (
    RenderRequest,
    RenderResult,
    SilhouetteRenderer,
)
from .mask_records import MaskFrame


# ---------------------------------------------------------------------------
# 1. Pinhole Camera Model and Analytic Projection
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class PinholeCameraModel:
    """Slotted camera geometry model supporting distortion, mirroring, and cropping."""

    camera_id: str
    width_px: int
    height_px: int
    fx: float
    fy: float
    cx: float
    cy: float
    k1: float = 0.0
    k2: float = 0.0
    p1: float = 0.0
    p2: float = 0.0
    k3: float = 0.0
    rotation_world_to_camera: tuple[float, ...] = (
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
    )
    translation_world_to_camera: tuple[float, ...] = (0.0, 0.0, 0.0)
    crop_box: tuple[int, int, int, int] | None = None
    is_mirrored: bool = False

    def __post_init__(self) -> None:
        check_id(self.camera_id, "camera_id")
        check_pos_int(self.width_px, "width_px")
        check_pos_int(self.height_px, "height_px")
        check_strict_float(self.fx, "fx")
        check_strict_float(self.fy, "fy")
        check_strict_float(self.cx, "cx")
        check_strict_float(self.cy, "cy")
        if self.fx <= 0.0 or self.fy <= 0.0:
            raise ValueError(
                f"Focal lengths must be positive, got fx={self.fx}, fy={self.fy}"
            )
        if len(self.rotation_world_to_camera) != 9:
            raise ValueError(
                f"rotation_world_to_camera must have 9 elements, got {len(self.rotation_world_to_camera)}"
            )
        if len(self.translation_world_to_camera) != 3:
            raise ValueError(
                f"translation_world_to_camera must have 3 elements, got {len(self.translation_world_to_camera)}"
            )
        if self.crop_box is not None:
            if len(self.crop_box) != 4:
                raise ValueError(
                    f"crop_box must have 4 elements, got {len(self.crop_box)}"
                )
            min_x, min_y, max_x, max_y = self.crop_box
            if not (
                0 <= min_x < max_x <= self.width_px
                and 0 <= min_y < max_y <= self.height_px
            ):
                raise ValueError(f"Invalid crop_box boundaries: {self.crop_box}")


def project_point_to_pixel(
    point_world: tuple[float, float, float],
    camera: PinholeCameraModel,
    *,
    depth_epsilon: float = 1.0e-9,
) -> tuple[float, float, bool]:
    """Project a 3D world coordinate into 2D camera pixels.

    Returns:
        `(pixel_u, pixel_v, is_visible)` where `is_visible` indicates whether the point
        is strictly in front of the camera ($Z_c > 0$) and within effective image bounds.
    """
    if not isinstance(camera, PinholeCameraModel):
        raise TypeError(f"Expected PinholeCameraModel, got {type(camera).__name__}")
    if len(point_world) != 3:
        raise ValueError(f"point_world must have 3 elements, got {len(point_world)}")

    xw, yw, zw = point_world
    r = camera.rotation_world_to_camera
    t = camera.translation_world_to_camera

    # Transform from world frame to camera frame: p_c = R @ p_w + t
    xc = r[0] * xw + r[1] * yw + r[2] * zw + t[0]
    yc = r[3] * xw + r[4] * yw + r[5] * zw + t[1]
    zc = r[6] * xw + r[7] * yw + r[8] * zw + t[2]

    # Check depth: points on or behind camera plane are not visible
    if zc <= depth_epsilon:
        return 0.0, 0.0, False

    # Normalized image coordinates
    xn = xc / zc
    yn = yc / zc

    # Brown-Conrady lens distortion
    r2 = xn * xn + yn * yn
    radial = 1.0 + camera.k1 * r2 + camera.k2 * (r2 * r2) + camera.k3 * (r2 * r2 * r2)
    xd = xn * radial + 2.0 * camera.p1 * xn * yn + camera.p2 * (r2 + 2.0 * xn * xn)
    yd = yn * radial + camera.p1 * (r2 + 2.0 * yn * yn) + 2.0 * camera.p2 * xn * yn

    # Intrinsics projection
    u = camera.fx * xd + camera.cx
    v = camera.fy * yd + camera.cy

    # Mirroring (horizontal flip)
    if camera.is_mirrored:
        u = float(camera.width_px - 1) - u

    # Cropping
    if camera.crop_box is not None:
        min_x, min_y, max_x, max_y = camera.crop_box
        eff_width = max_x - min_x
        eff_height = max_y - min_y
        u = u - float(min_x)
        v = v - float(min_y)
    else:
        eff_width = camera.width_px
        eff_height = camera.height_px

    # Visibility / image boundary check
    is_in_bounds = (0.0 <= u < float(eff_width)) and (0.0 <= v < float(eff_height))
    return u, v, is_in_bounds


# ---------------------------------------------------------------------------
# 2. Analytic Silhouette Renderer Adapter
# ---------------------------------------------------------------------------


class AnalyticSilhouetteRenderer:
    """Slotted reference renderer fulfilling the `SilhouetteRenderer` protocol."""

    __slots__ = ("_cameras",)

    def __init__(self, cameras: Mapping[str, PinholeCameraModel]) -> None:
        self._cameras = dict(cameras)

    def render(self, request: RenderRequest) -> RenderResult:
        """Render calibrated body and club silhouettes from state vector."""
        if not isinstance(request, RenderRequest):
            raise TypeError(f"Expected RenderRequest, got {type(request).__name__}")
        if request.camera_id not in self._cameras:
            raise KeyError(
                f"Camera ID {request.camera_id!r} not configured in renderer"
            )

        camera = self._cameras[request.camera_id]
        width, height = request.image_size_px
        total_px = width * height

        body_mask = [0] * total_px
        club_mask = [0] * total_px
        vis_mask = [1] * total_px

        state = request.state
        # State vector: first 3 elements are body landmark, next 3 are clubhead landmark
        if len(state) >= 3:
            bx, by, bz = state[0], state[1], state[2]
            bu, bv, b_vis = project_point_to_pixel((bx, by, bz), camera)
            if b_vis:
                col = int(round(bu))
                row = int(round(bv))
                if 0 <= row < height and 0 <= col < width:
                    body_mask[row * width + col] = 1

        if len(state) >= 6:
            cx, cy, cz = state[3], state[4], state[5]
            cu, cv, c_vis = project_point_to_pixel((cx, cy, cz), camera)
            if c_vis:
                col = int(round(cu))
                row = int(round(cv))
                if 0 <= row < height and 0 <= col < width:
                    club_mask[row * width + col] = 1

        return RenderResult(
            body_mask=tuple(body_mask),
            club_mask=tuple(club_mask),
            visibility_mask=tuple(vis_mask),
        )


# ---------------------------------------------------------------------------
# 3. Valid-Pixel Aware Silhouette Loss Formulation
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class SilhouetteLossResult:
    """Evaluation audit of rendered silhouette agreement against observed masks."""

    body_iou: float
    body_dice: float
    club_iou: float
    club_dice: float
    combined_loss: float
    valid_pixel_count: int
    is_valid: bool

    def __post_init__(self) -> None:
        check_strict_float(self.body_iou, "body_iou")
        check_strict_float(self.body_dice, "body_dice")
        check_strict_float(self.club_iou, "club_iou")
        check_strict_float(self.club_dice, "club_dice")
        check_strict_float(self.combined_loss, "combined_loss")


def _compute_channel_metrics(
    candidate: Sequence[int],
    observed: Sequence[int],
    valid: Sequence[int],
) -> tuple[float, float]:
    """Compute IoU and Dice similarity for a channel strictly over valid pixels."""
    length = len(candidate)
    intersection = 0
    cand_count = 0
    obs_count = 0

    for i in range(length):
        if valid[i] != 0:
            c = 1 if candidate[i] != 0 else 0
            o = 1 if observed[i] != 0 else 0
            if c and o:
                intersection += 1
            if c:
                cand_count += 1
            if o:
                obs_count += 1

    union = cand_count + obs_count - intersection
    iou = 1.0 if union == 0 else float(intersection) / float(union)
    total = cand_count + obs_count
    dice = 1.0 if total == 0 else (2.0 * float(intersection)) / float(total)
    return iou, dice


def compute_silhouette_loss(
    rendered: RenderResult,
    observed: MaskFrame,
    *,
    body_weight: float = 0.5,
    club_weight: float = 0.5,
) -> SilhouetteLossResult:
    """Compute valid-pixel aware silhouette residual loss.

    Preconditions:
        - `rendered` must be an instance of `RenderResult`.
        - `observed` must be an instance of `MaskFrame`.
        - `body_weight` and `club_weight` must be non-negative.
    """
    if not isinstance(rendered, RenderResult):
        raise TypeError(f"Expected RenderResult, got {type(rendered).__name__}")
    if not isinstance(observed, MaskFrame):
        raise TypeError(f"Expected MaskFrame, got {type(observed).__name__}")

    total_px = len(observed.valid)
    if len(rendered.body_mask) != total_px or len(rendered.club_mask) != total_px:
        raise ValueError(
            f"Rendered mask length ({len(rendered.body_mask)}) must match observed mask length ({total_px})"
        )

    valid_count = observed.valid.count(1)
    if valid_count == 0:
        return SilhouetteLossResult(
            body_iou=0.0,
            body_dice=0.0,
            club_iou=0.0,
            club_dice=0.0,
            combined_loss=1.0,
            valid_pixel_count=0,
            is_valid=False,
        )

    body_iou, body_dice = _compute_channel_metrics(
        rendered.body_mask,
        observed.body,
        observed.valid,
    )
    club_iou, club_dice = _compute_channel_metrics(
        rendered.club_mask,
        observed.club,
        observed.valid,
    )

    combined_loss = body_weight * (1.0 - body_iou) + club_weight * (1.0 - club_iou)

    return SilhouetteLossResult(
        body_iou=body_iou,
        body_dice=body_dice,
        club_iou=club_iou,
        club_dice=club_dice,
        combined_loss=combined_loss,
        valid_pixel_count=valid_count,
        is_valid=True,
    )
