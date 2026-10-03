"""OpenCV video glyph renderer: projects force/torque glyphs onto video frames (FTO-8, #11293).

Renders a GlyphSet onto a BGR video frame through a calibrated camera.
Draws anti-aliased, haloed, resolution-scaled 3D force arrows, torque arcs,
and 2D viewport summary legend boxes.

Authority reuse:
  - PinholeCamera / project_reference_to_camera from src.motion_capture.
  - CameraProjection from src.shared.python.motion_matching.historical_fit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import cv2
import numpy as np
import numpy.typing as npt

from src.motion_capture.reconstruct.cameras import PinholeCamera
from src.motion_capture.reference.registration import project_reference_to_camera
from src.shared.python.force_overlay.glyphs import (
    ArrowGlyph,
    GlyphSet,
    LegendSpec,
    TorqueArcGlyph,
)
from src.shared.python.motion_matching.historical_fit.contracts import CameraProjection
from src.shared.python.pose_estimation.observations import CameraCalibration


@runtime_checkable
class ImageProjector(Protocol):
    """Protocol for projecting 3D world points into 2D camera pixel coordinates."""

    world_frame: str

    def project(
        self, points_world: npt.NDArray[np.float64]
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_]]:
        """Project world points into pixel coordinates.

        Returns:
            projected_px: (N, 2) array of pixel coordinates (NaN where invalid/behind camera).
            valid: (N,) boolean mask indicating points in front of camera.
        """
        ...


class PinholeProjector:
    """Adapts a calibrated PinholeCamera or CameraCalibration to ImageProjector."""

    def __init__(
        self,
        camera: PinholeCamera | CameraCalibration,
        world_frame: str = "adr0041_world",
    ) -> None:
        self.camera = camera
        self.world_frame = world_frame

    def project(
        self, points_world: npt.NDArray[np.float64]
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_]]:
        pts = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
        valid_mask = np.ones(len(pts), dtype=bool)
        return project_reference_to_camera(
            pts, valid_mask, self.camera, clip_image=False
        )


class HypothesisProjector:
    """Adapts a monocular CameraProjection hypothesis to ImageProjector."""

    def __init__(
        self,
        projection: CameraProjection,
        world_frame: str = "hypothesis_world",
    ) -> None:
        self.projection = projection
        self.world_frame = world_frame

    def project(
        self, points_world: npt.NDArray[np.float64]
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_]]:
        pts = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
        n = len(pts)
        px = np.full((n, 2), np.nan, dtype=np.float64)
        valid = np.zeros(n, dtype=bool)
        for i in range(n):
            try:
                res = self.projection.project(pts[i : i + 1])
                px[i] = res[0]
                valid[i] = True
            except ValueError:
                valid[i] = False
        return px, valid


@dataclass(frozen=True)
class VideoGlyphStyle:
    """Resolution-dependent styling for vector overlay on video frames."""

    line_px: int = 3
    halo_px: int = 5
    halo_color_bgr: tuple[int, int, int] = (16, 16, 16)
    halo_alpha: float = 0.7
    head_px: int = 15
    show_legend: bool = True
    legend_bg_alpha: float = 0.3
    font_scale: float = 1.0

    @classmethod
    def for_height(cls, height_px: int) -> VideoGlyphStyle:
        h = max(1, height_px)
        line_px = max(1, round(2.5 * h / 1080.0))
        halo_px = line_px + 2
        head_px = 5 * line_px
        font_scale = max(0.4, h / 1080.0)
        return cls(
            line_px=line_px,
            halo_px=halo_px,
            head_px=head_px,
            font_scale=font_scale,
        )


@dataclass(frozen=True)
class VideoGlyphReceipt:
    """Execution receipt summarizing drawn and skipped glyph counts."""

    drawn: int
    skipped_behind_camera: int
    skipped_out_of_frame: int
    unavailable_labels: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "drawn": self.drawn,
            "skipped_behind_camera": self.skipped_behind_camera,
            "skipped_out_of_frame": self.skipped_out_of_frame,
            "unavailable_labels": list(self.unavailable_labels),
        }


def _rgba_to_bgr(rgba: tuple[float, float, float, float]) -> tuple[int, int, int]:
    r = int(np.clip(round(rgba[0] * 255), 0, 255))
    g = int(np.clip(round(rgba[1] * 255), 0, 255))
    b = int(np.clip(round(rgba[2] * 255), 0, 255))
    return (b, g, r)


def _draw_segment_with_halo(
    frame: np.ndarray,
    p1_px: npt.NDArray[np.float64],
    p2_px: npt.NDArray[np.float64],
    color_bgr: tuple[int, int, int],
    style: VideoGlyphStyle,
) -> bool:
    h, w = frame.shape[:2]
    if not (np.isfinite(p1_px).all() and np.isfinite(p2_px).all()):
        return False
    if max(abs(p1_px[0]), abs(p1_px[1]), abs(p2_px[0]), abs(p2_px[1])) > 1e6:
        return False

    valid, pt1, pt2 = cv2.clipLine(
        (0, 0, w, h),
        (int(round(p1_px[0])), int(round(p1_px[1]))),
        (int(round(p2_px[0])), int(round(p2_px[1]))),
    )
    if not valid:
        return False

    if style.halo_px > style.line_px:
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.line(mask, pt1, pt2, 255, style.halo_px, cv2.LINE_AA)
        halo_mask = mask > 0
        if np.any(halo_mask):
            halo_col = np.array(style.halo_color_bgr, dtype=np.float32)
            orig = frame[halo_mask].astype(np.float32)
            blended = style.halo_alpha * halo_col + (1.0 - style.halo_alpha) * orig
            frame[halo_mask] = np.clip(blended, 0, 255).astype(np.uint8)

    cv2.line(frame, pt1, pt2, color_bgr, style.line_px, cv2.LINE_AA)
    return True


def _draw_arrow_head(
    frame: np.ndarray,
    tip_px: npt.NDArray[np.float64],
    base_px: npt.NDArray[np.float64],
    color_bgr: tuple[int, int, int],
    style: VideoGlyphStyle,
) -> bool:
    h, w = frame.shape[:2]
    if not (np.isfinite(tip_px).all() and np.isfinite(base_px).all()):
        return False
    v = tip_px - base_px
    length = float(np.linalg.norm(v))
    if length < 1e-4:
        return False

    u = v / length
    n = np.array([-u[1], u[0]], dtype=np.float64)
    w_half = style.head_px / 2.0
    c1 = base_px + w_half * n
    c2 = base_px - w_half * n

    # Halo head
    if style.halo_px > style.line_px:
        mask = np.zeros((h, w), dtype=np.uint8)
        w_halo = w_half + 1.5
        c1_h = base_px - u * 1.5 + w_halo * n
        c2_h = base_px - u * 1.5 - w_halo * n
        tip_h = tip_px + u * 2.0
        pts_h = np.rint(np.array([tip_h, c1_h, c2_h])).astype(np.int32)
        cv2.fillConvexPoly(mask, pts_h, 255, cv2.LINE_AA)
        halo_mask = mask > 0
        if np.any(halo_mask):
            halo_col = np.array(style.halo_color_bgr, dtype=np.float32)
            orig = frame[halo_mask].astype(np.float32)
            blended = style.halo_alpha * halo_col + (1.0 - style.halo_alpha) * orig
            frame[halo_mask] = np.clip(blended, 0, 255).astype(np.uint8)

    pts = np.rint(np.array([tip_px, c1, c2])).astype(np.int32)
    cv2.fillConvexPoly(frame, pts, color_bgr, cv2.LINE_AA)
    return True


def draw_legend_box(
    frame_bgr: np.ndarray,
    legend: LegendSpec,
    style: VideoGlyphStyle,
    qualification_note: str = "",
) -> None:
    """Draw a translucent legend overlay box in the bottom-left corner of the frame."""
    h, w = frame_bgr.shape[:2]
    lines: list[str] = []

    if legend.force_reference_n is not None:
        lines.append(f"Ref Force: {legend.force_reference_n:.1f} N")
    if legend.torque_reference_nm is not None:
        lines.append(f"Ref Torque: {legend.torque_reference_nm:.1f} N*m")
    if legend.kinds_present:
        lines.append("Kinds: " + ", ".join(legend.kinds_present))
    if legend.unavailable_labels:
        lines.append("Unavailable: " + ", ".join(legend.unavailable_labels))
    if legend.engine:
        lines.append(f"Engine: {legend.engine}")
    if qualification_note:
        lines.append(qualification_note)

    if not lines:
        return

    font = cv2.FONT_HERSHEY_DUPLEX
    f_scale = style.font_scale * 0.55
    thickness = 1
    line_spacing = int(round(22 * style.font_scale))

    box_w = 0
    for text in lines:
        (tw, th), _ = cv2.getTextSize(text, font, f_scale, thickness)
        box_w = max(box_w, tw)
    box_w += int(round(24 * style.font_scale))
    box_h = len(lines) * line_spacing + int(round(16 * style.font_scale))

    x0 = int(round(10 * style.font_scale))
    y0 = h - box_h - int(round(10 * style.font_scale))
    x1 = min(w - 1, x0 + box_w)
    y1 = min(h - 1, y0 + box_h)

    # 30% black background
    sub = frame_bgr[y0:y1, x0:x1]
    blended = (1.0 - style.legend_bg_alpha) * sub.astype(np.float32)
    frame_bgr[y0:y1, x0:x1] = np.clip(blended, 0, 255).astype(np.uint8)

    # Text lines
    tx = x0 + int(round(12 * style.font_scale))
    ty = y0 + int(round(18 * style.font_scale))
    for text in lines:
        cv2.putText(
            frame_bgr,
            text,
            (tx, ty),
            font,
            f_scale,
            (240, 240, 240),
            thickness,
            cv2.LINE_AA,
        )
        ty += line_spacing


def draw_glyphs_on_frame(
    frame_bgr: np.ndarray,
    glyphs: GlyphSet,
    projector: ImageProjector,
    *,
    world_frame: str,
    style: VideoGlyphStyle | None = None,
    inplace: bool = False,
    qualification_note: str = "",
) -> tuple[np.ndarray, VideoGlyphReceipt]:
    """Project and draw force/torque glyphs onto an OpenCV BGR video frame."""
    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3 or frame_bgr.dtype != np.uint8:
        raise ValueError("frame_bgr must be a uint8 image with shape (H, W, 3)")
    if world_frame != projector.world_frame:
        raise ValueError(
            f"Frame mismatch: glyphs world_frame '{world_frame}' does not match projector frame '{projector.world_frame}'"
        )

    out_frame = frame_bgr if inplace else frame_bgr.copy()
    h, w = out_frame.shape[:2]
    eff_style = style if style is not None else VideoGlyphStyle.for_height(h)

    drawn = 0
    skipped_behind = 0
    skipped_out_of_frame = 0

    # 1. Force Arrows
    for arrow in glyphs.arrows:
        pts_3d = np.array(
            [arrow.tail_m, arrow.tip_m, arrow.head_base_m], dtype=np.float64
        )
        px_2d, valid = projector.project(pts_3d)

        # Behind camera: must have both tail and tip valid
        if not (valid[0] and valid[1]):
            skipped_behind += 1
            continue

        tail_px = px_2d[0]
        tip_px = px_2d[1]
        base_px = px_2d[2] if valid[2] else tail_px + 0.8 * (tip_px - tail_px)

        col_bgr = _rgba_to_bgr(arrow.rgba)
        shaft_drawn = _draw_segment_with_halo(
            out_frame, tail_px, base_px, col_bgr, eff_style
        )
        head_drawn = _draw_arrow_head(out_frame, tip_px, base_px, col_bgr, eff_style)

        if shaft_drawn or head_drawn:
            drawn += 1
        else:
            skipped_out_of_frame += 1

    # 2. Torque Arcs
    for arc in glyphs.torque_arcs:
        pts_poly = list(arc.polyline_m)
        pts_3d = np.array(
            pts_poly + [arc.head_base_m, arc.head_tip_m], dtype=np.float64
        )
        px_2d, valid = projector.project(pts_3d)

        # Check endpoints and head
        if not (valid[0] and valid[-1] and valid[-2]):
            skipped_behind += 1
            continue

        col_bgr = _rgba_to_bgr(arc.rgba)
        arc_drawn = False
        n_poly = len(pts_poly)

        for i in range(n_poly - 1):
            if valid[i] and valid[i + 1]:
                seg_drawn = _draw_segment_with_halo(
                    out_frame, px_2d[i], px_2d[i + 1], col_bgr, eff_style
                )
                if seg_drawn:
                    arc_drawn = True

        head_drawn = _draw_arrow_head(
            out_frame, px_2d[-1], px_2d[-2], col_bgr, eff_style
        )
        if head_drawn:
            arc_drawn = True

        if arc_drawn:
            drawn += 1
        else:
            skipped_out_of_frame += 1

    # 3. Legend Box
    if eff_style.show_legend:
        draw_legend_box(out_frame, glyphs.legend, eff_style, qualification_note)

    receipt = VideoGlyphReceipt(
        drawn=drawn,
        skipped_behind_camera=skipped_behind,
        skipped_out_of_frame=skipped_out_of_frame,
        unavailable_labels=glyphs.legend.unavailable_labels,
    )
    return out_frame, receipt
