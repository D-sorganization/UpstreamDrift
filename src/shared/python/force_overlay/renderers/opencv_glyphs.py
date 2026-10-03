"""OpenCV video glyph renderer projecting force/torque glyphs onto frames (FTO-8, #11293).

Draws a GlyphSet onto a BGR video frame through a calibrated camera.
The renderer never modifies the input array unless inplace=True.
Points passed to this renderer must already be in the camera's world frame (ADR-0041).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import cv2
import numpy as np
import numpy.typing as npt

from src.motion_capture.reconstruct.cameras import PinholeCamera
from src.motion_capture.reference.registration import project_reference_to_camera
from src.shared.python.core.contracts import require
from src.shared.python.force_overlay.glyphs import (
    ArrowGlyph,
    GlyphSet,
    LegendSpec,
    TorqueArcGlyph,
)
from src.shared.python.force_overlay.palette import FORCE_KIND_PALETTE
from src.shared.python.motion_matching.historical_fit.contracts import CameraProjection
from src.shared.python.pose_estimation.observations import CameraCalibration

__all__ = [
    "HypothesisProjector",
    "ImageProjector",
    "PinholeProjector",
    "VideoGlyphReceipt",
    "VideoGlyphStyle",
    "draw_glyphs_on_frame",
    "draw_legend_box",
]


class ImageProjector(Protocol):
    """Protocol for projecting 3D world coordinates onto 2D image pixels."""

    @property
    def world_frame(self) -> str: ...

    def project(
        self, points_world: npt.NDArray[np.float64]
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_]]: ...


class PinholeProjector:
    """Projector wrapping PinholeCamera / CameraCalibration."""

    def __init__(
        self, camera: PinholeCamera | CameraCalibration, *, world_frame: str = "adr0041"
    ) -> None:
        self._camera = camera
        self._world_frame = str(world_frame)

    @property
    def world_frame(self) -> str:
        return self._world_frame

    @property
    def camera(self) -> PinholeCamera | CameraCalibration:
        return self._camera

    def project(
        self, points_world: npt.NDArray[np.float64]
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_]]:
        pts = np.asarray(points_world, dtype=float)
        if not np.all(np.isfinite(pts)):
            raise ValueError("points_world must contain only finite coordinates")
        return project_reference_to_camera(
            pts, np.ones(pts.shape[:-1], bool), self._camera, clip_image=False
        )


class HypothesisProjector:
    """Projector wrapping CameraProjection catching behind-camera errors per point."""

    def __init__(
        self, projection: CameraProjection, *, world_frame: str = "adr0041"
    ) -> None:
        self._projection = projection
        self._world_frame = str(world_frame)

    @property
    def world_frame(self) -> str:
        return self._world_frame

    @property
    def projection(self) -> CameraProjection:
        return self._projection

    def project(
        self, points_world: npt.NDArray[np.float64]
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_]]:
        pts = np.asarray(points_world, dtype=float)
        if not np.all(np.isfinite(pts)):
            raise ValueError("points_world must contain only finite coordinates")
        flat = pts.reshape(-1, 3)
        valid = np.zeros(flat.shape[0], bool)
        pixels = np.full((flat.shape[0], 2), np.nan)
        for i in range(flat.shape[0]):
            try:
                pixels[i] = self._projection.project(flat[i : i + 1])[0]
                valid[i] = True
            except ValueError:
                pass
        return pixels.reshape(*pts.shape[:-1], 2), valid.reshape(pts.shape[:-1])


@dataclass(frozen=True)
class VideoGlyphStyle:
    """Visual styling for video glyph rendering."""

    line_px: int | None = None
    halo_px: int | None = None
    head_px: int | None = None
    halo_color_bgr: tuple[int, int, int] = (16, 16, 16)
    halo_alpha: float = 0.7
    legend_box: bool = True
    legend_alpha: float = 0.3
    font_scale: float | None = None
    font_face: int = cv2.FONT_HERSHEY_DUPLEX

    def resolve_line_px(self, h: int) -> int:
        return self.line_px or max(1, int(round(2.5 * h / 1080.0)))

    def resolve_halo_px(self, line_px: int) -> int:
        return self.halo_px or line_px + 2

    def resolve_head_px(self, line_px: int) -> int:
        return self.head_px or 5 * line_px

    def resolve_font_scale(self, h: int) -> float:
        return self.font_scale or max(0.25, 0.45 * (h / 1080.0))


@dataclass(frozen=True)
class VideoGlyphReceipt:
    """Execution receipt of video glyph rendering."""

    drawn: int
    skipped_behind_camera: int
    skipped_out_of_frame: int
    unavailable_labels: tuple[str, ...]
    frame: np.ndarray | None = field(default=None, repr=False, compare=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "drawn": self.drawn,
            "skipped_behind_camera": self.skipped_behind_camera,
            "skipped_out_of_frame": self.skipped_out_of_frame,
            "unavailable_labels": list(self.unavailable_labels),
        }

    def __iter__(self) -> Any:
        yield self.frame
        yield self


def _rgba_to_bgr(rgba: tuple[float, float, float, float]) -> tuple[int, int, int]:
    return (
        int(round(rgba[2] * 255.0)),
        int(round(rgba[1] * 255.0)),
        int(round(rgba[0] * 255.0)),
    )


def _hex_to_bgr(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return (
        (int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16))
        if len(h) >= 6
        else (255, 255, 255)
    )


def _clip_segment(
    rect: tuple[int, int, int, int], p1: tuple[int, int], p2: tuple[int, int]
) -> tuple[bool, tuple[int, int], tuple[int, int]]:
    ok, s, e = cv2.clipLine(rect, p1, p2)
    return ok, (int(s[0]), int(s[1])), (int(e[0]), int(e[1]))


@dataclass
class _DrawContext:
    rect: tuple[int, int, int, int]
    head_px: int
    halo_px: int
    halo_img: np.ndarray
    halo_bgr: tuple[int, int, int]
    shafts: list[tuple[tuple[int, int], tuple[int, int], tuple[int, int, int]]]
    heads: list[tuple[np.ndarray, tuple[int, int, int]]]
    halo_w: int


def _draw_item(
    segments: list[tuple[np.ndarray, np.ndarray]],
    head: tuple[np.ndarray, np.ndarray] | None,
    bgr: tuple[int, int, int],
    ctx: _DrawContext,
) -> bool:
    any_drawn = False
    for p1, p2 in segments:
        ok, s, e = _clip_segment(
            ctx.rect, tuple(np.rint(p1).astype(int)), tuple(np.rint(p2).astype(int))
        )
        if ok:
            cv2.line(ctx.halo_img, s, e, ctx.halo_bgr, ctx.halo_px, cv2.LINE_AA)
            ctx.shafts.append((s, e, bgr))
            any_drawn = True
    if head is not None:
        tip_px, base_px = head
        v = tip_px - base_px
        vl = float(np.hypot(v[0], v[1]))
        u = v / vl if vl > 1e-4 else np.array([0.0, 1.0])
        norm = np.array([-u[1], u[0]]) * (float(ctx.head_px) / 2.0)
        poly = np.array(
            [np.rint(tip_px), np.rint(base_px + norm), np.rint(base_px - norm)],
            np.int32,
        )
        if any(0 <= pt[0] < ctx.rect[2] and 0 <= pt[1] < ctx.rect[3] for pt in poly):
            cv2.fillConvexPoly(ctx.halo_img, poly, ctx.halo_bgr, cv2.LINE_AA)
            cv2.polylines(
                ctx.halo_img, [poly], True, ctx.halo_bgr, ctx.halo_w, cv2.LINE_AA
            )
            ctx.heads.append((poly, bgr))
            any_drawn = True
    return any_drawn


def draw_legend_box(
    frame: np.ndarray,
    legend: LegendSpec,
    style: VideoGlyphStyle,
    *,
    qualification: str | None = None,
) -> None:
    """Render semi-transparent legend box in bottom-left corner."""
    h, w = frame.shape[:2]
    scale = style.resolve_font_scale(h)
    dy, margin = (
        max(16, int(round(26.0 * scale / 0.45))),
        max(10, int(round(20.0 * h / 1080.0))),
    )
    src = f" ({', '.join(legend.source_labels)})" if legend.source_labels else ""
    lines: list[tuple[str, tuple[int, int, int] | None]] = []
    if legend.engine:
        lines.append((f"Engine: {legend.engine}{src}", None))
    if legend.force_reference_n is not None:
        lines.append((f"Force ref: {legend.force_reference_n:.1f} N", None))
    if legend.torque_reference_nm is not None:
        lines.append((f"Torque ref: {legend.torque_reference_nm:.1f} N*m", None))
    if legend.unavailable_labels:
        lines.append(
            (f"Unavailable: {', '.join(legend.unavailable_labels)}", (100, 100, 255))
        )
    if qualification:
        lines.append((f"Note: {qualification}", (180, 220, 255)))
    for k in legend.kinds_present:
        lines.append((k, _hex_to_bgr(FORCE_KIND_PALETTE.get(k, "#FFFFFF"))))
    if not lines:
        return
    bw, bh = max(260, int(round(340.0 * w / 1920.0))), dy * (len(lines) + 1)
    x1, y1, x2, y2 = margin, max(0, h - margin - bh), min(w, margin + bw), h - margin
    box = frame[y1:y2, x1:x2]
    frame[y1:y2, x1:x2] = cv2.addWeighted(
        np.zeros_like(box), style.legend_alpha, box, 1.0 - style.legend_alpha, 0
    )
    cv2.rectangle(frame, (x1, y1), (x2, y2), (60, 60, 60), 1, cv2.LINE_AA)
    ty = y1 + dy
    for text, bgr in lines:
        tx = x1 + 10
        if bgr is not None:
            r = max(3, int(round(5.0 * scale / 0.45)))
            cv2.circle(frame, (tx + r, ty - r), r, bgr, -1, cv2.LINE_AA)
            tx += r * 2 + 8
        cv2.putText(
            frame,
            text,
            (tx, ty),
            style.font_face,
            scale,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )
        ty += dy


def _draw_arrows(
    glyphs: GlyphSet,
    projector: ImageProjector,
    ctx: _DrawContext,
) -> tuple[int, int, int]:
    """Draw arrows from GlyphSet. Returns (drawn, skipped_behind, skipped_out)."""
    drawn, skipped_behind, skipped_out = 0, 0, 0
    for arrow in glyphs.arrows:
        pts = np.array([arrow.tail_m, arrow.tip_m, arrow.head_base_m], float)
        pix, val = projector.project(pts)
        if not (val[0] and val[1]):
            skipped_behind += 1
            continue
        base = pix[2] if val[2] else pix[1]
        bgr = _rgba_to_bgr(arrow.rgba)
        if _draw_item([(pix[0], base)], (pix[1], base), bgr, ctx):
            drawn += 1
        else:
            skipped_out += 1
    return drawn, skipped_behind, skipped_out


def _draw_torque_arcs(
    glyphs: GlyphSet,
    projector: ImageProjector,
    ctx: _DrawContext,
) -> tuple[int, int, int]:
    """Draw torque arcs from GlyphSet. Returns (drawn, skipped_behind, skipped_out)."""
    drawn, skipped_behind, skipped_out = 0, 0, 0
    for arc in glyphs.torque_arcs:
        if not arc.polyline_m:
            continue
        pts = np.vstack(
            [
                np.array(arc.polyline_m, float),
                np.array([arc.head_tip_m, arc.head_base_m], float),
            ]
        )
        pix, val = projector.project(pts)
        n = len(arc.polyline_m)
        if not (val[0] and val[n - 1]):
            skipped_behind += 1
            continue
        segs = [(pix[i], pix[i + 1]) for i in range(n - 1) if val[i] and val[i + 1]]
        head = (pix[n], pix[n + 1]) if (val[n] and val[n + 1]) else None
        bgr = _rgba_to_bgr(arc.rgba)
        if _draw_item(segs, head, bgr, ctx):
            drawn += 1
        else:
            skipped_out += 1
    return drawn, skipped_behind, skipped_out


def draw_glyphs_on_frame(
    frame_bgr: np.ndarray,
    glyphs: GlyphSet,
    projector: ImageProjector,
    *,
    style: VideoGlyphStyle | None = None,
    world_frame: str = "adr0041",
    inplace: bool = False,
    qualification: str | None = None,
) -> VideoGlyphReceipt:
    """Project and draw GlyphSet onto a BGR video frame."""
    require(isinstance(frame_bgr, np.ndarray), "frame_bgr must be a numpy ndarray")
    require(
        frame_bgr.dtype == np.uint8 and frame_bgr.ndim == 3 and frame_bgr.shape[2] == 3,
        "frame_bgr must be uint8 array with shape (H, W, 3)",
    )
    if world_frame != projector.world_frame:
        raise ValueError(
            f"Projector world frame mismatch: expected {world_frame}, got {projector.world_frame}"
        )

    out = frame_bgr if inplace else frame_bgr.copy()
    h, w = out.shape[:2]
    s = style or VideoGlyphStyle()
    line_px = s.resolve_line_px(h)
    halo_px = s.resolve_halo_px(line_px)
    head_px = s.resolve_head_px(line_px)
    halo_w = halo_px - line_px + 1
    halo_overlay = out.copy()
    shafts: list[tuple[tuple[int, int], tuple[int, int], tuple[int, int, int]]] = []
    heads: list[tuple[np.ndarray, tuple[int, int, int]]] = []

    ctx = _DrawContext(
        rect=(0, 0, w, h),
        head_px=head_px,
        halo_px=halo_px,
        halo_img=halo_overlay,
        halo_bgr=s.halo_color_bgr,
        shafts=shafts,
        heads=heads,
        halo_w=halo_w,
    )

    d_arr, behind_arr, out_arr = _draw_arrows(glyphs, projector, ctx)
    d_arc, behind_arc, out_arc = _draw_torque_arcs(glyphs, projector, ctx)
    drawn = d_arr + d_arc
    skipped_behind = behind_arr + behind_arc
    skipped_out = out_arr + out_arc

    cv2.addWeighted(halo_overlay, s.halo_alpha, out, 1.0 - s.halo_alpha, 0, out)
    for start, end, bgr in shafts:
        cv2.line(out, start, end, bgr, line_px, cv2.LINE_AA)
    for head_poly, bgr in heads:
        cv2.fillConvexPoly(out, head_poly, bgr, cv2.LINE_AA)
    if s.legend_box:
        draw_legend_box(out, glyphs.legend, s, qualification=qualification)

    return VideoGlyphReceipt(
        drawn=drawn,
        skipped_behind_camera=skipped_behind,
        skipped_out_of_frame=skipped_out,
        unavailable_labels=tuple(glyphs.legend.unavailable_labels),
        frame=out,
    )
