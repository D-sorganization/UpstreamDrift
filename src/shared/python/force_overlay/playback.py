"""Generic force overlay animated playback and video export (ADR-0052, #11302, FTO-17).

Engine-agnostic 3D animated playback with:
- Segment stick/capsule geometry shaded tension/compression via ForceColorScale
- 3D force and torque glyphs via draw_glyphs_3d
- Legend with scale bars and swatches via draw_legend
- Fixed global camera limits (equalize_3d_axes) to eliminate view jitter
- Exportable to MP4 via FFMpegWriter or PNG image sequence
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import io
import math
from pathlib import Path
from typing import Any, BinaryIO

import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from src.shared.python.body_part_viz.force_colors import ForceColorScale
from src.shared.python.force_overlay.conversions import SegmentAxis
from src.shared.python.force_overlay.glyphs import ForceGlyphStyle, build_glyphs
from src.shared.python.force_overlay.renderers.matplotlib_glyphs import (
    draw_glyphs_3d,
    draw_legend,
    equalize_3d_axes,
)
from src.shared.python.force_overlay.series import ForceTorqueSeries
from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)

__all__ = [
    "SegmentSeries",
    "PlaybackReceipt",
    "PlaybackOptions",
    "render_force_playback",
    "save_segment_series",
    "load_segment_series",
]


@dataclass(frozen=True)
class SegmentSeries:
    """Time-indexed segment endpoint geometry for body tension/compression shading."""

    names: tuple[str, ...]
    proximal: np.ndarray  # Shape: (T, N, 3)
    distal: np.ndarray  # Shape: (T, N, 3)

    def __post_init__(self) -> None:
        if not isinstance(self.names, tuple):
            object.__setattr__(self, "names", tuple(self.names))

        p = np.asarray(self.proximal, dtype=np.float64)
        d = np.asarray(self.distal, dtype=np.float64)

        if p.ndim != 3 or d.ndim != 3:
            raise ValueError(
                f"proximal and distal must be 3D arrays (T, N, 3), got shapes {p.shape} and {d.shape}"
            )
        if p.shape != d.shape:
            raise ValueError(
                f"proximal and distal shapes must match, got {p.shape} and {d.shape}"
            )
        if p.shape[2] != 3:
            raise ValueError(f"Coordinates dimension must be 3, got shape {p.shape}")
        if p.shape[1] != len(self.names):
            raise ValueError(
                f"Number of segment names ({len(self.names)}) does not match array second dimension ({p.shape[1]})"
            )
        if not (np.all(np.isfinite(p)) and np.all(np.isfinite(d))):
            raise ValueError(
                "proximal and distal arrays must contain only finite numbers"
            )

        object.__setattr__(self, "proximal", p)
        object.__setattr__(self, "distal", d)

    def to_npz(self, path_or_file: Path | str | BinaryIO) -> None:
        """Serialize segment endpoints to an NPZ archive (allow_pickle=False)."""
        np.savez(
            path_or_file,
            segment_names=np.array(self.names, dtype=str),
            proximal=self.proximal,
            distal=self.distal,
        )

    @classmethod
    def from_npz(cls, path_or_file: Path | str | BinaryIO) -> SegmentSeries:
        """Load segment endpoints from an NPZ archive."""
        with np.load(path_or_file, allow_pickle=False) as data:
            names = tuple(str(n) for n in data["segment_names"])
            proximal = np.asarray(data["proximal"], dtype=np.float64)
            distal = np.asarray(data["distal"], dtype=np.float64)
            return cls(names=names, proximal=proximal, distal=distal)


def save_segment_series(segments: SegmentSeries, path: Path | str) -> None:
    """Save SegmentSeries to an NPZ sidecar file."""
    segments.to_npz(path)


def load_segment_series(path: Path | str) -> SegmentSeries:
    """Load SegmentSeries from an NPZ sidecar file."""
    return SegmentSeries.from_npz(path)


@dataclass(frozen=True)
class PlaybackReceipt:
    """Receipt documenting playback render and export outcomes."""

    frame_count: int
    encoder: str  # 'ffmpeg' or 'png'
    out_path: Path
    size_px: tuple[int, int]
    fps: int
    segment_count: int


def _coerce_segment_series(
    segments: SegmentSeries | Mapping[str, Any] | Sequence[Sequence[SegmentAxis]],
    expected_frames: int,
) -> SegmentSeries:
    """Coerce various segment representations into a validated SegmentSeries."""
    if isinstance(segments, SegmentSeries):
        return segments

    if isinstance(segments, Mapping):
        names = tuple(str(n) for n in segments["segment_names"])
        p = np.asarray(segments["proximal"], dtype=np.float64)
        d = np.asarray(segments["distal"], dtype=np.float64)
        return SegmentSeries(names=names, proximal=p, distal=d)

    if isinstance(segments, Sequence):
        if len(segments) != expected_frames:
            raise ValueError(
                f"Sequence length ({len(segments)}) does not match expected frame count ({expected_frames})"
            )
        if expected_frames == 0:
            return SegmentSeries(
                names=(),
                proximal=np.empty((0, 0, 3)),
                distal=np.empty((0, 0, 3)),
            )

        first_frame = list(segments[0])
        names = tuple(axis.segment for axis in first_frame)
        num_segs = len(names)

        p = np.zeros((expected_frames, num_segs, 3), dtype=np.float64)
        d = np.zeros((expected_frames, num_segs, 3), dtype=np.float64)

        for t_idx, frame_axes in enumerate(segments):
            axis_map = {axis.segment: axis for axis in frame_axes}
            for s_idx, name in enumerate(names):
                if name in axis_map:
                    ax = axis_map[name]
                    p[t_idx, s_idx] = ax.proximal_m
                    d[t_idx, s_idx] = ax.distal_m

        return SegmentSeries(names=names, proximal=p, distal=d)

    raise TypeError(f"Unsupported segments type: {type(segments)}")


@dataclass(frozen=True)
class PlaybackOptions:
    """Render options for force playback."""

    fps: int = 30
    size_px: tuple[int, int] = (1920, 1080)
    base_segment_color: str = "#808080"
    segment_line_width: float = 4.0
    camera_elevation: float = 20.0
    camera_azimuth: float = -60.0


def _compute_bounding_points(
    seg_series: SegmentSeries,
    series: ForceTorqueSeries,
    style: ForceGlyphStyle,
) -> np.ndarray:
    points_collector: list[np.ndarray] = []
    if seg_series.proximal.size > 0:
        points_collector.append(seg_series.proximal.reshape(-1, 3))
        points_collector.append(seg_series.distal.reshape(-1, 3))

    for frame in series:
        for w in frame.wrenches:
            points_collector.append(np.array([w.point_m]))
            if w.force_n is not None:
                tip = (
                    np.array(w.point_m)
                    + np.array(w.force_n) * style.force_scale_m_per_n
                )
                points_collector.append(np.array([tip]))

    if points_collector:
        return np.vstack(points_collector)
    return np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])


@dataclass(frozen=True)
class _RenderContext:
    fig: plt.Figure
    series: ForceTorqueSeries
    seg_series: SegmentSeries
    style: ForceGlyphStyle
    color_scale: ForceColorScale
    opts: PlaybackOptions
    limits: tuple[Any, Any, Any]


def _draw_single_frame(ctx: _RenderContext, t_idx: int) -> None:
    ax = ctx.fig.axes[0]
    xlim, ylim, zlim = ctx.limits
    ax.cla()
    ax.view_init(elev=ctx.opts.camera_elevation, azim=ctx.opts.camera_azimuth)
    ax.set_xlim3d(xlim)
    ax.set_ylim3d(ylim)
    ax.set_zlim3d(zlim)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")

    frame = ctx.series[t_idx]
    ax.set_title(f"Time: {frame.time_s:.3f} s")

    axial_map = frame.axial_loads.values_n if frame.axial_loads is not None else {}
    for s_idx, name in enumerate(ctx.seg_series.names):
        p = ctx.seg_series.proximal[t_idx, s_idx]
        d = ctx.seg_series.distal[t_idx, s_idx]
        load_val = axial_map.get(name)
        color = ctx.color_scale.color(load_val, ctx.opts.base_segment_color)
        ax.plot(
            [p[0], d[0]],
            [p[1], d[1]],
            [p[2], d[2]],
            color=color,
            linewidth=ctx.opts.segment_line_width,
            solid_capstyle="round",
        )

    glyphs = build_glyphs(frame, style=ctx.style)
    draw_glyphs_3d(ax, glyphs)
    draw_legend(ax, glyphs.legend)


def _resolve_playback_options(
    options: PlaybackOptions | None,
    kwargs: Mapping[str, Any],
) -> PlaybackOptions:
    fps = kwargs.get("fps", options.fps if options else 30)
    size_px = kwargs.get("size_px", options.size_px if options else (1920, 1080))
    base_segment_color = kwargs.get(
        "base_segment_color", options.base_segment_color if options else "#808080"
    )
    segment_line_width = kwargs.get(
        "segment_line_width", options.segment_line_width if options else 4.0
    )
    camera_elevation = kwargs.get(
        "camera_elevation", options.camera_elevation if options else 20.0
    )
    camera_azimuth = kwargs.get(
        "camera_azimuth", options.camera_azimuth if options else -60.0
    )
    return PlaybackOptions(
        fps=fps,
        size_px=size_px,
        base_segment_color=base_segment_color,
        segment_line_width=segment_line_width,
        camera_elevation=camera_elevation,
        camera_azimuth=camera_azimuth,
    )


def _render_frames(
    ctx: _RenderContext,
    out_target: Path,
    encoder: str,
    png_dir: Path | None,
) -> None:
    if encoder == "ffmpeg":
        writer = animation.FFMpegWriter(fps=ctx.opts.fps, bitrate=4000)
        with writer.saving(ctx.fig, out_target, dpi=100):
            for t_idx in range(len(ctx.series)):
                _draw_single_frame(ctx, t_idx)
                writer.grab_frame()
    else:
        assert png_dir is not None
        for t_idx in range(len(ctx.series)):
            _draw_single_frame(ctx, t_idx)
            frame_path = png_dir / f"frame_{t_idx:04d}.png"
            ctx.fig.savefig(frame_path, dpi=100)


def render_force_playback(
    series: ForceTorqueSeries,
    segments: SegmentSeries | Mapping[str, Any] | Sequence[Sequence[SegmentAxis]],
    *,
    out_path: Path | str,
    style: ForceGlyphStyle | None = None,
    color_scale: ForceColorScale | None = None,
    options: PlaybackOptions | None = None,
    **kwargs: Any,
) -> PlaybackReceipt:
    """Render animated 3D force playback to video (MP4) or image frames (PNG)."""
    if not isinstance(series, ForceTorqueSeries):
        raise TypeError("series must be a ForceTorqueSeries")
    if len(series) == 0:
        raise ValueError("series must not be empty")

    opts = _resolve_playback_options(options, kwargs)
    if opts.fps <= 0:
        raise ValueError("fps must be positive")
    if opts.size_px[0] <= 0 or opts.size_px[1] <= 0:
        raise ValueError("size_px dimensions must be positive")

    seg_series = _coerce_segment_series(segments, len(series))
    if seg_series.proximal.shape[0] != len(series):
        raise ValueError(
            f"Length mismatch: series has {len(series)} frames but segments has {seg_series.proximal.shape[0]}"
        )

    out_target = Path(out_path)
    style = style or ForceGlyphStyle()
    color_scale = color_scale or ForceColorScale(enabled=True)

    can_ffmpeg = animation.FFMpegWriter.isAvailable() and out_target.suffix.lower() in (
        ".mp4",
        ".mov",
        ".avi",
    )
    encoder = "ffmpeg" if can_ffmpeg else "png"

    all_points = _compute_bounding_points(seg_series, series, style)

    fig_w = opts.size_px[0] / 100.0
    fig_h = opts.size_px[1] / 100.0
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=100)
    ax = fig.add_subplot(111, projection="3d")

    equalize_3d_axes(ax, all_points)
    limits = (ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d())

    png_dir: Path | None = None
    if encoder == "png":
        if out_target.suffix.lower() in (".mp4", ".mov", ".avi", ".png"):
            png_dir = out_target.parent / out_target.stem
        else:
            png_dir = out_target
        png_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_target.parent.mkdir(parents=True, exist_ok=True)

    ctx = _RenderContext(
        fig=fig,
        series=series,
        seg_series=seg_series,
        style=style,
        color_scale=color_scale,
        opts=opts,
        limits=limits,
    )

    try:
        _render_frames(
            ctx,
            out_target,
            encoder,
            png_dir,
        )
    finally:
        plt.close(fig)

    return PlaybackReceipt(
        frame_count=len(series),
        encoder=encoder,
        out_path=out_target if encoder == "ffmpeg" else (png_dir or out_target),
        size_px=opts.size_px,
        fps=opts.fps,
        segment_count=len(seg_series.names),
    )
