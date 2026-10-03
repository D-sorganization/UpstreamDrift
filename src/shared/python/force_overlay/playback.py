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


def render_force_playback(
    series: ForceTorqueSeries,
    segments: SegmentSeries | Mapping[str, Any] | Sequence[Sequence[SegmentAxis]],
    *,
    style: ForceGlyphStyle | None = None,
    color_scale: ForceColorScale | None = None,
    out_path: Path | str,
    fps: int = 30,
    size_px: tuple[int, int] = (1920, 1080),
    base_segment_color: str = "#808080",
    segment_line_width: float = 4.0,
    camera_elevation: float = 20.0,
    camera_azimuth: float = -60.0,
) -> PlaybackReceipt:
    """Render animated 3D force playback to video (MP4) or image frames (PNG).

    Parameters
    ----------
    series : ForceTorqueSeries
        Time series of force/torque frames with axial loads.
    segments : SegmentSeries | Mapping | Sequence[Sequence[SegmentAxis]]
        Segment endpoint geometry across frames.
    style : ForceGlyphStyle | None
        Glyph styling configuration (default: ForceGlyphStyle()).
    color_scale : ForceColorScale | None
        Tension/compression color mapping policy (default: ForceColorScale(enabled=True)).
    out_path : Path | str
        Output file path (.mp4) or directory path for PNG frames.
    fps : int
        Frames per second playback rate.
    size_px : tuple[int, int]
        (width, height) pixel dimensions.
    base_segment_color : str
        Hex color for neutral/unmeasured segments.
    segment_line_width : float
        Line width for rendered segment bones.
    camera_elevation : float
        3D view elevation angle [degrees].
    camera_azimuth : float
        3D view azimuth angle [degrees].

    Returns
    -------
    PlaybackReceipt
        Render metadata and encoder details.
    """
    if not isinstance(series, ForceTorqueSeries):
        raise TypeError("series must be a ForceTorqueSeries")
    if len(series) == 0:
        raise ValueError("series must not be empty")
    if fps <= 0:
        raise ValueError("fps must be positive")
    if size_px[0] <= 0 or size_px[1] <= 0:
        raise ValueError("size_px dimensions must be positive")

    seg_series = _coerce_segment_series(segments, len(series))
    if seg_series.proximal.shape[0] != len(series):
        raise ValueError(
            f"Length mismatch: series has {len(series)} frames but segments has {seg_series.proximal.shape[0]}"
        )

    out_target = Path(out_path)
    style = style or ForceGlyphStyle()
    color_scale = color_scale or ForceColorScale(enabled=True)

    # Determine encoder: ffmpeg if available and out_target is an MP4/video file
    can_ffmpeg = animation.FFMpegWriter.isAvailable() and out_target.suffix.lower() in (
        ".mp4",
        ".mov",
        ".avi",
    )
    encoder = "ffmpeg" if can_ffmpeg else "png"

    # Compute global 3D bounding box over all frames to fix camera limits (no jitter)
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
        all_points = np.vstack(points_collector)
    else:
        all_points = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])

    fig_w = size_px[0] / 100.0
    fig_h = size_px[1] / 100.0
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=100)
    ax = fig.add_subplot(111, projection="3d")

    equalize_3d_axes(ax, all_points)
    xlim = ax.get_xlim3d()
    ylim = ax.get_ylim3d()
    zlim = ax.get_zlim3d()

    png_dir: Path | None = None
    if encoder == "png":
        if out_target.suffix.lower() in (".mp4", ".mov", ".avi", ".png"):
            png_dir = out_target.parent / out_target.stem
        else:
            png_dir = out_target
        png_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_target.parent.mkdir(parents=True, exist_ok=True)

    def _draw_frame(t_idx: int) -> None:
        ax.cla()
        ax.view_init(elev=camera_elevation, azim=camera_azimuth)
        ax.set_xlim3d(xlim)
        ax.set_ylim3d(ylim)
        ax.set_zlim3d(zlim)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")

        frame = series[t_idx]
        ax.set_title(f"Time: {frame.time_s:.3f} s")

        # Draw segment sticks with tension/compression shading
        axial_map = frame.axial_loads.values_n if frame.axial_loads is not None else {}
        for s_idx, name in enumerate(seg_series.names):
            p = seg_series.proximal[t_idx, s_idx]
            d = seg_series.distal[t_idx, s_idx]
            load_val = axial_map.get(name)
            color = color_scale.color(load_val, base_segment_color)
            ax.plot(
                [p[0], d[0]],
                [p[1], d[1]],
                [p[2], d[2]],
                color=color,
                linewidth=segment_line_width,
                solid_capstyle="round",
            )

        # Draw 3D force & torque glyphs and legend
        glyphs = build_glyphs(frame, style=style)
        draw_glyphs_3d(ax, glyphs)
        draw_legend(ax, glyphs.legend)

    try:
        if encoder == "ffmpeg":
            writer = animation.FFMpegWriter(fps=fps, bitrate=4000)
            with writer.saving(fig, out_target, dpi=100):
                for t_idx in range(len(series)):
                    _draw_frame(t_idx)
                    writer.grab_frame()
        else:
            assert png_dir is not None
            for t_idx in range(len(series)):
                _draw_frame(t_idx)
                frame_path = png_dir / f"frame_{t_idx:04d}.png"
                fig.savefig(frame_path, dpi=100)
    finally:
        plt.close(fig)

    return PlaybackReceipt(
        frame_count=len(series),
        encoder=encoder,
        out_path=out_target if encoder == "ffmpeg" else (png_dir or out_target),
        size_px=size_px,
        fps=fps,
        segment_count=len(seg_series.names),
    )
