"""Headless video export for C3D-derived body targets."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from ..club_target import AlignOptions
from ..load_body_target import load_body_target_c3d
from ._skeleton_render import draw_body_target_frame, equalize_3d_axes


class _BodyTargetLike(Protocol):
    @property
    def marker_xyz(self) -> np.ndarray: ...

    @property
    def marker_names(self) -> tuple[str, ...]: ...


@dataclass(frozen=True)
class BodyTargetVideoResult:
    """Summary of a written body-target preview video."""

    output_path: Path
    frame_count: int
    fps: float
    width: int
    height: int


def _resolve_frame_indices(
    n_frames: int,
    frame_indices: Sequence[int] | None,
) -> tuple[int, ...]:
    if n_frames <= 0:
        raise ValueError("target.marker_xyz must contain at least one frame")
    if frame_indices is None:
        return tuple(range(n_frames))
    resolved = tuple(int(i) for i in frame_indices)
    if not resolved:
        raise ValueError("frame_indices must contain at least one frame")
    invalid = [i for i in resolved if i < 0 or i >= n_frames]
    if invalid:
        raise ValueError(f"frame_indices out of range [0, {n_frames}): {invalid[:5]}")
    return resolved


def _validate_video_geometry(width: int, height: int, fps: float) -> None:
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    if width % 2 or height % 2:
        raise ValueError("width and height must be even for MP4 export")
    if fps <= 0.0:
        raise ValueError("fps must be positive")


def _finite_points(target: _BodyTargetLike) -> np.ndarray:
    xyz = np.asarray(target.marker_xyz, dtype=float)
    if xyz.ndim != 3 or xyz.shape[2] != 3:
        raise ValueError(
            f"target.marker_xyz must have shape (N, M, 3), got {xyz.shape}"
        )
    if xyz.shape[1] != len(tuple(target.marker_names)):
        raise ValueError(
            "target.marker_names length must match marker_xyz.shape[1]; "
            f"got {len(tuple(target.marker_names))} vs {xyz.shape[1]}"
        )
    finite = np.isfinite(xyz).all(axis=-1)
    if not finite.any():
        raise ValueError("target.marker_xyz contains no finite marker samples")
    return xyz[finite].reshape(-1, 3)


def _draw_frame(
    ax,
    target: _BodyTargetLike,
    frame_idx: int,
    points_for_limits: np.ndarray,
    title: str | None,
) -> None:
    ax.clear()
    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=10)
    draw_body_target_frame(ax, target, frame_idx, linewidth=2.0)  # type: ignore[arg-type]
    frame = np.asarray(target.marker_xyz, dtype=float)[frame_idx]
    finite = np.isfinite(frame).all(axis=-1)
    if finite.any():
        ax.scatter(
            frame[finite, 0],
            frame[finite, 1],
            frame[finite, 2],
            s=12,
            c="#111111",
            depthshade=False,
        )
    equalize_3d_axes(ax, points_for_limits)
    ax.view_init(elev=18.0, azim=-62.0)


def _canvas_rgb(canvas: FigureCanvasAgg, height: int, width: int) -> np.ndarray:
    canvas.draw()
    rgba = np.asarray(canvas.buffer_rgba(), dtype=np.uint8).reshape(height, width, 4)
    return np.ascontiguousarray(rgba[:, :, :3])


def save_body_target_video(
    target: _BodyTargetLike,
    output_path: Path | str,
    *,
    frame_indices: Sequence[int] | None = None,
    fps: float = 30.0,
    width: int = 960,
    height: int = 720,
    title: str | None = None,
) -> BodyTargetVideoResult:
    """Save a headless MP4 preview of a body target's skeleton.

    The target is usually returned by :func:`load_body_target_c3d`. Rendering
    uses matplotlib's Agg canvas and OpenCV's ``mp4v`` writer so it works in
    CI and on machines without an interactive display.
    """
    _validate_video_geometry(width, height, fps)
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - dependency gated
        raise ImportError(
            "opencv-python-headless is required to export C3D preview videos"
        ) from exc

    xyz = np.asarray(target.marker_xyz)
    frames = _resolve_frame_indices(int(xyz.shape[0]), frame_indices)
    points_for_limits = _finite_points(target)

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    dpi = 100
    fig = Figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111, projection="3d")

    video_writer_fourcc = cast(Any, cv2).VideoWriter_fourcc
    fourcc = video_writer_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise OSError(f"Could not open video writer for {path}")
    try:
        for frame_idx in frames:
            _draw_frame(ax, target, frame_idx, points_for_limits, title)
            rgb = _canvas_rgb(canvas, height, width)
            writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()

    return BodyTargetVideoResult(
        output_path=path,
        frame_count=len(frames),
        fps=float(fps),
        width=width,
        height=height,
    )


def save_c3d_body_video(
    c3d_path: Path | str,
    output_path: Path | str,
    *,
    opts: AlignOptions | None = None,
    marker_set: Iterable[str] | None = None,
    frame_indices: Sequence[int] | None = None,
    fps: float = 30.0,
    width: int = 960,
    height: int = 720,
    title: str | None = None,
) -> BodyTargetVideoResult:
    """Load a C3D body target and save a skeleton preview video."""
    target = load_body_target_c3d(
        c3d_path,
        opts or AlignOptions(),
        marker_set=tuple(marker_set) if marker_set is not None else None,
    )
    return save_body_target_video(
        target,
        output_path,
        frame_indices=frame_indices,
        fps=fps,
        width=width,
        height=height,
        title=title,
    )


__all__ = [
    "BodyTargetVideoResult",
    "save_body_target_video",
    "save_c3d_body_video",
]
