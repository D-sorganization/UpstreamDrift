"""Service for exporting C3D viewer animation to video."""

from __future__ import annotations

import logging
from pathlib import Path

from ..core.models import C3DDataModel

logger = logging.getLogger(__name__)


def export_animation(
    model: C3DDataModel,
    output_path: Path | str,
    fps: float | None = None,
    width: int = 960,
    height: int = 720,
    **kwargs,
):
    """Export the currently loaded C3D file to an MP4 video."""
    if fps is None:
        fps = model.point_rate if model.point_rate > 0 else 30.0

    if not model.filepath:
        raise ValueError("Cannot export animation: model has no loaded C3D file path")

    from src.shared.python.motion_matching.diagnostics.body_target_video import (
        save_c3d_body_video,
    )

    path = Path(output_path)
    if path.suffix.lower() != ".mp4":
        raise ValueError("Export path must have an .mp4 suffix")

    if fps <= 0.0:
        raise ValueError("fps must be positive")

    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")

    if width % 2 != 0 or height % 2 != 0:
        raise ValueError("width and height must be even for MP4 export")

    if not model.markers:
        raise ValueError("Model has no markers to export")

    logger.info(f"Exporting C3D animation from {model.filepath} to {path} at {fps} fps")

    return save_c3d_body_video(
        c3d_path=model.filepath,
        output_path=path,
        fps=fps,
        width=width,
        height=height,
        **kwargs,
    )
