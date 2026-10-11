"""Projected 2D force/torque overlays for viewers without a 3D glyph API.

The simbody visualizer cannot take dynamic 3D decorations from Python, so its
frames get the shared OpenCV glyph renderer through a pinhole camera built
from the golf view preset (OpenCV convention: x right, y down, z forward).
"""

from __future__ import annotations

from collections.abc import Sequence
import math

import numpy as np
from numpy.typing import NDArray

from src.motion_capture.reconstruct.cameras import PinholeCamera
from src.shared.python.force_overlay.glyphs import GlyphSet
from src.shared.python.force_overlay.renderers.opencv_glyphs import (
    PinholeProjector,
    VideoGlyphStyle,
    draw_glyphs_on_frame,
)
from src.shared.python.golf_view_presets import ViewPreset, get_view_preset

Image8 = NDArray[np.uint8]


def pinhole_for_view(
    view: str | ViewPreset,
    lookat_m: Sequence[float],
    distance_m: float | None,
    fov_y_rad: float,
    size_px: tuple[int, int],
) -> PinholeCamera:
    """Pinhole camera matching a viewer camera placed by the view preset.

    ``size_px`` is ``(width, height)``; ``fov_y_rad`` is the vertical field of
    view. Postcondition: the look-at point projects to the image centre.
    """
    preset = get_view_preset(view) if isinstance(view, str) else view
    width, height = size_px
    if width < 1 or height < 1:
        raise ValueError("size_px must be positive")
    if not 0.0 < fov_y_rad < math.pi:
        raise ValueError("fov_y_rad must be in (0, pi)")
    dist = preset.default_distance_m if distance_m is None else float(distance_m)
    focal = (height / 2.0) / math.tan(fov_y_rad / 2.0)
    matrix = np.array(
        [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0, 0, 1.0]]
    )
    rotation = np.column_stack(
        [preset.image_right(), -preset.image_up(), preset.view_direction()]
    )
    return PinholeCamera(
        camera_id=f"preset_{preset.name}",
        matrix=matrix,
        rotation_world_from_camera=rotation,
        translation_world_from_camera_m=preset.camera_position(lookat_m, dist),
        image_size_px=(width, height),
    )


def draw_glyphs_rgb(
    frame_rgb: Image8, glyphs: GlyphSet, camera: PinholeCamera
) -> Image8:
    """Draw ``glyphs`` onto an RGB frame; returns a new array."""
    bgr = np.ascontiguousarray(frame_rgb[:, :, ::-1])
    draw_glyphs_on_frame(
        bgr,
        glyphs,
        PinholeProjector(camera),
        style=VideoGlyphStyle(legend_box=False, line_px=4, halo_px=6),
        inplace=True,
    )
    return np.ascontiguousarray(bgr[:, :, ::-1])
