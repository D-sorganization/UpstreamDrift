"""Unified swing playback and bidirectional overlay package (Epic #15660, #15669)."""

from .bidirectional_overlay import (
    BidirectionalOverlayConfig,
    OverlayMode,
    composite_bidirectional_overlay,
)
from .camera_perspective import (
    CameraViewParameters,
    frustum_quad_world_m,
    match_perspective_from_projection,
    projection_from_viewer_perspective,
    unproject_pixel_to_plane,
    unproject_pixel_to_ray,
)

__all__ = [
    "BidirectionalOverlayConfig",
    "OverlayMode",
    "composite_bidirectional_overlay",
    "CameraViewParameters",
    "match_perspective_from_projection",
    "projection_from_viewer_perspective",
    "unproject_pixel_to_ray",
    "unproject_pixel_to_plane",
    "frustum_quad_world_m",
]
