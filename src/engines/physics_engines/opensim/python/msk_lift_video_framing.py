"""Framing and overlay helpers for OpenSim lift videos (OSL-8, #13045)."""

from src.engines.physics_engines.opensim.python.msk_lift_video_camera import (
    BARBELL_LIFTS,
    LIFT_OVERLAYS,
    LIFT_VIEWS,
    SETTLE_S,
    UI_STRIP_PX,
    VIEWER_FOV_Y_RAD,
    LiftVideoConfig,
    compute_lift_framing,
    lift_camera_transforms,
    lift_video_schedule,
    pinhole_for_lifting_view,
)
from src.engines.physics_engines.opensim.python.msk_lift_video_overlays import (
    draw_lift_legend,
    extract_lift_wrenches,
    muscle_color_rgb,
)

__all__ = [
    "BARBELL_LIFTS",
    "LIFT_OVERLAYS",
    "LIFT_VIEWS",
    "SETTLE_S",
    "UI_STRIP_PX",
    "VIEWER_FOV_Y_RAD",
    "LiftVideoConfig",
    "compute_lift_framing",
    "draw_lift_legend",
    "extract_lift_wrenches",
    "lift_camera_transforms",
    "lift_video_schedule",
    "muscle_color_rgb",
    "pinhole_for_lifting_view",
]
