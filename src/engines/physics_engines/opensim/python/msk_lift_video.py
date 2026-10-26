"""Public lift video API (camera and overlays; render added in follow-up PRs)."""

from src.engines.physics_engines.opensim.python.msk_lift_video_framing import (
    BARBELL_LIFTS,
    LIFT_OVERLAYS,
    LIFT_VIEWS,
    SETTLE_S,
    UI_STRIP_PX,
    VIEWER_FOV_Y_RAD,
    LiftVideoConfig,
    compute_lift_framing,
    draw_lift_legend,
    extract_lift_wrenches,
    lift_camera_transforms,
    lift_video_schedule,
    muscle_color_rgb,
    pinhole_for_lifting_view,
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
