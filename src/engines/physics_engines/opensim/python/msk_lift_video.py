"""Public lift video API (camera, overlays, and motion solve; render in follow-up PR)."""

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
from src.engines.physics_engines.opensim.python.msk_lift_video_motion import (
    solve_full_lift_motion_and_muscles,
    solve_full_lift_motion_targets,
    solve_lift_motion_and_muscles,
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
    "solve_full_lift_motion_and_muscles",
    "solve_full_lift_motion_targets",
    "solve_lift_motion_and_muscles",
]
