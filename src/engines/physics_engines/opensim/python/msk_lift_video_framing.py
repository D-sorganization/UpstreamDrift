"""Framing helpers for OpenSim lift videos (OSL-8, #13045)."""

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

__all__ = [
    "BARBELL_LIFTS",
    "LIFT_OVERLAYS",
    "LIFT_VIEWS",
    "SETTLE_S",
    "UI_STRIP_PX",
    "VIEWER_FOV_Y_RAD",
    "LiftVideoConfig",
    "compute_lift_framing",
    "lift_camera_transforms",
    "lift_video_schedule",
    "pinhole_for_lifting_view",
]
