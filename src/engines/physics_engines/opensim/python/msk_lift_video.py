"""Public lift video API (camera, overlays, motion solve, contact sheets)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

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
    "generate_contact_sheet",
    "lift_camera_transforms",
    "lift_video_schedule",
    "muscle_color_rgb",
    "pinhole_for_lifting_view",
    "solve_full_lift_motion_and_muscles",
    "solve_full_lift_motion_targets",
    "solve_lift_motion_and_muscles",
]


def generate_contact_sheet(
    frames_or_video: Path | Sequence[np.ndarray],
    out_path: Path,
    count: int = 6,
    window_frame_indices: Sequence[int] | None = None,
) -> Path:
    """Generate a 2x3 contact sheet of 6 representative frames.

    When ``window_frame_indices`` is provided, ensures at least 2 frames capture
    the active OSL-4 window with visible force overlays and muscle activations.
    """
    import imageio.v2 as imageio

    if isinstance(frames_or_video, Path):
        reader = imageio.get_reader(str(frames_or_video))
        all_frames = list(reader)  # type: ignore[call-overload]
        reader.close()
    else:
        all_frames = list(frames_or_video)

    total = len(all_frames)
    if (
        window_frame_indices is not None
        and len(window_frame_indices) > 0
        and count == 6
        and total >= 6
    ):
        w_start = int(window_frame_indices[0])
        w_end = int(window_frame_indices[-1])
        w_mid1 = w_start + (w_end - w_start) // 3
        w_mid2 = w_start + 2 * (w_end - w_start) // 3
        pre1 = 0
        pre2 = w_start // 2
        post1 = w_end + (total - 1 - w_end) // 2
        post2 = total - 1
        step_indices = np.array([pre1, pre2, w_mid1, w_mid2, post1, post2], dtype=int)
    else:
        step_indices = np.linspace(0, total - 1, count, dtype=int)

    frames = [all_frames[i] for i in step_indices]

    row0 = np.concatenate(frames[:3], axis=1)
    row1 = np.concatenate(frames[3:6], axis=1)
    sheet = np.concatenate([row0, row1], axis=0)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(str(out_path), sheet)
    return out_path
