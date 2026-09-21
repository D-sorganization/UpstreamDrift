"""Headless replay GIF rendering for MyoSuite kinematic evidence (MS-52)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from typing import TypeAlias

from numpy.typing import NDArray

from src.shared.python.contracts import precondition
from src.shared.python.motion_matching.cross_engine_replay import (
    render_marker_overlay_animation,
)

Array: TypeAlias = NDArray[np.float64]
BoolArray: TypeAlias = NDArray[np.bool_]


@precondition(lambda time_s: len(time_s) > 0, "non-empty timeline")
def render_playback_gif(
    time_s: Array,
    target_markers_m: Array,
    model_markers_m: Array,
    output_path: Path | str,
    *,
    valid_mask: BoolArray | None = None,
    stride: int = 8,
) -> Path:
    """Write a marker overlay GIF comparing source and MyoSuite predictions."""
    return render_marker_overlay_animation(
        time_s=time_s,
        target_markers_m=target_markers_m,
        model_markers_m=model_markers_m,
        output_gif_path=output_path,
        engine_name="myosuite",
        stride=stride,
        valid_mask=valid_mask,
    )
