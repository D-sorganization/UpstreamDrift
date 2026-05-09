"""Self-contained Qt widgets for the starting-pose matcher (issue #4706).

Widgets here are deliberately decoupled from ``gui.py`` so they can be
unit-tested under ``QT_QPA_PLATFORM=offscreen`` without spinning up the
full matcher window.
"""

from __future__ import annotations

from .calibration_dialog import (
    CalibrationDialog,
    CalibrationResult,
    build_subject_record,
)
from .joint_slider_panel import (
    DEFAULT_JOINT_COORDS,
    PoseState,
    JointSliderPanel,
)

__all__ = [
    "CalibrationDialog",
    "CalibrationResult",
    "DEFAULT_JOINT_COORDS",
    "JointSliderPanel",
    "PoseState",
    "build_subject_record",
]
