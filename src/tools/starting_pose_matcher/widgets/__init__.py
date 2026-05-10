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
from .run_fit_button import FitWorker, RunFitButton
from .save_fit_button import (
    FIT_RESULT_SCHEMA_VERSION,
    SaveFitButton,
    compute_source_file_sha256,
    serialize_fit_result,
)

__all__ = [
    "CalibrationDialog",
    "CalibrationResult",
    "DEFAULT_JOINT_COORDS",
    "FIT_RESULT_SCHEMA_VERSION",
    "FitWorker",
    "JointSliderPanel",
    "PoseState",
    "RunFitButton",
    "SaveFitButton",
    "build_subject_record",
    "compute_source_file_sha256",
    "serialize_fit_result",
]
