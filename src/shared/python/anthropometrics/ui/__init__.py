"""Qt UI surface for the anthropometrics package.

Public widgets that surface :class:`SegmentProperties` values to the
user. Importing this package is safe even when PyQt6 is not installed:
the panel module itself defers the PyQt6 import to module import time
and raises a clear ``ImportError`` only when something actually
constructs a widget.
"""

from __future__ import annotations

from .calibration_dialog import SubjectCalibrationDialog
from .segment_properties_panel import SegmentPropertiesPanel

__all__ = ["SegmentPropertiesPanel", "SubjectCalibrationDialog"]
