"""Pose Editor Module.

Provides shared functionality for model pose manipulation and editing
across all physics engines (Drake, Pinocchio, MuJoCo).

Key Features:
- Joint manipulation (sliders, spinboxes, direct entry)
- Gravity toggle for static posing
- Pose library (save/load/export/import/interpolate)
- Preset poses for common configurations
- Drag-and-drop manipulation (where supported)
- Constraint system for IK-based posing
"""

from .core import (
    JointInfo,
    JointType,
    PoseEditorInterface,
    PoseEditorState,
)
from .library import (
    PRESET_POSES,
    PoseInterpolator,
    PoseLibrary,
    PresetPoseCategory,
    StoredPose,
)
from .widgets import (
    GravityControlWidget,
    JointSliderWidget,
    PoseEditorWidget,
    PoseLibraryWidget,
)

__all__ = [
    # Core types
    "JointInfo",
    "JointType",
    "PoseEditorInterface",
    "PoseEditorState",
    # Library
    "PoseLibrary",
    "StoredPose",
    "PoseInterpolator",
    "PresetPoseCategory",
    "PRESET_POSES",
    # Widgets
    "PoseEditorWidget",
    "JointSliderWidget",
    "PoseLibraryWidget",
    "GravityControlWidget",
]
