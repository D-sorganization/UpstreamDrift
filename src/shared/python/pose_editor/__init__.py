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
    BasePoseEditor,
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

__all__ = [
    # Core types
    "BasePoseEditor",
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
]

# Conditionally import PyQt6-dependent widgets
# Note: AttributeError catches the case when QtWidgets is None and
# class definitions try to inherit from QtWidgets.QWidget
try:
    from .widgets import (
        GravityControlWidget,
        JointSliderWidget,
        PoseEditorWidget,
        PoseLibraryWidget,
    )

    __all__ += [
        "PoseEditorWidget",
        "JointSliderWidget",
        "PoseLibraryWidget",
        "GravityControlWidget",
    ]
except (ImportError, AttributeError):
    # PyQt6 not available - widgets not exported
    pass
