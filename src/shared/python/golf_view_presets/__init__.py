"""Shared golf camera view presets with per-engine camera adapters (NV-1, #11674).

Headless import safe: no engine SDK is imported here.
"""

from __future__ import annotations

from .adapters import (
    MeshcatCamera,
    MujocoFixedCamera,
    MujocoFreeCamera,
    drake_meshcat_camera_pose,
    meshcat_camera,
    mujoco_camera_params,
    mujoco_fixed_camera,
    simbody_camera_transform,
)
from .framing import (
    DEFAULT_FRAME_MARGIN,
    VIEWER_FOV_Y_RAD,
    fit_distance_m,
    projected_extent,
)
from .presets import (
    VIEW_ORDER,
    VIEW_PRESETS,
    ViewPreset,
    get_view_preset,
    tracked_lookats,
)

__all__ = [
    "DEFAULT_FRAME_MARGIN",
    "VIEWER_FOV_Y_RAD",
    "VIEW_ORDER",
    "VIEW_PRESETS",
    "MeshcatCamera",
    "MujocoFixedCamera",
    "MujocoFreeCamera",
    "ViewPreset",
    "drake_meshcat_camera_pose",
    "fit_distance_m",
    "get_view_preset",
    "meshcat_camera",
    "mujoco_camera_params",
    "mujoco_fixed_camera",
    "projected_extent",
    "simbody_camera_transform",
    "tracked_lookats",
]
