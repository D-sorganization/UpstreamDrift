"""Per-engine camera adapters for the golf view presets (NV-1, #11674).

Every adapter returns plain tuples and dataclasses so it can be unit-tested
without MuJoCo, Drake, MeshCat or OpenSim installed.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .presets import ViewPreset, check_point3, get_view_preset

Vec3 = tuple[float, float, float]
Array = NDArray[np.float64]

# MeshCat (three.js) is Y-up; its scene root applies Rx(-90 deg) to Z-up data.
_RX_MINUS_90 = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])


def _tuple3(vec: Sequence[float] | Array) -> Vec3:
    return (float(vec[0]), float(vec[1]), float(vec[2]))


def _resolve(
    preset: str | ViewPreset,
    lookat_m: Sequence[float] | Array,
    distance_m: float | None,
) -> tuple[ViewPreset, Array, float]:
    p = get_view_preset(preset) if isinstance(preset, str) else preset
    look = check_point3(lookat_m, "lookat_m")
    dist = p.default_distance_m if distance_m is None else float(distance_m)
    p.camera_position(look, dist)  # validates the distance
    return p, look, dist


@dataclass(frozen=True)
class MujocoFreeCamera:
    """Arguments for ``mujoco.MjvCamera`` (free camera)."""

    azimuth: float
    elevation: float
    distance: float
    lookat: Vec3


def mujoco_camera_params(
    preset: str | ViewPreset,
    lookat_m: Sequence[float] | Array,
    distance_m: float | None = None,
) -> MujocoFreeCamera:
    """MuJoCo free-camera azimuth/elevation/distance/lookat for ``preset``."""
    p, look, dist = _resolve(preset, lookat_m, distance_m)
    return MujocoFreeCamera(
        azimuth=p.azimuth_deg,
        elevation=p.elevation_deg,
        distance=dist,
        lookat=_tuple3(look),
    )


@dataclass(frozen=True)
class MujocoFixedCamera:
    """Attributes for an MJCF ``<camera pos= xyaxes=>`` element."""

    position: Vec3
    xyaxes: tuple[float, float, float, float, float, float]


def mujoco_fixed_camera(
    preset: str | ViewPreset,
    lookat_m: Sequence[float] | Array,
    distance_m: float | None = None,
) -> MujocoFixedCamera:
    """Fixed MJCF camera (image-right then image-up axes) for ``preset``."""
    p, look, dist = _resolve(preset, lookat_m, distance_m)
    right, up = p.image_right(), p.image_up()
    return MujocoFixedCamera(
        position=_tuple3(p.camera_position(look, dist)),
        xyaxes=(*_tuple3(right), *_tuple3(up)),
    )


def drake_meshcat_camera_pose(
    preset: str | ViewPreset,
    lookat_m: Sequence[float] | Array,
    distance_m: float | None = None,
) -> tuple[Vec3, Vec3]:
    """``(position, target)`` in the Z-up world for ``Meshcat.SetCameraPose``.

    Drake's Meshcat applies the Z-up scene transform itself, so these are
    ordinary world coordinates.
    """
    p, look, dist = _resolve(preset, lookat_m, distance_m)
    return _tuple3(p.camera_position(look, dist)), _tuple3(look)


@dataclass(frozen=True)
class MeshcatCamera:
    """A camera in world (Z-up) and three.js (Y-up, Rx(-90 deg)) coordinates.

    ``node_offset_three`` is the camera position relative to its look-at
    target in the three.js frame, which is what the ``/Cameras/default``
    node of meshcat-python expects for its ``<object>`` child.
    """

    position_world: Vec3
    target_world: Vec3
    position_three: Vec3
    target_three: Vec3
    node_offset_three: Vec3


def meshcat_camera(
    preset: str | ViewPreset,
    lookat_m: Sequence[float] | Array,
    distance_m: float | None = None,
) -> MeshcatCamera:
    """MeshCat camera for ``preset`` including the Rx(-90 deg) scene transform."""
    p, look, dist = _resolve(preset, lookat_m, distance_m)
    pos = p.camera_position(look, dist)
    return MeshcatCamera(
        position_world=_tuple3(pos),
        target_world=_tuple3(look),
        position_three=_tuple3(_RX_MINUS_90 @ pos),
        target_three=_tuple3(_RX_MINUS_90 @ look),
        node_offset_three=_tuple3(_RX_MINUS_90 @ (pos - look)),
    )


def simbody_camera_transform(
    preset: str | ViewPreset,
    lookat_m: Sequence[float] | Array,
    distance_m: float | None = None,
) -> tuple[tuple[Vec3, Vec3, Vec3], Vec3]:
    """Rotation rows and position of the simbody visualizer camera.

    The simbody camera frame looks along its own -Z with +Y up, expressed in
    the (Z-up) ground frame. The rotation is returned as three row tuples so
    ``osim.Transform(osim.Rotation(Mat33), osim.Vec3(position))`` can be built
    by the caller.
    """
    p, look, dist = _resolve(preset, lookat_m, distance_m)
    cols = np.column_stack([p.image_right(), p.image_up(), -p.view_direction()])
    rows = (_tuple3(cols[0]), _tuple3(cols[1]), _tuple3(cols[2]))
    return rows, _tuple3(p.camera_position(look, dist))
