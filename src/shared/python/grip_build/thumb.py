"""Thumb kinematics for HRG-GB3 (engine-free)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import numpy as np

from src.shared.python.grip_build.geometry import rotation_about_axis
from src.shared.python.model_appearance.grip_hand_mesh import HandMeshParams

if TYPE_CHECKING:
    from src.shared.python.grip_spec.types import HandGripSpec

_IP_MAX_FLEX_RAD: Final[float] = math.radians(35.0)
_THUMB_FLEX_AXIS = np.array([0.0, 1.0, 0.0], dtype=np.float64)


@dataclass
class ThumbJointAngles:
    """Flexion angles (radians) for the two-segment thumb chain."""

    cmc_rad: float = 0.0
    mcp_rad: float = 0.0
    ip_rad: float = 0.0


def thumb_ip_cap_rad(hand_spec: HandGripSpec) -> float:
    """Maximum IP flex permitted by ``thumb_straightness`` (1 = straight)."""
    straight = float(np.clip(hand_spec.thumb_straightness, 0.0, 1.0))
    return (1.0 - straight) * _IP_MAX_FLEX_RAD


def thumb_radial_unit(azimuth_rad: float) -> np.ndarray:
    return np.array(
        [math.cos(azimuth_rad), math.sin(azimuth_rad), 0.0], dtype=np.float64
    )


def thumb_capsules(
    mesh: HandMeshParams,
    azimuth_rad: float,
    angles: ThumbJointAngles,
    grip_radius_m: float,
    *,
    radial_unit: np.ndarray | None = None,
    palm_wrap_rad: float = 0.0,
    axial_shift_m: float = 0.0,
) -> list[tuple[np.ndarray, np.ndarray, float]]:
    """Proximal and distal thumb capsules in the grip build frame."""
    scale = mesh.scale
    r = scale * mesh.thumb_radius_m
    pad = mesh.pad_clearance_m * scale
    radial = thumb_radial_unit(azimuth_rad) if radial_unit is None else radial_unit
    radial = np.asarray(radial, dtype=np.float64)
    radial[2] = 0.0
    norm = float(np.linalg.norm(radial))
    if norm < 1e-12:
        raise ValueError("radial_unit must be non-zero")
    radial = radial / norm
    centre = (grip_radius_m + pad + r) * radial
    z0, z1, z2 = (scale * s for s in mesh.thumb_stations_m)
    offsets = (z0, z1, z2)
    direction = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    origin = centre + np.array([0.0, 0.0, offsets[0] + axial_shift_m])
    segments: list[tuple[float, float]] = [
        (offsets[1] - offsets[0], angles.cmc_rad + angles.mcp_rad),
        (offsets[2] - offsets[1], angles.ip_rad),
    ]
    capsules: list[tuple[np.ndarray, np.ndarray, float]] = []
    for seg_len, flex_rad in segments:
        if flex_rad != 0.0:
            direction = rotation_about_axis(_THUMB_FLEX_AXIS, flex_rad) @ direction
        end = origin + direction * seg_len
        capsules.append((origin.copy(), end.copy(), r))
        origin = end
    wrap = rotation_about_axis(np.array([0.0, 0.0, 1.0]), palm_wrap_rad)
    return [(wrap @ p0, wrap @ p1, radius) for p0, p1, radius in capsules]


def thumb_radial_direction(
    capsules: list[tuple[np.ndarray, np.ndarray, float]],
) -> np.ndarray:
    """Unit vector of the distal thumb segment projected into the x-y plane."""
    if not capsules:
        raise ValueError("thumb capsules missing")
    p0, p1, _ = capsules[-1]
    vec = p1 - p0
    vec[2] = 0.0
    norm = float(np.linalg.norm(vec))
    if norm < 1e-9:
        vec = p0.copy()
        vec[2] = 0.0
        norm = float(np.linalg.norm(vec))
    if norm < 1e-9:
        raise ValueError("degenerate thumb direction")
    return vec / norm
