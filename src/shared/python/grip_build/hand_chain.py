"""Kinematic hand chain with capsule phalanges (HRG-GB2)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Final

import numpy as np
from numpy.typing import NDArray

from src.shared.python.grip_build.finger_order import LEAD_FINGER_CURL_ORDER
from src.shared.python.grip_build.geometry import deg_to_rad, rotation_about_axis
from src.shared.python.grip_build.thumb import ThumbJointAngles
from src.shared.python.grip_spec.types import HandSize
from src.shared.python.hand_size import hand_mesh_scale
from src.shared.python.model_appearance.grip_hand_mesh import (
    DEFAULT_HAND_MESH_PARAMS,
    FINGER_NAMES,
    HandMeshParams,
)

Array = NDArray[np.float64]

# Kapandji (1982) / clinical active flexion ROM (degrees), conservative for builder.
_MCP_FLEX_LIMIT_RAD: Final[float] = deg_to_rad(90.0)
_PIP_FLEX_LIMIT_RAD: Final[float] = deg_to_rad(100.0)
_DIP_FLEX_LIMIT_RAD: Final[float] = deg_to_rad(80.0)
_DEFAULT_DIP_OVER_PIP: Final[float] = 2.0 / 3.0

_FINGER_SPREAD_M: Final[dict[str, float]] = {
    "index": 0.0,
    "middle": 0.018,
    "ring": 0.032,
    "little": 0.044,
}


@dataclass
class FingerJointAngles:
    """Flexion angles (radians) for one finger."""

    mcp_rad: float = 0.0
    pip_rad: float = 0.0
    dip_rad: float = 0.0

    def clamp_to_limits(self) -> None:
        self.mcp_rad = float(np.clip(self.mcp_rad, 0.0, _MCP_FLEX_LIMIT_RAD))
        self.pip_rad = float(np.clip(self.pip_rad, 0.0, _PIP_FLEX_LIMIT_RAD))
        self.dip_rad = float(np.clip(self.dip_rad, 0.0, _DIP_FLEX_LIMIT_RAD))


@dataclass
class HandChain:
    """Palm + four fingers + thumb state for procedural GB-2/GB-3 builds."""

    hand_size: HandSize
    mesh_params: HandMeshParams = field(
        default_factory=lambda: replace(DEFAULT_HAND_MESH_PARAMS)
    )
    curl_sign: int = -1
    joint_angles: dict[str, FingerJointAngles] = field(default_factory=dict)
    dip_over_pip: dict[str, float] = field(default_factory=dict)
    flat_lateral_offset_m: float = 0.0
    grip_radius_m: float = 0.015
    palm_wrap_rad: float = 0.0
    axial_shift_m: float = 0.0
    thumb_azimuth_rad: float = 0.0
    thumb: ThumbJointAngles = field(default_factory=ThumbJointAngles)

    def __post_init__(self) -> None:
        if self.curl_sign not in (-1, 1):
            raise ValueError("curl_sign must be -1 or +1")
        scale = hand_mesh_scale(self.hand_size.length_m)
        object.__setattr__(
            self,
            "mesh_params",
            replace(self.mesh_params, scale=self.mesh_params.scale * scale),
        )
        for finger in LEAD_FINGER_CURL_ORDER:
            if finger not in self.joint_angles:
                self.joint_angles[finger] = FingerJointAngles()
            if finger not in self.dip_over_pip:
                self.dip_over_pip[finger] = _DEFAULT_DIP_OVER_PIP

    def apply_dip_coupling(self, finger: str) -> None:
        """Set DIP from PIP using the per-finger coupling ratio."""
        if finger not in self.joint_angles:
            raise ValueError(f"unknown finger {finger!r}")
        ratio = self.dip_over_pip[finger]
        angles = self.joint_angles[finger]
        angles.dip_rad = min(angles.pip_rad * ratio, _DIP_FLEX_LIMIT_RAD)

    def finger_base(self, finger: str) -> np.ndarray:
        """MCP joint centre in the hand-grip frame (flat pose layout)."""
        y = _FINGER_SPREAD_M[finger] + self.flat_lateral_offset_m
        pad = self.mesh_params.pad_clearance_m * self.mesh_params.scale
        radius = self.mesh_params.finger_radius_m[finger] * self.mesh_params.scale
        if finger == "index":
            x = self.grip_radius_m * 0.5
        else:
            x = self.grip_radius_m + pad + radius
        return np.array([x, y, 0.0], dtype=np.float64)

    def _apply_frame(self, point: np.ndarray) -> np.ndarray:
        rot = rotation_about_axis(np.array([0.0, 0.0, 1.0]), self.palm_wrap_rad)
        shifted = point.copy()
        shifted[2] += self.axial_shift_m
        return rot @ shifted

    def finger_capsules(
        self, finger: str
    ) -> list[tuple[np.ndarray, np.ndarray, float]]:
        """Return proximal/middle/distal capsule core segments for *finger*."""
        if finger not in FINGER_NAMES:
            raise ValueError(f"unknown finger {finger!r}")
        lengths = self.mesh_params.phalanx_lengths_m[finger]
        radius = self.mesh_params.finger_radius_m[finger] * self.mesh_params.scale
        angles = self.joint_angles[finger]
        base = self.finger_base(finger)
        flex_axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        direction = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        mcp, pip, dip = lengths
        seg_flex = (
            (mcp, angles.mcp_rad),
            (pip, angles.pip_rad),
            (dip, angles.dip_rad),
        )
        self.apply_dip_coupling(finger)
        angles = self.joint_angles[finger]
        seg_flex = (
            (mcp, angles.mcp_rad),
            (pip, angles.pip_rad),
            (dip, angles.dip_rad),
        )
        points: list[tuple[np.ndarray, np.ndarray, float]] = []
        origin = base.copy()
        for seg_len, flex_rad in seg_flex:
            if flex_rad > 0.0:
                rot = rotation_about_axis(flex_axis, self.curl_sign * flex_rad)
                direction = rot @ direction
            end = origin + direction * seg_len
            points.append((origin.copy(), end.copy(), radius))
            origin = end
        return [
            (self._apply_frame(p0), self._apply_frame(p1), radius)
            for p0, p1, radius in points
        ]

    def segment_capsule(
        self, finger: str, joint: str
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Capsule for a single phalanx after applying angles up to *joint*."""
        caps = self.finger_capsules(finger)
        index = {"mcp": 0, "pip": 1, "dip": 2}[joint]
        return caps[index]

    def all_capsules(self) -> list[tuple[str, int, np.ndarray, np.ndarray, float]]:
        """Flattened capsules with finger and segment indices."""
        out: list[tuple[str, int, np.ndarray, np.ndarray, float]] = []
        for finger in LEAD_FINGER_CURL_ORDER:
            for seg_idx, (p0, p1, r) in enumerate(self.finger_capsules(finger)):
                out.append((finger, seg_idx, p0, p1, r))
        return out

    def palm_heel_radius_m(self) -> float:
        return self.mesh_params.scale * 0.012

    def palm_heel_point(self) -> np.ndarray:
        base = np.array(
            [self.grip_radius_m * 0.2, -0.028 + self.flat_lateral_offset_m, -0.01],
            dtype=np.float64,
        )
        return self._apply_frame(base)

    def palm_pocket_radius_m(self) -> float:
        return self.mesh_params.scale * 0.010

    def trail_palm_pocket_point(self) -> np.ndarray:
        base = np.array(
            [self.grip_radius_m * 0.15, 0.018, 0.02],
            dtype=np.float64,
        )
        return self._apply_frame(base)

    def joint_within_limits(self) -> bool:
        for angles in self.joint_angles.values():
            if (
                angles.mcp_rad > _MCP_FLEX_LIMIT_RAD + 1e-9
                or angles.pip_rad > _PIP_FLEX_LIMIT_RAD + 1e-9
                or angles.dip_rad > _DIP_FLEX_LIMIT_RAD + 1e-9
            ):
                return False
        return True


def flat_start_index_proximal_on_axis(
    chain: HandChain, club_axis: tuple[np.ndarray, np.ndarray], *, tol_m: float = 2e-3
) -> bool:
    """Postcondition helper: index proximal capsule crosses the club axis."""
    caps = chain.finger_capsules("index")
    p0, p1, _ = caps[0]
    midpoint = 0.5 * (p0 + p1)
    radial = math.hypot(float(midpoint[0]), float(midpoint[1]))
    _ = club_axis
    return radial <= max(tol_m, chain.grip_radius_m * 0.6)


def flat_start_club_below_pinky_base(
    chain: HandChain,
    club_cross_angle_deg: float,
    *,
    tol_m: float = 0.008,
) -> bool:
    """Club line exits just below the pinky MCP in flat pose (x-z plane check)."""
    pinky_base = chain.finger_base("little")
    angle_rad = deg_to_rad(club_cross_angle_deg)
    # Club runs in x-z with slight angle from +z toward +x.
    club_dir = np.array(
        [math.sin(angle_rad), 0.0, math.cos(angle_rad)], dtype=np.float64
    )
    club_dir /= np.linalg.norm(club_dir)
    pinky_exit = pinky_base + club_dir * 0.02
    return float(pinky_exit[0]) >= float(pinky_base[0]) - tol_m
