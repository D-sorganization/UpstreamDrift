"""Engine-neutral contract for the spatial two-hand forward-contact study.

The model is intentionally reduced: two finite-mass translational hand
carriages interact with one freely moving rigid club through paired Kelvin--
Voigt interfaces.  A grounded reference driver acts only on the carriages;
there is no direct generalized force or torque applied to the club.  MuJoCo
and Pinocchio consume this same immutable parameter record and independently
evaluate forward dynamics.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class SpatialContactParameters:
    """Declared common model and integration parameters in SI units."""

    hand_mass: float = 0.55
    hand_inertia: float = 8.0e-4
    club_mass: float = 0.32
    club_inertia: tuple[float, float, float] = (1.2e-3, 2.5e-2, 2.5e-2)
    lead_grip_offset: tuple[float, float, float] = (-0.080, 0.015, 0.0)
    trail_grip_offset: tuple[float, float, float] = (0.080, -0.015, 0.0)
    club_initial_position: tuple[float, float, float] = (0.65, 0.0, 1.05)
    contact_stiffness: float = 1800.0
    contact_damping: float = 18.0
    driver_stiffness: float = 420.0
    driver_damping: float = 24.0
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)
    duration: float = 0.240
    time_step: float = 0.00025
    killswitch_time: float = 0.180
    comparison_stride: int = 4

    def __post_init__(self) -> None:
        positive = {
            "hand_mass": self.hand_mass,
            "hand_inertia": self.hand_inertia,
            "club_mass": self.club_mass,
            "contact_stiffness": self.contact_stiffness,
            "contact_damping": self.contact_damping,
            "driver_stiffness": self.driver_stiffness,
            "driver_damping": self.driver_damping,
            "duration": self.duration,
            "time_step": self.time_step,
        }
        for name, value in positive.items():
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        inertia = np.asarray(self.club_inertia, dtype=float)
        if (
            inertia.shape != (3,)
            or np.any(~np.isfinite(inertia))
            or np.any(inertia <= 0)
        ):
            raise ValueError("club_inertia must contain three positive values")
        offsets = np.asarray([self.lead_grip_offset, self.trail_grip_offset])
        if offsets.shape != (2, 3) or np.any(~np.isfinite(offsets)):
            raise ValueError("grip offsets must be two finite 3-vectors")
        if np.linalg.norm(offsets[0] - offsets[1]) <= 1e-9:
            raise ValueError("grip offsets must be spatially separated")
        if not 0.0 < self.killswitch_time < self.duration:
            raise ValueError("killswitch_time must lie inside the simulation")
        if self.comparison_stride < 1:
            raise ValueError("comparison_stride must be at least one")

    def canonical_record(self) -> dict[str, Any]:
        """Return the JSON-compatible immutable model record."""

        record = asdict(self)
        record["model_name"] = "spatial-two-hand-carriage-club-v1"
        record["coordinates"] = {
            "lead_hand": "world translation xyz",
            "trail_hand": "world translation xyz",
            "club": "free rigid body SE(3)",
        }
        record["contact_law"] = "paired Kelvin-Voigt point interfaces"
        record["club_direct_actuation"] = "none"
        return record

    def model_digest(self) -> str:
        """Return a stable SHA-256 digest of the common model record."""

        payload = json.dumps(
            self.canonical_record(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class CanonicalSpatialState:
    """Engine-independent achieved state at one sample."""

    hand_positions: FloatArray
    hand_velocities: FloatArray
    club_position: FloatArray
    club_quaternion_wxyz: FloatArray
    club_linear_velocity: FloatArray
    club_angular_velocity: FloatArray

    def __post_init__(self) -> None:
        shapes = {
            "hand_positions": (2, 3),
            "hand_velocities": (2, 3),
            "club_position": (3,),
            "club_quaternion_wxyz": (4,),
            "club_linear_velocity": (3,),
            "club_angular_velocity": (3,),
        }
        for name, shape in shapes.items():
            value = np.asarray(getattr(self, name), dtype=float)
            if value.shape != shape or np.any(~np.isfinite(value)):
                raise ValueError(f"{name} must have finite shape {shape}")
        quaternion_norm = float(np.linalg.norm(self.club_quaternion_wxyz))
        if not np.isclose(quaternion_norm, 1.0, atol=1.0e-10):
            raise ValueError("club_quaternion_wxyz must be unit length")


def default_spatial_state(params: SpatialContactParameters) -> CanonicalSpatialState:
    """Return the legacy zero-preload initial state in canonical coordinates."""

    center = np.asarray(params.club_initial_position, dtype=float)
    offsets = np.asarray([params.lead_grip_offset, params.trail_grip_offset])
    return CanonicalSpatialState(
        hand_positions=center + offsets,
        hand_velocities=np.zeros((2, 3)),
        club_position=center,
        club_quaternion_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        club_linear_velocity=np.zeros(3),
        club_angular_velocity=np.zeros(3),
    )


def canonical_spatial_state_digest(state: CanonicalSpatialState) -> str:
    """Return a stable digest of one engine-independent initial state."""

    if not isinstance(state, CanonicalSpatialState):
        raise TypeError("state must be a CanonicalSpatialState")
    payload = {
        field: np.asarray(getattr(state, field), dtype=float).tolist()
        for field in (
            "hand_positions",
            "hand_velocities",
            "club_position",
            "club_quaternion_wxyz",
            "club_linear_velocity",
            "club_angular_velocity",
        )
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def rotation_matrix_from_quaternion(quaternion_wxyz: FloatArray) -> FloatArray:
    """Return a world-from-local rotation matrix for a unit quaternion."""

    q = np.asarray(quaternion_wxyz, dtype=float)
    if q.shape != (4,) or np.any(~np.isfinite(q)):
        raise ValueError("quaternion_wxyz must be one finite 4-vector")
    norm = float(np.linalg.norm(q))
    if norm <= 1e-12:
        raise ValueError("quaternion_wxyz must have nonzero norm")
    w, x, y, z = q / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def contact_pair(
    *,
    hand_position: FloatArray,
    hand_velocity: FloatArray,
    club_point_position: FloatArray,
    club_point_velocity: FloatArray,
    stiffness: float,
    damping: float,
) -> tuple[FloatArray, FloatArray, float, float]:
    """Evaluate one paired compliant interface.

    Returns force on club, force on hand, reversible storage-rate contribution,
    and non-positive dissipative power.  The last two terms sum to the total
    power delivered to the two bodies by the interface.
    """

    vectors = [hand_position, hand_velocity, club_point_position, club_point_velocity]
    if any(np.asarray(value).shape != (3,) for value in vectors):
        raise ValueError("contact state values must be 3-vectors")
    if stiffness <= 0.0 or damping < 0.0:
        raise ValueError("contact stiffness must be positive and damping nonnegative")
    displacement = np.asarray(hand_position, dtype=float) - np.asarray(
        club_point_position, dtype=float
    )
    relative_velocity = np.asarray(hand_velocity, dtype=float) - np.asarray(
        club_point_velocity, dtype=float
    )
    force_on_club = stiffness * displacement + damping * relative_velocity
    force_on_hand = -force_on_club
    storage_power = -float(stiffness * displacement @ relative_velocity)
    dissipated_power = -float(damping * relative_velocity @ relative_velocity)
    return force_on_club, force_on_hand, storage_power, dissipated_power


def transport_wrench(
    *, reference: FloatArray, points: FloatArray, forces: FloatArray
) -> FloatArray:
    """Transport point forces to ``reference`` as ``[force, moment]``."""

    reference_array = np.asarray(reference, dtype=float)
    points_array = np.asarray(points, dtype=float)
    forces_array = np.asarray(forces, dtype=float)
    if reference_array.shape != (3,):
        raise ValueError("reference must be one 3-vector")
    if points_array.ndim != 2 or points_array.shape[1] != 3:
        raise ValueError("points must have shape (n, 3)")
    if forces_array.shape != points_array.shape:
        raise ValueError("forces must match the point array shape")
    force = np.sum(forces_array, axis=0)
    moment = np.sum(np.cross(points_array - reference_array, forces_array), axis=0)
    return np.concatenate([force, moment])


def smoothstep5(value: float) -> float:
    """Return a quintic unit-interval step with zero endpoint acceleration."""

    x = float(np.clip(value, 0.0, 1.0))
    return x * x * x * (10.0 + x * (-15.0 + 6.0 * x))


def _rotation_x(angle: float) -> FloatArray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _rotation_y(angle: float) -> FloatArray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _rotation_z(angle: float) -> FloatArray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def driver_targets(time: float, params: SpatialContactParameters) -> FloatArray:
    """Return the two grounded carriage reference positions at ``time``."""

    if not np.isfinite(time):
        raise ValueError("time must be finite")
    phase = smoothstep5(time / params.duration)
    theta = 1.45 * phase
    plane_tilt = 0.16 * np.sin(np.pi * phase)
    roll = 0.11 * np.sin(1.25 * np.pi * phase)
    rotation = _rotation_z(theta) @ _rotation_y(plane_tilt) @ _rotation_x(roll)
    center0 = np.asarray(params.club_initial_position, dtype=float)
    center = center0 + np.array(
        [
            0.075 * (np.cos(theta) - 1.0),
            0.22 * np.sin(theta),
            0.045 * np.sin(np.pi * phase),
        ]
    )
    offsets = np.asarray([params.lead_grip_offset, params.trail_grip_offset])
    targets = center + (rotation @ offsets.T).T
    # A bounded late differential path supplies a falsifiable nonplanar input;
    # it is a declared driver command, not a fitted human hand trajectory.
    differential = 0.006 * np.sin(np.pi * phase) ** 4
    targets[0, 2] += differential
    targets[1, 2] -= differential
    return targets


def driver_target_velocities(
    time: float, params: SpatialContactParameters
) -> FloatArray:
    """Return centered finite-difference target velocities."""

    step = min(1.0e-5, 0.1 * params.time_step)
    lower = max(0.0, time - step)
    upper = min(params.duration, time + step)
    if upper == lower:
        return np.zeros((2, 3))
    return (driver_targets(upper, params) - driver_targets(lower, params)) / (
        upper - lower
    )


__all__ = [
    "CanonicalSpatialState",
    "SpatialContactParameters",
    "canonical_spatial_state_digest",
    "contact_pair",
    "default_spatial_state",
    "driver_target_velocities",
    "driver_targets",
    "rotation_matrix_from_quaternion",
    "smoothstep5",
    "transport_wrench",
]
