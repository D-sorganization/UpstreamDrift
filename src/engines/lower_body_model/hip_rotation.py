"""Inclined-plane hip rotation target profiles for the lower-body model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HipRotationSample:
    """A deterministic point on the requested hip rotation target."""

    time_sec: float
    rotation_deg: float
    plane_point: np.ndarray


@dataclass(frozen=True)
class InclinedPlaneHipRotationTarget:
    """Two-phase golf hip rotation target viewed from above.

    Models the rotation of the pelvis around an axis that is tilted forward
    by ``incline_degrees`` (the golfer's spine angle). Phase one is a
    clockwise backswing to ``-backswing_degrees``; phase two is a
    counterclockwise downswing of ``counterclockwise_degrees`` from the
    reversal point, both in equal halves of ``duration_sec``.

    A lateral shift profile (``lateral_shift_m``) is layered on top:
    during the backswing the pelvis stays centered, and during the
    downswing it smoothly translates by up to ``lateral_shift_m`` metres
    in the lateral (+Y) direction via a smoothstep ramp — mirroring the
    golfer's weight shift to the front foot.
    """

    duration_sec: float
    backswing_degrees: float = 45.0
    counterclockwise_degrees: float = 90.0
    incline_degrees: float = 12.0
    sample_count: int = 181
    lateral_shift_m: float = 0.0

    def __post_init__(self) -> None:
        if self.duration_sec <= 0.0:
            raise ValueError(f"duration_sec must be positive, got {self.duration_sec}")
        if not 0.0 < self.backswing_degrees <= 90.0:
            raise ValueError(
                f"backswing_degrees must be in (0, 90], got {self.backswing_degrees}"
            )
        if not 0.0 < self.counterclockwise_degrees <= 180.0:
            raise ValueError(
                "counterclockwise_degrees must be in (0, 180], got "
                f"{self.counterclockwise_degrees}"
            )
        if not -60.0 <= self.incline_degrees <= 60.0:
            raise ValueError(
                f"incline_degrees must be in [-60, 60], got {self.incline_degrees}"
            )
        if self.sample_count < 2:
            raise ValueError(
                f"sample_count must be at least 2, got {self.sample_count}"
            )
        if not -0.5 <= self.lateral_shift_m <= 0.5:
            raise ValueError(
                f"lateral_shift_m must be in [-0.5, 0.5] metres, "
                f"got {self.lateral_shift_m}"
            )

    @property
    def reversal_time_sec(self) -> float:
        """Time where the clockwise phase reverses."""
        return self.duration_sec / 2.0

    @property
    def finish_rotation_deg(self) -> float:
        """Final rotation angle in degrees."""
        return -self.backswing_degrees + self.counterclockwise_degrees

    def rotation_degrees_at(self, time_sec: float) -> float:
        """Return the target rotation in degrees at ``time_sec``."""
        if time_sec < 0.0:
            raise ValueError(f"time_sec must be non-negative, got {time_sec}")
        clamped_time = min(time_sec, self.duration_sec)

        if clamped_time <= self.reversal_time_sec:
            phase = clamped_time / self.reversal_time_sec
            return -self.backswing_degrees * phase

        phase = (clamped_time - self.reversal_time_sec) / self.reversal_time_sec
        return -self.backswing_degrees + self.counterclockwise_degrees * phase

    def plane_point_at(self, time_sec: float) -> np.ndarray:
        """Return a unit-radius point on the inclined target plane."""
        rotation_rad = np.radians(self.rotation_degrees_at(time_sec))
        incline_rad = np.radians(self.incline_degrees)
        return np.array(
            [
                np.cos(rotation_rad),
                np.sin(rotation_rad) * np.cos(incline_rad),
                np.sin(rotation_rad) * np.sin(incline_rad),
            ],
            dtype=float,
        )

    def lateral_shift_at(self, time_sec: float) -> float:
        """Return the lateral (+Y) shift in metres at ``time_sec``.

        Zero during the backswing; during the downswing a smoothstep
        ramp ``s(p) = 3p^2 - 2p^3`` from 0 to ``lateral_shift_m``.
        """
        if time_sec < 0.0:
            raise ValueError(f"time_sec must be non-negative, got {time_sec}")
        clamped_time = min(time_sec, self.duration_sec)
        if clamped_time <= self.reversal_time_sec:
            return 0.0
        phase = (clamped_time - self.reversal_time_sec) / self.reversal_time_sec
        smoothstep = 3.0 * phase * phase - 2.0 * phase * phase * phase
        return float(self.lateral_shift_m * smoothstep)

    def target_quaternion_at(self, time_sec: float) -> np.ndarray:
        """Return the target pelvis quaternion (w, x, y, z) at ``time_sec``.

        The rotation axis is the world-Z axis tilted forward by
        ``incline_degrees`` about the +Y (lateral) axis; the rotation
        amount is ``rotation_degrees_at(time_sec)``.
        """
        if time_sec < 0.0:
            raise ValueError(f"time_sec must be non-negative, got {time_sec}")

        incline_rad = np.radians(self.incline_degrees)
        axis = np.array(
            [
                np.sin(incline_rad),  # forward (X) component from forward lean
                0.0,
                np.cos(incline_rad),  # remaining vertical (Z) component
            ],
            dtype=float,
        )
        axis /= np.linalg.norm(axis)

        angle_rad = np.radians(self.rotation_degrees_at(time_sec))
        half = 0.5 * angle_rad
        s = np.sin(half)
        return np.array(
            [np.cos(half), axis[0] * s, axis[1] * s, axis[2] * s], dtype=float
        )

    def sample(self) -> list[HipRotationSample]:
        """Sample the full target trajectory deterministically."""
        times = np.linspace(0.0, self.duration_sec, self.sample_count)
        return [
            HipRotationSample(
                time_sec=float(time_sec),
                rotation_deg=self.rotation_degrees_at(float(time_sec)),
                plane_point=self.plane_point_at(float(time_sec)),
            )
            for time_sec in times
        ]
