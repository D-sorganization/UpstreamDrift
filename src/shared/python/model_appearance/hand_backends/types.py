"""Shared types for grip hand visual backends (ADR-0057 A2)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from src.shared.python.model_appearance.geometry import Mesh
from src.shared.python.model_appearance.hand_backends.joint_angles import (
    HandJointAngles,
    HandSide,
)

HAND_SIDES: tuple[HandSide, ...] = ("lead", "trail")


@dataclass(frozen=True)
class HandVisualResult:
    """One visual hand mesh in the hand-grip frame.

    ``mesh``, ``skin_rgb`` and ``grip_radius_m`` are what every renderer
    needs. The optional fields let a caller audit the pose; ``contact_gaps_m``
    are visual gaps only, since contact stays on the capsule pads.
    ``frame_focus`` marks the vertices a renderer frames its camera on;
    ``None`` means all of them.
    """

    backend: str
    side: HandSide
    mesh: Mesh
    skin_rgb: tuple[float, float, float]
    grip_radius_m: float
    vertex_normals: NDArray[np.float64] | None = None
    joint_angles: HandJointAngles | None = None
    contact_gaps_m: dict[str, float] = field(default_factory=dict)
    frame_focus: NDArray[np.bool_] | None = None

    def __post_init__(self) -> None:
        if self.side not in HAND_SIDES:
            raise ValueError(f"side must be one of {HAND_SIDES}, got {self.side!r}")
        if len(self.skin_rgb) != 3 or not all(
            math.isfinite(c) and 0.0 <= c <= 1.0 for c in self.skin_rgb
        ):
            raise ValueError(
                f"skin_rgb must be three numbers in [0, 1], got {self.skin_rgb!r}"
            )
        if not (math.isfinite(self.grip_radius_m) and self.grip_radius_m > 0.0):
            raise ValueError("grip_radius_m must be positive and finite")


__all__ = ["HAND_SIDES", "HandJointAngles", "HandSide", "HandVisualResult"]
