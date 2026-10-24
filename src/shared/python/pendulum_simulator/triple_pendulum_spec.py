"""Canonical planar three-link pendulum parameters shared by engine adapters."""

from __future__ import annotations

import math
from dataclasses import dataclass

from src.shared.python.core.physics_constants import GRAVITY_M_S2


@dataclass(frozen=True)
class TriplePendulumSegmentSpec:
    length_m: float
    mass_kg: float
    center_of_mass_ratio: float
    inertia_about_com: float

    def __post_init__(self) -> None:
        for name in ("length_m", "mass_kg", "inertia_about_com"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if (
            not math.isfinite(self.center_of_mass_ratio)
            or not 0.0 <= self.center_of_mass_ratio <= 1.0
        ):
            raise ValueError("center_of_mass_ratio must be in [0, 1]")


@dataclass(frozen=True)
class TriplePendulumSpec:
    segments: tuple[
        TriplePendulumSegmentSpec,
        TriplePendulumSegmentSpec,
        TriplePendulumSegmentSpec,
    ]
    damping_nm_s_rad: tuple[float, float, float]
    gravity_m_s2: float = float(GRAVITY_M_S2)
    gravity_plane: str = "xz"
    coordinate_convention: str = "relative"
    inertia_model: str = "distributed"

    def __post_init__(self) -> None:
        if len(self.segments) != 3:
            raise ValueError("exactly three segments are required")
        if len(self.damping_nm_s_rad) != 3:
            raise ValueError("exactly three damping coefficients are required")
        if self.gravity_plane != "xz":
            raise ValueError(f"unknown gravity plane: {self.gravity_plane!r}")
        if self.coordinate_convention != "relative":
            raise ValueError(
                f"unknown coordinate convention: {self.coordinate_convention!r}"
            )
        if self.inertia_model not in {"distributed", "point_mass"}:
            raise ValueError(f"unknown inertia model: {self.inertia_model!r}")
        if not math.isfinite(self.gravity_m_s2) or self.gravity_m_s2 <= 0:
            raise ValueError("gravity_m_s2 must be finite and positive")
        if any(
            not math.isfinite(value) or value < 0 for value in self.damping_nm_s_rad
        ):
            raise ValueError("damping coefficients must be finite and non-negative")


TRIPLE_PENDULUM_SPEC = TriplePendulumSpec(
    segments=(
        TriplePendulumSegmentSpec(0.75, 7.5, 0.45, (1.0 / 12.0) * 7.5 * 0.75**2),
        TriplePendulumSegmentSpec(0.6, 1.8, 0.46, (1.0 / 12.0) * 1.8 * 0.6**2),
        TriplePendulumSegmentSpec(1.0, 0.35, 0.7557142857142858, 0.04034857142857143),
    ),
    damping_nm_s_rad=(0.35, 0.3, 0.25),
)

# Explicit point-mass variant for the legacy simulator's tip-mass equations.
POINT_MASS_TRIPLE_PENDULUM_SPEC = TriplePendulumSpec(
    segments=tuple(
        TriplePendulumSegmentSpec(
            segment.length_m,
            segment.mass_kg,
            1.0,
            segment.inertia_about_com,
        )
        for segment in TRIPLE_PENDULUM_SPEC.segments
    ),
    damping_nm_s_rad=TRIPLE_PENDULUM_SPEC.damping_nm_s_rad,
    inertia_model="point_mass",
)

UNDAMPED_POINT_MASS_TRIPLE_PENDULUM_SPEC = TriplePendulumSpec(
    segments=POINT_MASS_TRIPLE_PENDULUM_SPEC.segments,
    damping_nm_s_rad=(0.0, 0.0, 0.0),
    inertia_model="point_mass",
)
