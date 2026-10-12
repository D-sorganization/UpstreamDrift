"""Explicit equipment assumptions for the phase-two matched-delivery study."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace


MAX_CLUB_SPEED_MPS = 100.0


@dataclass(frozen=True)
class MatchedConfig:
    """Static geometry and delivery; SI units except explicitly named degrees.

    Inertia components are about the head CG in the normal/heel/face-up frame.
    Manufacturer references establish loft/lie only. Other head properties are
    research assumptions, not reconstructions of those commercial products.
    Zero curvature radius denotes a flat face; positive radii use a local
    parabolic face approximation. Off-face and nonairborne samples are invalid
    model outcomes and must remain in the population accounting.
    """

    club_id: str = "driver"
    club_speed_mps: float = 45.0
    loft_deg: float = 10.0
    lie_deg: float = 58.5
    shaft_lean_deg: float = 0.0
    attack_angle_deg: float = -0.9
    clubhead_mass_kg: float = 0.200
    inertia_diagonal_kg_m2: tuple[float, float, float] = (0.00025, 0.00040, 0.00045)
    cg_depth_m: float = 0.035
    face_half_width_m: float = 0.050
    face_half_height_m: float = 0.027
    bulge_radius_m: float = 0.300
    roll_radius_m: float = 0.300
    cor: float = 0.83
    friction_coefficient: float = 0.4
    contact_law: str = "incremental_coulomb_energetic"
    steps_per_compression: int = 128
    contact_backend: str = "auto"
    dt_s: float = 0.02
    max_time_s: float = 12.0
    loft_source: str = "https://www.titleist.com/golf-clubs/drivers/gt2"

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            if isinstance(value, (int, float)) and not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if not self.club_id or not self.loft_source:
            raise ValueError("club identity and geometry source are required")
        if not 0 < self.club_speed_mps < MAX_CLUB_SPEED_MPS:
            raise ValueError("club speed must be in (0,100) m/s")
        if not 0 < self.loft_deg < 80 or not 0 < self.lie_deg < 90:
            raise ValueError("static loft must be in (0,80), lie in (0,90) degrees")
        if abs(self.shaft_lean_deg) > 30 or abs(self.attack_angle_deg) >= 30:
            raise ValueError("lean and attack exceed the supported geometric domain")
        for name in ("clubhead_mass_kg", "face_half_width_m", "face_half_height_m"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.cg_depth_m < 0 or min(self.bulge_radius_m, self.roll_radius_m) < 0:
            raise ValueError("CG depth and curvature radii must be nonnegative")
        inertia = self.inertia_diagonal_kg_m2
        if len(inertia) != 3 or any(not math.isfinite(v) or v <= 0 for v in inertia):
            raise ValueError("inertia must have three finite positive components")
        if 2 * max(inertia) > sum(inertia):
            raise ValueError("principal inertias must satisfy triangle inequalities")
        if not 0 <= self.cor <= 1 or self.friction_coefficient < 0:
            raise ValueError("restitution must be in [0,1] and friction nonnegative")
        if self.contact_law not in (
            "incremental_coulomb_energetic",
            "maximum_dissipation_newton",
        ):
            raise ValueError("contact law must name a supported impulse model")
        if (
            isinstance(self.steps_per_compression, bool)
            or not isinstance(self.steps_per_compression, int)
            or not 16 <= self.steps_per_compression <= 4096
        ):
            raise ValueError("compression steps must be an integer in [16,4096]")
        if self.contact_backend not in ("auto", "python_reference", "numba"):
            raise ValueError("contact backend must be auto, python_reference, or numba")
        if not 0 < self.dt_s <= 0.05 or self.max_time_s <= self.dt_s:
            raise ValueError("time step must be in (0,.05] and below flight horizon")


def matched_preset(club_id: str) -> MatchedConfig:
    """Return documented static loft/lie plus explicitly assumed research inputs."""
    driver = MatchedConfig()
    if club_id == "driver":
        return driver
    iron = replace(
        driver,
        club_id=club_id,
        bulge_radius_m=0.0,
        roll_radius_m=0.0,
        face_half_width_m=0.035,
        loft_source="https://www.titleist.com/golf-clubs/irons/720-t100",
    )
    if club_id == "seven_iron":
        return replace(
            iron,
            club_speed_mps=36.0,
            loft_deg=34.0,
            lie_deg=63.0,
            attack_angle_deg=-4.0,
            clubhead_mass_kg=0.272,
            inertia_diagonal_kg_m2=(0.00012, 0.00020, 0.00025),
            cg_depth_m=0.010,
            face_half_height_m=0.023,
        )
    if club_id == "pitching_wedge":
        return replace(
            iron,
            club_speed_mps=32.0,
            loft_deg=46.0,
            lie_deg=64.0,
            attack_angle_deg=-5.0,
            clubhead_mass_kg=0.300,
            inertia_diagonal_kg_m2=(0.00010, 0.00020, 0.00022),
            cg_depth_m=0.008,
            face_half_height_m=0.025,
        )
    raise ValueError(f"unknown matched-delivery club: {club_id}")
