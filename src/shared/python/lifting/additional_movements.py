"""Specifications for additional weightlifting movements (#11750, LIFT-10).

Defines five additional barbell movements extending the biomechanics parity
standard exercises:
- split_jerk: explosive drive with asymmetric split-stance catch
- push_press: leg-assisted overhead press with dip-drive transition
- overhead_press: strict overhead press without lower-body drive
- front_squat: anterior deltoid bar rack with deep bilateral squat
- lunge: asymmetric split-stance unilateral lower-body movement
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from .lift_spec import LiftGrip, LiftPhase, ViaPose

ADDITIONAL_LIFTS: Final[tuple[str, ...]] = (
    "split_jerk",
    "push_press",
    "overhead_press",
    "front_squat",
    "lunge",
)


@dataclass(frozen=True)
class StanceGeometry:
    """Stance geometry including foot width, toe out, and split stance placement."""

    foot_width_m: float
    toe_out_deg: float
    is_split: bool = False
    split_length_m: float = 0.0
    lead_foot: str | None = None
    trail_foot: str | None = None
    lead_foot_position_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    trail_foot_position_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    lead_toe_out_deg: float = 0.0
    trail_toe_out_deg: float = 0.0

    def foot_positions(self) -> dict[str, tuple[float, float, float]]:
        """Return canonical foot placements (x forward, y left, z up)."""
        if not self.is_split:
            half_w = self.foot_width_m / 2.0
            return {
                "l": (0.0, half_w, 0.0),
                "r": (0.0, -half_w, 0.0),
            }
        lead = self.lead_foot or "l"
        trail = self.trail_foot or "r"
        return {
            lead: self.lead_foot_position_m,
            trail: self.trail_foot_position_m,
        }


@dataclass(frozen=True)
class BarbellLoad:
    """Barbell plate and sleeve configuration."""

    plate_diameter_m: float = 0.45
    plates_per_side_kg: float = 50.0
    bar_mass_kg: float = 20.0

    @property
    def total_mass_kg(self) -> float:
        """Total mass of barbell with plates."""
        return self.bar_mass_kg + 2.0 * self.plates_per_side_kg


@dataclass(frozen=True)
class AdditionalMovementSpec:
    """Full specification of an additional movement."""

    name: str
    phase_count: int
    root_joint: str
    phases: tuple[LiftPhase, ...]
    grip: LiftGrip
    stance: StanceGeometry
    load: BarbellLoad
    via_poses: dict[str, ViaPose]

    def __post_init__(self) -> None:
        if self.phase_count != len(self.phases):
            raise ValueError(
                f"{self.name}: phase_count {self.phase_count} != {len(self.phases)}"
            )
        fractions = [p.t_frac for p in self.phases]
        if fractions[0] != 0.0 or fractions[-1] >= 1.0:
            raise ValueError(f"{self.name}: phase fractions must span [0, 1)")
        for a, b in zip(fractions, fractions[1:], strict=False):
            if b <= a:
                raise ValueError(f"{self.name}: phase fractions must strictly increase")

    def phase_names(self) -> tuple[str, ...]:
        """Ordered phase names of the movement."""
        return tuple(p.name for p in self.phases)

    def via_pose(self, name: str) -> ViaPose:
        """Retrieve named via-pose."""
        if name not in self.via_poses:
            raise ValueError(f"{self.name} has no via-pose {name!r}")
        return self.via_poses[name]


def _symmetric_coords(base: dict[str, float]) -> dict[str, float]:
    out: dict[str, float] = {}
    for k, v in base.items():
        if "{s}" in k:
            out[k.format(s="l")] = v
            out[k.format(s="r")] = v
        else:
            out[k] = v
    return out


def _make_split_jerk() -> AdditionalMovementSpec:
    phases = (
        LiftPhase("dip", "Short vertical dip through hips and knees.", 0.0),
        LiftPhase("drive", "Explosive triple extension launching bar.", 0.15),
        LiftPhase("split_drop", "Lifter drops into asymmetric split stance.", 0.35),
        LiftPhase(
            "overhead_catch", "Bar caught overhead with locked arms in split.", 0.55
        ),
        LiftPhase("recovery_front", "Lead foot steps back halfway to recover.", 0.75),
        LiftPhase("lockout", "Feet aligned, standing upright with bar locked.", 0.90),
    )
    grip = LiftGrip("palm_centre", "pronated", 0.60, (0.45, 0.80))
    stance = StanceGeometry(
        foot_width_m=0.35,
        toe_out_deg=15.0,
        is_split=True,
        split_length_m=0.75,
        lead_foot="l",
        trail_foot="r",
        lead_foot_position_m=(0.40, 0.175, 0.0),
        trail_foot_position_m=(-0.35, -0.175, 0.0),
        lead_toe_out_deg=5.0,
        trail_toe_out_deg=10.0,
    )
    dip_coords = _symmetric_coords(
        {
            "lumbar_flex": 0.05,
            "neck_flex": 0.0,
            "shoulder_{s}_flex": 1.15,
            "shoulder_{s}_adduct": -0.22,
            "elbow_{s}_flex": 2.25,
            "wrist_{s}_flex": 0.05,
            "hip_{s}_flex": 0.30,
            "knee_{s}_flex": -0.45,
            "ankle_{s}_flex": 0.18,
        }
    )
    catch_coords = {
        "lumbar_flex": 0.02,
        "neck_flex": 0.0,
        "shoulder_l_flex": 2.98,
        "shoulder_r_flex": 2.98,
        "shoulder_l_adduct": -0.22,
        "shoulder_r_adduct": -0.22,
        "elbow_l_flex": 0.0,
        "elbow_r_flex": 0.0,
        "wrist_l_flex": 0.0,
        "wrist_r_flex": 0.0,
        "hip_l_flex": 0.85,
        "hip_r_flex": -0.28,
        "hip_l_adduct": -0.05,
        "hip_r_adduct": -0.05,
        "knee_l_flex": -1.18,
        "knee_r_flex": -0.78,
        "ankle_l_flex": 0.22,
        "ankle_r_flex": -0.08,
    }
    lockout_coords = _symmetric_coords(
        {
            "lumbar_flex": 0.02,
            "neck_flex": 0.0,
            "shoulder_{s}_flex": 2.98,
            "shoulder_{s}_adduct": -0.22,
            "elbow_{s}_flex": 0.0,
            "wrist_{s}_flex": 0.0,
            "hip_{s}_flex": 0.05,
            "knee_{s}_flex": -0.05,
            "ankle_{s}_flex": 0.02,
        }
    )
    via_poses = {
        "start": ViaPose(
            "start",
            "dip",
            (-0.05, 0.0, 0.94),
            (0.0, 0.05, 0.0),
            (0.06, 0.0, 1.42),
            dip_coords,
        ),
        "catch": ViaPose(
            "catch",
            "overhead_catch",
            (-0.05, 0.0, 0.78),
            (0.0, 0.02, 0.0),
            (0.02, 0.0, 2.05),
            catch_coords,
        ),
        "lockout": ViaPose(
            "lockout",
            "lockout",
            (-0.05, 0.0, 1.01),
            (0.0, 0.0, 0.0),
            (0.02, 0.0, 2.16),
            lockout_coords,
        ),
    }
    return AdditionalMovementSpec(
        "split_jerk", 6, "free", phases, grip, stance, BarbellLoad(), via_poses
    )


def _make_push_press() -> AdditionalMovementSpec:
    phases = (
        LiftPhase("setup", "Upright stance with bar resting in front rack.", 0.0),
        LiftPhase("dip", "Controlled vertical dip flexing knees and hips.", 0.15),
        LiftPhase("drive", "Rapid vertical extension propelling bar.", 0.35),
        LiftPhase("press_transition", "Arms press through as bar clears head.", 0.55),
        LiftPhase("lockout", "Arms fully extended overhead with hips locked.", 0.85),
    )
    grip = LiftGrip("palm_centre", "pronated", 0.55, (0.45, 0.75))
    stance = StanceGeometry(foot_width_m=0.32, toe_out_deg=15.0, is_split=False)
    setup_coords = _symmetric_coords(
        {
            "lumbar_flex": 0.04,
            "neck_flex": 0.0,
            "shoulder_{s}_flex": 1.15,
            "shoulder_{s}_adduct": -0.20,
            "elbow_{s}_flex": 2.25,
            "wrist_{s}_flex": 0.04,
            "hip_{s}_flex": 0.05,
            "knee_{s}_flex": -0.05,
            "ankle_{s}_flex": 0.02,
        }
    )
    dip_coords = _symmetric_coords(
        {
            "lumbar_flex": 0.04,
            "neck_flex": 0.0,
            "shoulder_{s}_flex": 1.15,
            "shoulder_{s}_adduct": -0.20,
            "elbow_{s}_flex": 2.25,
            "wrist_{s}_flex": 0.04,
            "hip_{s}_flex": 0.28,
            "knee_{s}_flex": -0.38,
            "ankle_{s}_flex": 0.14,
        }
    )
    lockout_coords = _symmetric_coords(
        {
            "lumbar_flex": 0.02,
            "neck_flex": 0.0,
            "shoulder_{s}_flex": 2.98,
            "shoulder_{s}_adduct": -0.20,
            "elbow_{s}_flex": 0.0,
            "wrist_{s}_flex": 0.0,
            "hip_{s}_flex": 0.05,
            "knee_{s}_flex": -0.05,
            "ankle_{s}_flex": 0.02,
        }
    )
    via_poses = {
        "start": ViaPose(
            "start",
            "setup",
            (0.0, 0.0, 1.01),
            (0.0, 0.02, 0.0),
            (0.06, 0.0, 1.48),
            setup_coords,
        ),
        "catch": ViaPose(
            "catch",
            "dip",
            (0.0, 0.0, 0.94),
            (0.0, 0.04, 0.0),
            (0.06, 0.0, 1.41),
            dip_coords,
        ),
        "lockout": ViaPose(
            "lockout",
            "lockout",
            (0.0, 0.0, 1.01),
            (0.0, 0.0, 0.0),
            (0.02, 0.0, 2.15),
            lockout_coords,
        ),
    }
    return AdditionalMovementSpec(
        "push_press", 5, "free", phases, grip, stance, BarbellLoad(), via_poses
    )


def _make_overhead_press() -> AdditionalMovementSpec:
    phases = (
        LiftPhase("setup", "Bar on anterior deltoids, core braced, legs locked.", 0.0),
        LiftPhase("press", "Strict vertical arm press without leg assistance.", 0.20),
        LiftPhase("head_pass", "Bar clears head, torso moves under bar.", 0.50),
        LiftPhase("lockout", "Bar locked overhead in line with spine.", 0.80),
    )
    grip = LiftGrip("palm_centre", "pronated", 0.50, (0.40, 0.70))
    stance = StanceGeometry(foot_width_m=0.28, toe_out_deg=10.0, is_split=False)
    setup_coords = _symmetric_coords(
        {
            "lumbar_flex": 0.02,
            "neck_flex": 0.0,
            "shoulder_{s}_flex": 1.10,
            "shoulder_{s}_adduct": -0.18,
            "elbow_{s}_flex": 2.25,
            "wrist_{s}_flex": 0.02,
            "hip_{s}_flex": 0.05,
            "knee_{s}_flex": -0.05,
            "ankle_{s}_flex": 0.02,
        }
    )
    pass_coords = _symmetric_coords(
        {
            "lumbar_flex": 0.02,
            "neck_flex": -0.05,
            "shoulder_{s}_flex": 2.15,
            "shoulder_{s}_adduct": -0.18,
            "elbow_{s}_flex": 0.85,
            "wrist_{s}_flex": 0.0,
            "hip_{s}_flex": 0.05,
            "knee_{s}_flex": -0.05,
            "ankle_{s}_flex": 0.02,
        }
    )
    lockout_coords = _symmetric_coords(
        {
            "lumbar_flex": 0.02,
            "neck_flex": 0.0,
            "shoulder_{s}_flex": 2.98,
            "shoulder_{s}_adduct": -0.18,
            "elbow_{s}_flex": 0.0,
            "wrist_{s}_flex": 0.0,
            "hip_{s}_flex": 0.05,
            "knee_{s}_flex": -0.05,
            "ankle_{s}_flex": 0.02,
        }
    )
    via_poses = {
        "start": ViaPose(
            "start",
            "setup",
            (0.0, 0.0, 1.01),
            (0.0, 0.01, 0.0),
            (0.06, 0.0, 1.48),
            setup_coords,
        ),
        "catch": ViaPose(
            "catch",
            "head_pass",
            (0.0, 0.0, 1.01),
            (0.0, 0.0, 0.0),
            (0.03, 0.0, 1.82),
            pass_coords,
        ),
        "lockout": ViaPose(
            "lockout",
            "lockout",
            (0.0, 0.0, 1.01),
            (0.0, 0.0, 0.0),
            (0.02, 0.0, 2.15),
            lockout_coords,
        ),
    }
    return AdditionalMovementSpec(
        "overhead_press", 4, "free", phases, grip, stance, BarbellLoad(), via_poses
    )


def _make_front_squat() -> AdditionalMovementSpec:
    phases = (
        LiftPhase("standing", "Upright with bar in front rack, elbows high.", 0.0),
        LiftPhase("descent", "Hips and knees flex with vertical torso.", 0.15),
        LiftPhase("bottom", "Lowest depth, hip crease below knee.", 0.45),
        LiftPhase("ascent", "Ascent maintaining upright torso posture.", 0.50),
        LiftPhase(
            "lockout", "Upright standing position with hips/knees extended.", 0.90
        ),
    )
    grip = LiftGrip("palm_centre", "clean", 0.55, (0.45, 0.75))
    stance = StanceGeometry(foot_width_m=0.42, toe_out_deg=25.0, is_split=False)
    standing_coords = _symmetric_coords(
        {
            "lumbar_flex": 0.03,
            "neck_flex": 0.0,
            "shoulder_{s}_flex": 1.25,
            "shoulder_{s}_adduct": -0.22,
            "elbow_{s}_flex": 2.45,
            "wrist_{s}_flex": 0.03,
            "hip_{s}_flex": 0.05,
            "knee_{s}_flex": -0.05,
            "ankle_{s}_flex": 0.03,
        }
    )
    bottom_coords = _symmetric_coords(
        {
            "lumbar_flex": 0.12,
            "neck_flex": 0.0,
            "shoulder_{s}_flex": 1.30,
            "shoulder_{s}_adduct": -0.25,
            "elbow_{s}_flex": 2.45,
            "wrist_{s}_flex": 0.03,
            "hip_{s}_flex": 1.95,
            "hip_{s}_adduct": -0.40,
            "knee_{s}_flex": -1.90,
            "ankle_{s}_flex": 0.42,
        }
    )
    via_poses = {
        "start": ViaPose(
            "start",
            "standing",
            (0.0, 0.0, 1.01),
            (0.0, 0.04, 0.0),
            (0.06, 0.0, 1.48),
            standing_coords,
        ),
        "catch": ViaPose(
            "catch",
            "bottom",
            (-0.18, 0.0, 0.60),
            (0.0, 0.22, 0.0),
            (0.04, 0.0, 1.06),
            bottom_coords,
        ),
        "lockout": ViaPose(
            "lockout",
            "lockout",
            (0.0, 0.0, 1.01),
            (0.0, 0.04, 0.0),
            (0.06, 0.0, 1.48),
            standing_coords,
        ),
    }
    return AdditionalMovementSpec(
        "front_squat", 5, "free", phases, grip, stance, BarbellLoad(), via_poses
    )


def _make_lunge() -> AdditionalMovementSpec:
    phases = (
        LiftPhase("standing", "Upright stance with bar on back.", 0.0),
        LiftPhase("step_descent", "Lead foot steps into split, hips descend.", 0.15),
        LiftPhase("bottom", "Lead thigh parallel, trailing knee above floor.", 0.45),
        LiftPhase("drive_recovery", "Lead leg drives back toward standing.", 0.50),
        LiftPhase("lockout", "Feet together in tall standing position.", 0.90),
    )
    grip = LiftGrip("palm_centre", "pronated", 0.65, (0.50, 0.85))
    stance = StanceGeometry(
        foot_width_m=0.30,
        toe_out_deg=10.0,
        is_split=True,
        split_length_m=0.80,
        lead_foot="l",
        trail_foot="r",
        lead_foot_position_m=(0.42, 0.15, 0.0),
        trail_foot_position_m=(-0.38, -0.15, 0.0),
        lead_toe_out_deg=0.0,
        trail_toe_out_deg=5.0,
    )
    standing_coords = _symmetric_coords(
        {
            "lumbar_flex": 0.05,
            "neck_flex": 0.0,
            "shoulder_{s}_flex": 2.79,
            "shoulder_{s}_adduct": -2.85,
            "shoulder_{s}_rotate": 0.68,
            "elbow_{s}_flex": 2.49,
            "wrist_{s}_deviate": -0.03,
            "hip_{s}_flex": 0.05,
            "knee_{s}_flex": -0.05,
            "ankle_{s}_flex": 0.03,
        }
    )
    bottom_coords = {
        "lumbar_flex": 0.08,
        "neck_flex": 0.0,
        "shoulder_l_flex": 2.79,
        "shoulder_r_flex": 2.79,
        "shoulder_l_adduct": -2.85,
        "shoulder_r_adduct": -2.85,
        "shoulder_l_rotate": 0.68,
        "shoulder_r_rotate": 0.68,
        "elbow_l_flex": 2.49,
        "elbow_r_flex": 2.49,
        "wrist_l_deviate": -0.03,
        "wrist_r_deviate": -0.03,
        "hip_l_flex": 1.48,
        "hip_r_flex": -0.22,
        "hip_l_adduct": -0.05,
        "hip_r_adduct": -0.05,
        "knee_l_flex": -1.58,
        "knee_r_flex": -1.52,
        "ankle_l_flex": 0.38,
        "ankle_r_flex": 0.12,
    }
    via_poses = {
        "start": ViaPose(
            "start",
            "standing",
            (0.03, 0.0, 1.01),
            (0.0, 0.10, 0.0),
            (-0.03, 0.0, 1.59),
            standing_coords,
        ),
        "catch": ViaPose(
            "catch",
            "bottom",
            (0.02, 0.0, 0.65),
            (0.0, 0.08, 0.0),
            (0.0, 0.0, 1.22),
            bottom_coords,
        ),
        "lockout": ViaPose(
            "lockout",
            "lockout",
            (0.03, 0.0, 1.01),
            (0.0, 0.10, 0.0),
            (-0.03, 0.0, 1.59),
            standing_coords,
        ),
    }
    return AdditionalMovementSpec(
        "lunge", 5, "free", phases, grip, stance, BarbellLoad(), via_poses
    )


_SPECS: dict[str, AdditionalMovementSpec] = {
    "split_jerk": _make_split_jerk(),
    "push_press": _make_push_press(),
    "overhead_press": _make_overhead_press(),
    "front_squat": _make_front_squat(),
    "lunge": _make_lunge(),
}


def is_additional_movement(lift: str) -> bool:
    """Return True if *lift* is one of the five additional movements."""
    return lift in _SPECS


def additional_movement_spec(lift: str) -> AdditionalMovementSpec:
    """Retrieve the movement specification for an additional lift.

    Raises:
        ValueError: If *lift* is not one of :data:`ADDITIONAL_LIFTS`.
    """
    if lift not in _SPECS:
        raise ValueError(
            f"unknown additional lift {lift!r}; choose from {list(ADDITIONAL_LIFTS)}"
        )
    return _SPECS[lift]
