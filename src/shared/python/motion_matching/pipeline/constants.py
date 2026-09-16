"""Constants for ground-supported motion matching and simulation pipelines.

All physical units are SI (metres, radians, seconds, Hertz, Newtons).
Preconditions / Invariants:
- Frequencies, sample steps, and iterations must be strictly positive.
- Relaxation gains must be in (0, 1].
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.shared.python.motion_matching.range_of_motion import (
    HUMAN_RANGES_DEG,
    LOWER_LIMB_RANGES_DEG,
)

REPO_ROOT: Path = Path(__file__).resolve().parents[5]
DATA_DIR: Path = REPO_ROOT / "data"
CAPTURES: dict[str, Path] = {
    "driver": DATA_DIR / "C3D_TA_Driver.c3d",
    "iron": DATA_DIR / "C3D_TA_Iron.c3d",
}
FULL_BODY_DIR: Path = REPO_ROOT / "docs/development/full_body_models"
SPEC: Path = FULL_BODY_DIR / "full_body_spec_v2.json"
BUILD_RECEIPT: Path = FULL_BODY_DIR / "build_receipt_v2.json"
UPPER_SPEC: Path = (
    REPO_ROOT
    / "docs/development/simscape_tour_matching/native_evidence/native_geometry_spec_9967.json"
)
CANDIDATE: Path = FULL_BODY_DIR / "evidence/native_candidates/returned81_candidate.json"

UP_AXIS = np.array([0.0, 0.0, 1.0])
FORWARD_AXIS = np.array([-1.0, 0.0, 0.0])
RIGHT_AXIS = np.array([0.0, 1.0, 0.0])

TOE_STANDOFF_M: float = (
    0.03  # marker centre above the sole (shoe upper plus marker radius)
)

# Contact spheres extending support polygon to the toe tips:
TOE_SPHERES: dict[str, tuple[str, tuple[float, float, float], float]] = {
    "toe_r": ("calcn_r", (0.23, -0.010, 0.0), 0.025),
    "toe_l": ("calcn_l", (0.23, -0.010, 0.0), 0.025),
}

LEG_SEEDS: dict[str, tuple[str, tuple[float, float, float]]] = {
    # OpenSim body frames: x forward, y up the segment, z to the right
    "RKneeOut": ("femur_r", (0.0, -0.40, 0.06)),
    "LKneeOut": ("femur_l", (0.0, -0.39, -0.06)),
    "RAnkleOut": ("tibia_r", (-0.01, -0.44, 0.055)),
    "LAnkleOut": ("tibia_l", (-0.01, -0.43, -0.055)),
    "RToeIn": ("calcn_r", (0.19, 0.03, -0.03)),
    "RToeOut": ("calcn_r", (0.17, 0.03, 0.06)),
    "LToeIn": ("calcn_l", (0.19, 0.03, 0.03)),
    "LToeOut": ("calcn_l", (0.17, 0.03, -0.06)),
}
LEG_LABELS: tuple[str, ...] = tuple(LEG_SEEDS)

BOUND_WIDENING: float = 1.0

ADDRESS_SEEDS_DEG: list[dict[str, float]] = [
    {"hip_flexion": flexion, "knee_angle": knee, "hip_rotation": rotation}
    for flexion, knee in ((20.0, -20.0), (45.0, -25.0), (65.0, -30.0))
    for rotation in (-20.0, 0.0, 20.0)
]

STANCE_TOLERANCE_M: float = 0.02  # marker within this height of address: on ground
REFERENCE_CUTOFF_HZ: float = 12.0  # zero-phase low-pass on IK reference
TRACKING_CUTOFF_HZ: float = 12.0  # zero-phase low-pass on re-solved reference

ZMP_MARGIN_M: float = 0.02
ZMP_FILTER_ITERATIONS: int = 3
ZMP_COM_WEIGHT: float = 200.0

SHOOTING_LOCKED: tuple[str, ...] = (
    "TranslationInputX",
    "TranslationInputY",
    "HipInputX",
    "HipInputY",
    "HipInputZ",
)
SHOOTING_RELAXATION: float = 0.7

CONSISTENCY_PRIOR: float = 0.1
CALIBRATION_STRIDE: int = 6
STATIC_FRAMES: int = 24
HEAD_MARKER_WEIGHT: float = 0.1
TRAJECTORY_RESTARTS: int = 4
TRAJECTORY_RESTART_THRESHOLD_M: float = 0.03
TRAJECTORY_RESTART_MARGIN_M: float = 0.003

SHOULDER_GIMBALS: tuple[tuple[str, str, str], ...] = tuple(
    (f"{s}SInputX", f"{s}SInputY", f"{s}SInputZ") for s in ("L", "R")
)

SPIN_PRIOR: float = 0.1
SPIN_COORDINATES: tuple[str, ...] = tuple(
    f"{s}{c}" for s in ("L", "R") for c in ("SInputZ", "FInput", "WInputY")
)

NEUTRAL_LOCKS: dict[str, float] = {
    "LScapInputX": 0.0,
    "RScapInputX": 0.0,
    "RScapInputY": 0.0,
}
NEUTRAL_BOUNDS_DEG: dict[str, tuple[float, float]] = {
    "SpineInputX": (-5.0, 5.0),
    "SpineInputY": (-10.0, 10.0),
    "LScapInputY": (0.0, 20.0),
}

ADDRESS_RESTARTS: int = 6
ADDRESS_RESTART_SPREAD_RAD: float = 0.5
ADDRESS_ELBOW_BOUNDS_DEG: dict[str, tuple[float, float]] = {
    "LEInput": (-35.0, 5.0),
    "REInput": (-30.0, -3.0),
}

ELBOW_PIT_MARKERS: dict[str, tuple[str, str, str]] = {
    "L": ("LShoulderTop", "LElbowOut", "LWristTop"),
    "R": ("RShoulderBack", "RElbowOut", "RWristTop"),
}
ELBOW_PIT_WEIGHT: float = 0.02
ELBOW_PIT_WEIGHT_SWING: float = 0.01
ADDRESS_BALANCE_WEIGHT: float = 3.0
ELBOW_PIT_WEIGHTS_NEUTRAL: tuple[float, ...] = (0.02,)

CALIBRATION_ITERATIONS: int = 3
CALIBRATION_PRIOR_FRAMES: float = 40.0
SCALE_GRID: tuple[float, ...] = (0.94, 0.97, 1.00)

DT_S: float = 1e-3
OMEGA_RAD_S: float = 30.0
BALANCE: tuple[float, float] = (60.0, 15.0)
RATE_HZ: float = 360.0
PLAYBACK_STRIDE: int = 6
PRIOR: float = 1e-3
CONTACT_STIFFNESS_N_M: float = 2.0e5

IK_UNBOUNDED: frozenset[str] = frozenset(
    {"LWInputX", "RWInputX", "LWInputY", "RWInputY", "LFInput", "RFInput"}
)

TRAIL_WRIST_ADDRESS_DEG: dict[str, float] = {
    "RWInputX": -10.0,
    "RWInputY": 0.0,
    "RFInput": 30.0,
}
WRIST_COORDINATES: tuple[str, ...] = (
    "LWInputX",
    "LWInputY",
    "LFInput",
    "RWInputX",
    "RWInputY",
    "RFInput",
)
