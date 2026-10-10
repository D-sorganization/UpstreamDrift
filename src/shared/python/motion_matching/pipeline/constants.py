"""Constants for ground-supported motion matching and simulation pipelines.

All physical units are SI (metres, radians, seconds, Hertz, Newtons).
Preconditions / Invariants:
- Frequencies, sample steps, and iterations must be strictly positive.
- Relaxation gains must be in (0, 1].
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
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
CAPTURE_NAMES: tuple[str, ...] = ("driver", "iron", "owner")


def capture_path(name: str) -> Path:
    """Resolve a capture name to its file path.

    Preconditions: ``name`` must be one of :data:`CAPTURE_NAMES`.

    ``"driver"`` and ``"iron"`` resolve to the public fixtures checked into
    ``data/``. ``"owner"`` resolves lazily, at call time, through the private
    capture registry (:func:`src.motion_capture.capture_registry.resolve_capture`)
    so importing this module never touches the private dataset location and
    never requires ``CAPTURE_DATA_DIR`` to be set.
    """
    if name == "owner":
        from src.motion_capture.capture_registry import resolve_capture

        return resolve_capture("capture-O")
    if name not in CAPTURES:
        raise ValueError(
            f"Unknown capture name: {name!r}; expected one of {CAPTURE_NAMES}"
        )
    return CAPTURES[name]


def rate_from_times(times: Sequence[float] | np.ndarray) -> float:
    """Derive the sample rate in Hz from a capture's strictly monotonic timestamps.

    Preconditions:
    - At least 2 samples.
    - All spacings are positive and uniform (relative tolerance 1e-6).

    Raises ``ValueError`` if the preconditions are not met -- there is no
    silent fallback to a default rate.
    """
    arr = np.asarray(times, dtype=float)
    if arr.size < 2:
        raise ValueError("rate_from_times requires at least 2 samples")
    deltas = np.diff(arr)
    if np.any(deltas <= 0.0):
        raise ValueError(
            "rate_from_times requires strictly increasing, positive sample spacing"
        )
    if not np.allclose(deltas, deltas[0], rtol=1e-6):
        raise ValueError("rate_from_times requires uniform sample spacing")
    return float(1.0 / deltas[0])


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


def square_forefoot_seeds(
    seeds: Mapping[str, tuple[str, tuple[float, float, float]]],
) -> dict[str, tuple[str, tuple[float, float, float]]]:
    """Copy of ``seeds`` whose ToeIn/ToeOut markers share one forward (x) offset.

    The stock seeds stagger ToeIn 0.02 m ahead of ToeOut over a 0.09 m span, an
    unmeasured 12.5 degree yaw bias: the C3D ToeIn-ToeOut line is perpendicular
    to the foot axis within a few degrees (OSV-4, #11730), so a marker fit
    against the stock seeds turns the model foot by that bias. Each side's x is
    set to the mean of the two seeds, keeping the toe-midpoint where it was.
    """
    out = dict(seeds)
    for side in ("R", "L"):
        inner, outer = f"{side}ToeIn", f"{side}ToeOut"
        if inner not in seeds or outer not in seeds:
            raise ValueError(f"seeds must contain {inner} and {outer}")
        mean_x = 0.5 * (seeds[inner][1][0] + seeds[outer][1][0])
        for label in (inner, outer):
            body, (_, y, z) = seeds[label]
            out[label] = (body, (mean_x, y, z))
    return out


BOUND_WIDENING: float = 1.0

ADDRESS_SEEDS_DEG: list[dict[str, float]] = [
    {"hip_flexion": flexion, "knee_angle": knee, "hip_rotation": rotation}
    for flexion, knee in ((20.0, -20.0), (45.0, -25.0), (65.0, -30.0))
    for rotation in (-20.0, 0.0, 20.0)
]

STANCE_TOLERANCE_M: float = 0.02  # marker within this height of address: on ground
REFERENCE_CUTOFF_HZ: float = 12.0  # zero-phase low-pass on IK reference
TRACKING_CUTOFF_HZ: float = 12.0  # zero-phase low-pass on re-solved reference
#: Pre-contact tracking cutoffs tried, lowest first, by the release-preserving
#: selection (GCV-20, #11767; DESIGN_DECISIONS section 11). The first is the
#: base cutoff; post-contact samples always keep TRACKING_CUTOFF_HZ.
RELEASE_CUTOFF_CANDIDATES_HZ: tuple[float, ...] = (12.0, 15.0, 18.0, 20.0, 25.0, 30.0)
#: Impact-speed agreement with the unfiltered reference that keeps a release.
RELEASE_SPEED_TOL: float = 0.01
# Trail-arm coordinates that re-close the dual-grip weld on the tracked
# reference (GCV-20, #11767); the lead arm and body keep the filtered pose.
TRAIL_ARM_WELD_COORDINATES: tuple[str, ...] = (
    "RScapInputX",
    "RScapInputY",
    "RSInputX",
    "RSInputY",
    "RSInputZ",
    "REInput",
    "RFInput",
    "RWInputX",
    "RWInputY",
)

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
#: Restart policies of the full-capture trajectory IK (#12042): ``free`` keeps
#: the legacy jittered restarts (the prior follows the jittered seed and any
#: 3 mm better retry wins, so a weakly observed coordinate can hop branches
#: between two frames); ``seeded`` draws each frame's jitter from a
#: frame-indexed generator; ``anchored`` keeps the retry's prior on the
#: frame's start pose; ``stable`` does both; ``continuous`` anchors the retry's prior to the
#: frame's start and rejects a retry that implies a joint speed above
#: ``RESTART_MAX_JOINT_SPEED_RAD_S``; ``off`` disables restarts.
IK_RESTART_POLICIES: tuple[str, ...] = (
    "free",
    "seeded",
    "anchored",
    "stable",
    "continuous",
    "off",
)
DEFAULT_IK_RESTART_POLICY: str = "free"
#: 2000 deg/s, above every joint speed of the smoothed address-to-impact
#: references of capture-A and capture-B (peak 1299 deg/s, lead wrist).
RESTART_MAX_JOINT_SPEED_RAD_S: float = 34.9
#: Coordinates the follow-through markers leave under-determined (#12042):
#: toes, shoulder axial rotation and forearm rotation. The optional posture
#: prior pulls them weakly toward the calibrated address pose.
POSTURE_PRIOR_COORDINATES: tuple[str, ...] = (
    "mtp_angle_r",
    "mtp_angle_l",
    "LSInputZ",
    "RSInputZ",
    "LFInput",
    "RFInput",
)

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
PRIOR: float = 1e-3
CONTACT_STIFFNESS_N_M: float = 2.0e5
DEFAULT_MJX_ITERATIONS: int = 40

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
