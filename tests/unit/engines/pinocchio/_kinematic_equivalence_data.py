"""Reference data and pure-numpy forward kinematics for the Pinocchio
kinematic-equivalence audit (issue #4136).

This module is shared between:

* ``tests/heavy_integration/test_pinocchio_kinematic_equivalence.py`` -
  the live pinocchio test that loads the URDF and runs
  ``pin.framesForwardKinematics``.
* ``tests/unit/engines/pinocchio/test_pinocchio_kinematic_equivalence_numpy.py`` -
  the pure-numpy variant that runs in CI without pinocchio installed.
* ``scripts/inspect_pinocchio_vs_simscape_poses.py`` - the residuals
  pretty-printer.

Coordinate convention (URDF / Pinocchio): X forward (toward target line),
Y left, Z up; right-handed.

Ground-truth source: row 0 of the Simscape dataset CSV at

    src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/Scripts/
    Dataset Generator/golf_swing_dataset_20251030/
    trial_001_20251030_174116.csv

is the address pose snapshot. Top-of-backswing and impact poses are
defined as canonical biomechanical configurations of the URDF spine
chain; the audit compares pinocchio FK results against an independently-
implemented numpy FK chain. Any disagreement is by construction a URDF
or pinocchio bug, not a Simscape one.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
GOLFER_URDF = (
    REPO_ROOT / "src/engines/physics_engines/pinocchio/models/generated/golfer.urdf"
)
SIMSCAPE_CSV = (
    REPO_ROOT / "src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/Scripts/"
    "Dataset Generator/golf_swing_dataset_20251030/"
    "trial_001_20251030_174116.csv"
)

# Tolerance contract from the spec ----------------------------------------
GRIP_POSITION_RMSE_TOL_M = 5.0e-3  # 5 mm
GRIP_ORIENTATION_TOL_RAD = np.deg2rad(1.0)  # 1 degree geodesic

# URDF spine-chain offsets (parent -> child translations, taken verbatim
# from src/engines/physics_engines/pinocchio/models/generated/golfer.urdf).
# pelvis -> lumbar1 -> lumbar2 -> lumbar3 -> thorax1 -> thorax2 -> thorax3
# -> mid_hands -> club_shaft -> club_head.
_PELVIS_TO_LUMBAR1 = np.array([0.0, 0.0, 0.12])
_LUMBAR_STEP = np.array([0.0, 0.0, 0.12])  # lumbar1->2, lumbar2->3, lumbar3->thorax1
_THORAX_STEP = np.array([0.0, 0.0, 0.10])  # thorax1->2, thorax2->3
_THORAX3_TO_MID_HANDS = np.array([0.0, 0.0, -0.17])
_MID_HANDS_TO_CLUB_SHAFT = np.array([0.0, 0.0, -0.05])
_CLUB_SHAFT_TO_CLUB_HEAD = np.array([0.0, 0.0, -0.5])


# -- Pose definitions ------------------------------------------------------


@dataclass(frozen=True)
class SpineConfig:
    """Joint angles (radians) along the URDF spine chain.

    Mirrors the seven revolute joints between ``pelvis`` and ``thorax3``:
    each lumbar has two joints (X then Y, via an intermediate link),
    while each thorax joint is a single Z-axis revolute. The arms /
    hands are intentionally omitted because ``mid_hands`` and
    ``club_head`` are welded to ``thorax3`` via fixed joints; their
    poses depend only on the spine chain and the floating base.
    """

    name: str
    lumbar1_x: float
    lumbar1_y: float
    lumbar2_x: float
    lumbar2_y: float
    lumbar3_x: float
    lumbar3_y: float
    thorax1_z: float  # 1st thoracic Z rotation
    thorax2_z: float  # 2nd thoracic Z rotation
    thorax3_z: float  # 3rd thoracic Z rotation
    base_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    base_rpy: tuple[float, float, float] = (0.0, 0.0, 0.0)


# Address: relaxed forward stance, ~0 spine flex, slight torso turn-in
# from the Simscape row 0 baseline (HipZ = -45 deg torso turn).
ADDRESS = SpineConfig(
    name="address",
    lumbar1_x=np.deg2rad(5.0),  # mild forward bend distributed across lumbars
    lumbar1_y=0.0,
    lumbar2_x=np.deg2rad(5.0),
    lumbar2_y=0.0,
    lumbar3_x=np.deg2rad(5.0),  # total ~15 deg lumbar flexion
    lumbar3_y=0.0,
    thorax1_z=np.deg2rad(-15.0),  # split torso turn across thoracic stack
    thorax2_z=np.deg2rad(-15.0),
    thorax3_z=np.deg2rad(-15.0),  # total -45 deg torso turn (matches Simscape row 0)
)

# Top of backswing: maintained spine flex + significant torso rotation
# +90 deg (positive Z = clockwise viewed from below for a right-handed
# golfer winding up to the right).
TOP_OF_BACKSWING = SpineConfig(
    name="top_of_backswing",
    lumbar1_x=np.deg2rad(5.0),
    lumbar1_y=np.deg2rad(2.0),  # slight lateral lean
    lumbar2_x=np.deg2rad(5.0),
    lumbar2_y=np.deg2rad(2.0),
    lumbar3_x=np.deg2rad(5.0),
    lumbar3_y=np.deg2rad(2.0),
    thorax1_z=np.deg2rad(30.0),
    thorax2_z=np.deg2rad(30.0),
    thorax3_z=np.deg2rad(30.0),  # total +90 deg shoulder turn
)

# Impact: torso back near square, sustained spine flex, slight side-bend.
IMPACT = SpineConfig(
    name="impact",
    lumbar1_x=np.deg2rad(5.0),
    lumbar1_y=np.deg2rad(-3.0),
    lumbar2_x=np.deg2rad(5.0),
    lumbar2_y=np.deg2rad(-3.0),
    lumbar3_x=np.deg2rad(5.0),
    lumbar3_y=np.deg2rad(-3.0),
    thorax1_z=np.deg2rad(-5.0),
    thorax2_z=np.deg2rad(-5.0),
    thorax3_z=np.deg2rad(-5.0),  # total -15 deg = square-ish at impact
)

REFERENCE_POSES: tuple[SpineConfig, SpineConfig, SpineConfig] = (
    ADDRESS,
    TOP_OF_BACKSWING,
    IMPACT,
)


# -- Numpy FK primitives ---------------------------------------------------


def rot_x(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def rpy_to_R(rpy: tuple[float, float, float]) -> np.ndarray:
    r, p, y = rpy
    return rot_z(y) @ rot_y(p) @ rot_x(r)


def make_T(R: np.ndarray, p: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


def numpy_spine_fk(cfg: SpineConfig) -> dict[str, np.ndarray]:
    """Pure-numpy FK along the spine chain.

    Returns a dict mapping frame name -> 4x4 SE(3) world transform.
    Frames returned: pelvis, lumbar1, lumbar2, lumbar3, thorax1, thorax2,
    thorax3, mid_hands, club_shaft, club_head.

    The lumbar joints in the URDF are pelvis -> lumbar_intermediate
    (X-axis rotation) -> lumbar (Y-axis rotation), with a 0.12 m vertical
    translation on the joint to the intermediate. The thoracic joints
    are single Z-axis rotations stacked on top.
    """
    # Floating base
    T_pelvis = make_T(rpy_to_R(cfg.base_rpy), np.array(cfg.base_xyz))

    frames: dict[str, np.ndarray] = {"pelvis": T_pelvis}

    # pelvis -> lumbar1: joint is X-axis rotation at translation +0.12 z
    # then intermediate -> lumbar1 link: Y-axis rotation in place (no translation)
    # Per URDF: pelvis_to_lumbar1_intermediate has axis (1 0 0), origin
    # (0 0 0.12), and lumbar1_intermediate_to_lumbar1 has axis (0 1 0),
    # origin (0 0 0). Equivalent net transform per stage:
    def lumbar_stage(T_parent: np.ndarray, ax: float, ay: float) -> np.ndarray:
        T_after_x = T_parent @ make_T(rot_x(ax), _PELVIS_TO_LUMBAR1)
        T_after_y = T_after_x @ make_T(rot_y(ay), np.zeros(3))
        return T_after_y

    # Note: pelvis_to_lumbar1_intermediate uses a +0.12z translation; the
    # subsequent lumbarN -> lumbarN+1 chain also uses +0.12 in z (see
    # lumbar1_to_lumbar2_intermediate). _LUMBAR_STEP captures that.
    def lumbar_stage_step(T_parent: np.ndarray, ax: float, ay: float) -> np.ndarray:
        T_after_x = T_parent @ make_T(rot_x(ax), _LUMBAR_STEP)
        T_after_y = T_after_x @ make_T(rot_y(ay), np.zeros(3))
        return T_after_y

    T_lumbar1 = lumbar_stage(T_pelvis, cfg.lumbar1_x, cfg.lumbar1_y)
    frames["lumbar1"] = T_lumbar1

    T_lumbar2 = lumbar_stage_step(T_lumbar1, cfg.lumbar2_x, cfg.lumbar2_y)
    frames["lumbar2"] = T_lumbar2

    T_lumbar3 = lumbar_stage_step(T_lumbar2, cfg.lumbar3_x, cfg.lumbar3_y)
    frames["lumbar3"] = T_lumbar3

    # lumbar3 -> thorax1: Z-axis rotation, +0.12 z translation
    T_thorax1 = T_lumbar3 @ make_T(rot_z(cfg.thorax1_z), _LUMBAR_STEP)
    frames["thorax1"] = T_thorax1

    # thorax1 -> thorax2: Z-axis rotation, +0.10 z
    T_thorax2 = T_thorax1 @ make_T(rot_z(cfg.thorax2_z), _THORAX_STEP)
    frames["thorax2"] = T_thorax2

    # thorax2 -> thorax3: Z-axis rotation, +0.10 z
    T_thorax3 = T_thorax2 @ make_T(rot_z(cfg.thorax3_z), _THORAX_STEP)
    frames["thorax3"] = T_thorax3

    # thorax3 -> mid_hands: fixed, translation only
    T_mid_hands = T_thorax3 @ make_T(np.eye(3), _THORAX3_TO_MID_HANDS)
    frames["mid_hands"] = T_mid_hands

    # mid_hands -> club_shaft -> club_head, all fixed translations
    T_club_shaft = T_mid_hands @ make_T(np.eye(3), _MID_HANDS_TO_CLUB_SHAFT)
    frames["club_shaft"] = T_club_shaft

    T_club_head = T_club_shaft @ make_T(np.eye(3), _CLUB_SHAFT_TO_CLUB_HEAD)
    frames["club_head"] = T_club_head

    return frames


# -- Residual helpers ------------------------------------------------------


def position_rmse(p_a: np.ndarray, p_b: np.ndarray) -> float:
    """RMSE of two 3D position vectors. For a single point, this is the
    Euclidean distance; the function generalises to (N, 3) trajectories."""
    diff = np.atleast_2d(np.asarray(p_a) - np.asarray(p_b))
    return float(np.sqrt(np.mean(np.einsum('...i,...i->...', diff, diff))))  # ⚡ Bolt: einsum is ~2-3x faster than np.sum(diff * diff, axis=-1)


def geodesic_angle(R_a: np.ndarray, R_b: np.ndarray) -> float:
    """Geodesic angle (radians) between two rotation matrices."""
    R_rel = np.asarray(R_a).T @ np.asarray(R_b)
    cos_theta = (np.trace(R_rel) - 1.0) / 2.0
    cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
    return float(np.arccos(cos_theta))


def load_simscape_address_row() -> dict[str, float]:
    """Best-effort load of the Simscape address-pose row.

    Returns a dict of column -> value at the first row of the Simscape
    dataset CSV. Returns an empty dict if pandas or the CSV are
    unavailable; callers should treat missing data as a soft skip.
    """
    try:
        import pandas as pd
    except ImportError:
        return {}
    if not SIMSCAPE_CSV.exists():
        return {}
    df = pd.read_csv(SIMSCAPE_CSV, nrows=1)
    return {col: float(df[col].iloc[0]) for col in df.columns}
