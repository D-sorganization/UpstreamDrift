"""Pure-data + math core for the Starting-Pose Matcher.

This module deliberately contains NO Qt / matplotlib imports so it can be
unit-tested in environments that don't have the GUI stack working.  The
GUI module (``gui.py``) imports everything from here.

This is a single-engine starting-pose matcher today (Simscape 3D golf
model).  The cross-engine port is tracked by issue #4367 — see the
``skeleton_provider`` module.

Public API
----------
Constants:
    CM_TO_M, SESSION_SCHEMA_VERSION,
    EVENT_KEYS, EVENT_LABEL_PRESETS, DEFAULT_EVENT_PRESET,
    PHASE_KEYS, PHASE_BOUNDS, DEFAULT_PHASE,
    FALLBACK_SEGMENTS

Dataclasses:
    MocapEvents, Skeleton, RigidTransform,
    SkeletonTrajectory, PoseSlot

Functions:
    load_mocap_xlsx(path, sheet)         -> pd.DataFrame  (cm → m)
    read_event_header(path, sheet)       -> MocapEvents
    load_skeleton(json_path, fallback)   -> Skeleton
    load_simscape_trajectory_csv(path)   -> SkeletonTrajectory
    solve_shaft_rz_deg(...)              -> float
    phase_display_label(key, labels)     -> str
    phase_key_from_label(label)          -> str | None

Shared infrastructure (per AGENTS.md / issue #4376)
---------------------------------------------------
* ``forward_kinematics`` and ``reference_golfer_setup`` from
  ``src.shared.python.motion_matching.diagnostics`` are used to BUILD
  the fallback skeletons from canonical joint angles instead of from
  hand-tuned Cartesian dicts.  Result: when a teammate updates the
  reference pose, our fallbacks update with it.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# Adopt shared infrastructure (per #4376).  We MAP between the shared
# point-name vocabulary ("pelvis", "spine_top", "torso_top",
# "l_shoulder", ...) and the matcher's compact short names ("hip",
# "spine", "torso", "ls", ...) below.
from src.shared.python.motion_matching.diagnostics import (
    SkeletonPose,
    forward_kinematics,
    reference_golfer_setup,
)

# Canonical ClubTarget adapter (per #4404 - replace local Wiffle loader)
from src.shared.python.motion_matching.loaders.excel import (
    ExcelEventMarkers,
    read_excel_event_markers,
)
from src.shared.python.motion_matching.load_club_target import load_club_target
from src.shared.python.motion_matching.multi_source_target import MultiSourceTarget
from src.shared.python.motion_matching.target import AlignOptions, ClubTarget
from src.tools.starting_pose_matcher.session_schema import (
    SESSION_SCHEMA_VERSION as SESSION_SCHEMA_VERSION,
)

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# Units: Wiffle xlsx positions are in CM.  See MATLAB_GOLF_MODEL_GUIDE.md.
# ----------------------------------------------------------------------------
CM_TO_M = 0.01

# Schema version for session JSON: re-exported from ``session_schema`` at
# the top of this file.  v4 added the ``data_sources`` block (issue #4480).

# Event-label conventions.
EVENT_KEYS: tuple[str, ...] = ("A", "T", "I", "F")
EVENT_LABEL_PRESETS: dict[str, dict[str, str]] = {
    "Wiffle (A/T/I/F)": {
        "A": "Address",
        "T": "Top of Backswing",
        "I": "Impact",
        "F": "Finish",
    },
    "Trackman P-system": {
        "A": "P1 Address",
        "T": "P4 Top",
        "I": "P7 Impact",
        "F": "P10 Finish",
    },
    "Plain English": {
        "A": "Setup",
        "T": "Backswing top",
        "I": "Strike",
        "F": "Follow-through end",
    },
    "Sequence numbers": {
        "A": "Phase 1",
        "T": "Phase 2",
        "I": "Phase 3",
        "F": "Phase 4",
    },
}
DEFAULT_EVENT_PRESET = "Wiffle (A/T/I/F)"

# Phase windows.  Stable LOGICAL identifiers; user-facing labels are built
# from the current event labels via ``phase_display_label()``.
PHASE_KEYS: tuple[str, ...] = (
    "none",
    "backswing",
    "downswing",
    "follow_through",
    "full_swing",
    "manual",
)
PHASE_BOUNDS: dict[str, tuple[str | None, str | None]] = {
    "none": (None, None),
    "backswing": ("A", "T"),
    "downswing": ("T", "I"),
    "follow_through": ("I", "F"),
    "full_swing": ("A", "F"),
    "manual": ("manual", "manual"),
}
DEFAULT_PHASE = "full_swing"

# Backwards-compatible alias used by older sessions.
PHASE_LEGACY_LABELS: dict[str, str] = {
    "None": "none",
    "Backswing (A → T)": "backswing",
    "Downswing (T → I)": "downswing",
    "Follow-through (I → F)": "follow_through",
    "Full swing (A → F)": "full_swing",
    "Manual range": "manual",
}


def phase_display_label(key: str, event_labels: dict[str, str]) -> str:
    """Spell out a phase key using the current event labels.

    >>> phase_display_label("backswing",
    ...     {"A": "Address", "T": "Top of Backswing", "I": "Impact", "F": "Finish"})
    'Backswing (Address to Top of Backswing)'
    """
    a = event_labels.get("A", "Address")
    t = event_labels.get("T", "Top of Backswing")
    i = event_labels.get("I", "Impact")
    f = event_labels.get("F", "Finish")
    table = {
        "none": "None - draw entire data range",
        "backswing": f"Backswing ({a} to {t})",
        "downswing": f"Downswing ({t} to {i})",
        "follow_through": f"Follow-through ({i} to {f})",
        "full_swing": f"Full swing ({a} to {f})",
        "manual": "Manual frame range",
    }
    return table.get(key, key)


def phase_key_from_label(label: str) -> str | None:
    """Look up a phase key from a (legacy or current) display label."""
    if label in PHASE_LEGACY_LABELS:
        return PHASE_LEGACY_LABELS[label]
    if label in PHASE_BOUNDS:
        return label
    leading = label.split(" ", 1)[0].lower().rstrip(":-")
    for k in PHASE_KEYS:
        if k.startswith(leading):
            return k
    return None


# ----------------------------------------------------------------------------
# Dataclasses
# ----------------------------------------------------------------------------


@dataclass
class MocapEvents:
    """Sample numbers (1-based) for A/T/I/F + CHS_mph (NaN if missing)."""

    A_sample: float = float("nan")
    T_sample: float = float("nan")
    I_sample: float = float("nan")
    F_sample: float = float("nan")
    CHS_mph: float = float("nan")

    def frame_for(self, label: str) -> int | None:
        v = getattr(self, f"{label}_sample", float("nan"))
        if v != v:
            return None
        return max(0, int(v) - 1)


@dataclass
class Skeleton:
    """Joint world positions (metres) at one model pose."""

    name: str = "default"
    joints: dict[str, np.ndarray] = field(default_factory=dict)
    segments: list[tuple[str, str]] = field(default_factory=list)


@dataclass
class RigidTransform:
    """7-DOF transform: Tx/Ty/Tz (m), Rx/Ry/Rz (deg), Scale.

    P' = scale * R * (P - pivot) + pivot + t   where R = Rz @ Ry @ Rx.
    """

    tx: float = 0.0
    ty: float = 0.0
    tz: float = 0.0
    rx: float = 0.0
    ry: float = 0.0
    rz: float = 0.0
    scale: float = 1.0
    pivot: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def matrix(self) -> tuple[np.ndarray, np.ndarray]:
        cx, cy, cz = (np.cos(np.deg2rad(a)) for a in (self.rx, self.ry, self.rz))
        sx, sy, sz = (np.sin(np.deg2rad(a)) for a in (self.rx, self.ry, self.rz))
        Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
        return (Rz @ Ry @ Rx) * float(self.scale), np.array([self.tx, self.ty, self.tz])

    def apply(self, points: np.ndarray) -> np.ndarray:
        R, t = self.matrix()
        pivot = np.array(self.pivot)
        return (points - pivot) @ R.T + pivot + t


@dataclass
class SkeletonTrajectory:
    """Time series of skeleton joint positions (one Skeleton per frame)."""

    times: np.ndarray = field(default_factory=lambda: np.zeros(0))
    frames: list[Skeleton] = field(default_factory=list)
    source_path: str = ""

    def __len__(self) -> int:
        return len(self.frames)

    def frame_at_time(self, t: float) -> int:
        if len(self.times) == 0:
            return 0
        i = int(np.argmin(np.abs(self.times - t)))
        return max(0, min(i, len(self.frames) - 1))


@dataclass
class PoseSlot:
    name: str
    skeleton: Skeleton
    color: str
    mocap_color: str
    target_event: str
    visible: bool = True
    trajectory: SkeletonTrajectory | None = None
    trajectory_frame_index: int = 0


# ----------------------------------------------------------------------------
# Skeleton fallback — derived from shared FK so the matcher tracks the
# canonical reference golfer setup automatically (see #4376).
# ----------------------------------------------------------------------------


# Body chain segments — connect the matcher's short joint names.
FALLBACK_SEGMENTS: list[tuple[str, str]] = [
    ("hip", "spine"),
    ("spine", "torso"),
    ("torso", "hub"),
    ("hub", "ls"),
    ("hub", "rs"),
    ("ls", "le"),
    ("rs", "re"),
    ("le", "lw"),
    ("re", "rw"),
    ("lw", "mp"),
    ("rw", "mp"),
    ("mp", "ch"),
]

# Map shared FK landmark names -> matcher's compact short names.  The
# shared FK exposes ``pelvis / spine_top / torso_top / l_shoulder /
# r_shoulder / l_elbow / r_elbow / l_wrist / r_wrist / l_hand / r_hand /
# butt / clubhead``.  The matcher uses ``hip / spine / torso / hub /
# ls / rs / le / re / lw / rw / mp / ch``.  ``hub`` is synthesised as the
# top of the torso for visualisation; ``mp`` is mid-hands.
_FK_TO_SHORT: dict[str, str] = {
    "pelvis": "hip",
    "spine_top": "spine",
    "torso_top": "hub",  # top of torso = hub in the matcher's vocabulary
    "l_shoulder": "ls",
    "r_shoulder": "rs",
    "l_elbow": "le",
    "r_elbow": "re",
    "l_wrist": "lw",
    "r_wrist": "rw",
    # l_hand and r_hand exist but the matcher uses mp = mid-of-hands
    "butt": "mp",
    "clubhead": "ch",
}


def _fk_to_skeleton(pose: SkeletonPose, name: str) -> Skeleton:
    """Convert a shared :class:`SkeletonPose` (shared-FK landmarks) to the
    matcher's :class:`Skeleton` (compact short names + ``torso`` joint).

    Synthesises the ``torso`` revolute-joint landmark at 20% from spine
    to hub (matching ``UpperTorsoBase = 0.2 * UpperTorsoLength`` in
    GolfSwing3D_Kinetic.mdl).
    """
    joints: dict[str, np.ndarray] = {}
    for fk_name, short in _FK_TO_SHORT.items():
        if fk_name in pose.points:
            joints[short] = np.array(pose.points[fk_name], dtype=float)
    if "spine" in joints and "hub" in joints:
        joints["torso"] = joints["spine"] + 0.2 * (joints["hub"] - joints["spine"])
    if "mp" in joints:
        joints["butt"] = joints["mp"].copy()
    return Skeleton(name=name, joints=joints, segments=list(FALLBACK_SEGMENTS))


def _build_address_fallback() -> Skeleton:
    """Address pose: shared canonical golfer setup, FK-evaluated.

    Uses the shared :func:`forward_kinematics` against
    :func:`reference_golfer_setup` so the matcher's Address fallback
    tracks any future updates to the canonical reference golfer angles.
    """
    angles = reference_golfer_setup()
    pose = forward_kinematics(angles)
    return _fk_to_skeleton(pose, name="Impact")


# Hand-tuned Cartesian Top-of-Backswing pose (metres).  We deliberately
# do NOT route this through ``forward_kinematics(reference_golfer_setup())``
# with mutated angles because the shared FK's hand-rest convention
# (lengths.hand applied along ±local Y) produces asymmetric hand heights
# that are difficult to fold back into a credible TOB by perturbing the
# angle set.  Instead we encode a known-plausible Cartesian pose directly
# — enough to show the user the expected body coil before they run
# ``export_default_skeleton.m`` for the real model joints.
#
# Numbers are anchored relative to the Address fallback's hub Z (~0.345 m
# above pelvis with the shared FK's default segment lengths) so the body
# proportions match across both poses.
_HANDCRAFTED_TOB_JOINTS: dict[str, list[float]] = {
    "hip": [0.00, 0.00, 0.00],
    "spine": [0.00, -0.10, 0.17],
    "torso": [0.00, -0.13, 0.21],  # ~20% from spine to hub (revolute joint)
    "hub": [0.00, -0.20, 0.34],
    # The FK-derived Address has shoulders along world ±Y because the
    # spine is tilted forward.  Top-of-backswing twists the torso 90°,
    # so the shoulder line should rotate to lie along world ±X.
    "ls": [+0.22, -0.18, 0.36],  # left shoulder swung to +X
    "rs": [-0.22, -0.22, 0.32],  # right shoulder swung to -X (and slightly back)
    # Hands raised high — both wrists above shoulder level.
    "le": [+0.15, -0.05, 0.60],
    "re": [-0.15, -0.35, 0.55],
    "lw": [+0.20, +0.05, 0.85],
    "rw": [+0.18, +0.10, 0.82],
    "mp": [+0.19, +0.08, 0.83],
    "ch": [-0.30, +0.40, 0.85],
}


def _build_top_of_backswing_fallback() -> Skeleton:
    """Top-of-Backswing pose: hand-tuned Cartesian skeleton.

    See ``_HANDCRAFTED_TOB_JOINTS`` for rationale.  Synthesises ``butt``
    as an alias for ``mp`` to match the convention used by the trajectory
    loader.
    """
    import numpy as _np  # local alias keeps this self-contained

    joints = {k: _np.array(v, dtype=float) for k, v in _HANDCRAFTED_TOB_JOINTS.items()}
    if "mp" in joints:
        joints["butt"] = joints["mp"].copy()
    return Skeleton(
        name="TopofBackswing", joints=joints, segments=list(FALLBACK_SEGMENTS)
    )


def fallback_skeleton(pose_name: str) -> Skeleton:
    """Return the fallback skeleton for a given pose name.

    Used when ``simscape_skeleton_<pose>.json`` is absent — the user
    can still launch the matcher and see a plausible body before
    running ``export_default_skeleton.m`` in MATLAB.
    """
    if pose_name.lower().startswith("top"):
        return _build_top_of_backswing_fallback()
    return _build_address_fallback()


# ----------------------------------------------------------------------------
# xlsx loaders — canonical ClubTarget adapter (per #4404)
# ----------------------------------------------------------------------------
# The matcher now uses the shared motion-matching infrastructure for xlsx
# loading. The local functions remain for backwards compatibility but are
# thin wrappers around the canonical loaders.
# ----------------------------------------------------------------------------


def _clubtarget_to_dataframe(target: ClubTarget) -> pd.DataFrame:
    """Convert a ClubTarget to a DataFrame compatible with the matcher GUI.

    This adapter converts the canonical ClubTarget format (butt/clubhead/quaternion)
    into the DataFrame schema expected by the matcher GUI (mid_X/Y/Z, club_X/Y/Z).

    Args:
        target: ClubTarget from shared loader

    Returns:
        DataFrame with columns: time, mid_X, mid_Y, mid_Z, club_X, club_Y, club_Z
        plus the 9 direction cosine columns for rotation matrix
    """
    n = len(target.time)
    # Build rotation matrix from quaternion for direction cosines
    # Quaternion format: [w, x, y, z] -> rotation matrix
    rotmats = np.empty((n, 3, 3), dtype=np.float64)
    for i in range(n):
        q = target.club_quat[i]
        w, x, y, z = q[0], q[1], q[2], q[3]
        # Rotation matrix from quaternion (row-major for direction cosines)
        rotmats[i] = np.array(
            [
                [
                    1 - 2 * y * y - 2 * z * z,
                    2 * x * y - 2 * z * w,
                    2 * x * z + 2 * y * w,
                ],
                [
                    2 * x * y + 2 * z * w,
                    1 - 2 * x * x - 2 * z * z,
                    2 * y * z - 2 * x * w,
                ],
                [
                    2 * x * z - 2 * y * w,
                    2 * y * z + 2 * x * w,
                    1 - 2 * x * x - 2 * y * y,
                ],
            ]
        )

    rows = []
    for i in range(n):
        rec = {
            "time": float(target.time[i]),
            "mid_X": float(target.butt[i, 0]),
            "mid_Y": float(target.butt[i, 1]),
            "mid_Z": float(target.butt[i, 2]),
            "club_X": float(target.clubhead[i, 0]),
            "club_Y": float(target.clubhead[i, 1]),
            "club_Z": float(target.clubhead[i, 2]),
            # Direction cosine columns (club_Xx, club_Xy, club_Xz, etc.)
            "club_Xx": float(rotmats[i, 0, 0]),
            "club_Xy": float(rotmats[i, 0, 1]),
            "club_Xz": float(rotmats[i, 0, 2]),
            "club_Yx": float(rotmats[i, 1, 0]),
            "club_Yy": float(rotmats[i, 1, 1]),
            "club_Yz": float(rotmats[i, 1, 2]),
            "club_Zx": float(rotmats[i, 2, 0]),
            "club_Zy": float(rotmats[i, 2, 1]),
            "club_Zz": float(rotmats[i, 2, 2]),
        }
        rows.append(rec)
    df = pd.DataFrame(rows)
    pos_cols = ["mid_X", "mid_Y", "mid_Z", "club_X", "club_Y", "club_Z"]
    # Optimize norm calculation along axis using einsum
    diff = df[["club_X", "club_Y", "club_Z"]].to_numpy(dtype=float) - df[
        ["mid_X", "mid_Y", "mid_Z"]
    ].to_numpy(dtype=float)
    shaft = np.sqrt(np.einsum("ij,ij->i", diff, diff))
    finite = np.isfinite(shaft) & (shaft > 1e-6)
    if finite.any() and float(np.median(shaft[finite])) > 1.4:
        # Wiffle workbook positions are centimetres. The legacy parser used by
        # the canonical loader historically applied an inches factor; keep the
        # matcher contract in metres without changing the shared loader here.
        df.loc[:, pos_cols] *= CM_TO_M / 0.0254
    return df


def load_mocap_xlsx(xlsx_path: str | Path, sheet_name: str) -> pd.DataFrame:
    """Load a Wiffle xlsx sheet into a DataFrame in metres.

    This function now uses the canonical ClubTarget loader and converts
    the result to the DataFrame format expected by the matcher GUI.

    Schema (subset): time (s), mid_X/Y/Z (m), club_X/Y/Z (m).
    """
    target = load_club_target(Path(xlsx_path), sheet=sheet_name, opts=AlignOptions())
    return _clubtarget_to_dataframe(target)


def _safe(row: pd.Series, idx: int, default: float = 0.0) -> float:
    try:
        v = row[idx]
    except (IndexError, KeyError):
        return default
    if pd.isna(v):
        return default
    try:
        return float(v)
    except (ValueError, TypeError):
        return default


def read_event_header(xlsx_path: str | Path, sheet_name: str) -> MocapEvents:
    """Parse the row-1 event-marker band: A=<n> T=<n> I=<n> F=<n> CHS=<mph>.

    This function now uses the canonical ExcelEventMarkers from the shared
    motion-matching infrastructure.
    """
    ev_markers: ExcelEventMarkers = read_excel_event_markers(
        Path(xlsx_path), sheet_name
    )
    return MocapEvents(
        A_sample=ev_markers.A_sample,
        T_sample=ev_markers.T_sample,
        I_sample=ev_markers.I_sample,
        F_sample=ev_markers.F_sample,
        CHS_mph=ev_markers.CHS_mph,
    )


# ----------------------------------------------------------------------------
# Skeleton loader — JSON file produced by export_default_skeleton.m, with
# FK-derived fallback when the file isn't available.
# ----------------------------------------------------------------------------


def load_skeleton(json_path: str | Path, fallback_pose: str = "Impact") -> Skeleton:
    """Load skeleton from JSON; fall back to FK-derived approximate pose.

    The fallback uses :func:`fallback_skeleton` which evaluates
    ``forward_kinematics(reference_golfer_setup())`` (with twist mutations
    for Top of Backswing) — so the fallback tracks any future updates to
    the canonical reference golfer pose automatically.
    """
    json_path = Path(json_path)
    if json_path.exists():
        with open(json_path) as f:
            data = json.load(f)
        joints = {k: np.array(v, dtype=float) for k, v in data["joints"].items()}
        raw_segments = data.get("segments", [])
        segments: list[tuple[str, str]] = []
        for s in raw_segments:
            if isinstance(s, list) and len(s) == 2:
                segments.append((str(s[0]), str(s[1])))
        return Skeleton(
            name=data.get("pose", fallback_pose),
            joints=joints,
            segments=segments or list(FALLBACK_SEGMENTS),
        )
    logger.warning(
        "%s not found - using FK-derived fallback %s pose. Run "
        "export_default_skeleton('%s') in MATLAB for actual model joints.",
        json_path,
        fallback_pose,
        fallback_pose,
    )
    return fallback_skeleton(fallback_pose)


# ----------------------------------------------------------------------------
# Skeleton-trajectory loader (Simscape CSV)
# ----------------------------------------------------------------------------


# Map "<our short name>" -> list of CSV column-name candidates for X.  Each
# candidate's _Y/_Z (or _2/_3, _y/_z) is auto-derived.
_TRAJECTORY_COLUMN_MAP: dict[str, list[str]] = {
    "ch": ["club_head_X", "club_head_x"],
    "lw": ["left_hand_X", "left_hand_x"],
    "rw": ["right_hand_X", "right_hand_x"],
    "ls": ["left_shoulder_X", "left_shoulder_x"],
    "rs": ["right_shoulder_X", "right_shoulder_x"],
    "le": ["left_elbow_X", "left_elbow_x"],
    "re": ["right_elbow_X", "right_elbow_x"],
    "hub": ["hub_X", "hub_x"],
    "torso": ["torso_X", "torso_x"],  # may be missing — synthesized
    "spine": ["spine_X", "spine_x"],
    "hip": ["hip_X", "hip_x"],
}

_TRAJECTORY_LONG_FORM: dict[str, str] = {
    "ch": "ClubLogs_CHGlobalPosition_1",
    "lw": "LWLogs_LHGlobalPosition_1",
    "rw": "RWLogs_RHGlobalPosition_1",
    "ls": "LSLogs_GlobalPosition_1",
    "rs": "RSLogs_GlobalPosition_1",
    "le": "LELogs_LArmonLForearmFGlobal_1",
    "re": "RELogs_RArmonLForearmFGlobal_1",
    "hub": "HipLogs_HUBGlobalPosition_1",
    "torso": "TorsoLogs_GlobalPosition_1",
    "spine": "SpineLogs_GlobalPosition_1",
    "hip": "HipLogs_HipGlobalPosition_dim1",
}


def _xyz_columns_for(df_columns: list[str], stem: str) -> list[str] | None:
    """Resolve X/Y/Z column names for a stem (e.g. 'club_head_X' or
    'ClubLogs_CHGlobalPosition_1').  None when any of three is missing.
    """
    cols_set = set(df_columns)
    if stem.endswith("_X"):
        a, b, c = stem, stem[:-1] + "Y", stem[:-1] + "Z"
    elif stem.endswith("_x"):
        a, b, c = stem, stem[:-1] + "y", stem[:-1] + "z"
    elif stem.endswith(("_1", "_dim1")):
        a, b, c = stem, stem[:-1] + "2", stem[:-1] + "3"
    else:
        return None
    if a in cols_set and b in cols_set and c in cols_set:
        return [a, b, c]
    return None


def load_simscape_trajectory_csv(path: str | Path) -> SkeletonTrajectory:
    """Load a Simscape forward-dynamics output CSV into a SkeletonTrajectory.

    Accepts both column conventions (short ``club_head_X`` and long
    ``ClubLogs_CHGlobalPosition_1``).  The CSV must have a ``time`` column.
    Synthesizes ``mp`` (mid-hands) and ``torso`` when missing.
    """
    path = Path(path)
    df = pd.read_csv(path)
    if "time" not in df.columns:
        raise ValueError(f"CSV missing 'time' column: {path}")
    cols = list(df.columns)

    resolved: dict[str, list[str]] = {}
    for short, candidates in _TRAJECTORY_COLUMN_MAP.items():
        for cand in candidates:
            xyz = _xyz_columns_for(cols, cand)
            if xyz is not None:
                resolved[short] = xyz
                break
    for short, long_stem in _TRAJECTORY_LONG_FORM.items():
        if short in resolved:
            continue
        xyz = _xyz_columns_for(cols, long_stem)
        if xyz is not None:
            resolved[short] = xyz

    if not resolved:
        raise ValueError(
            f"CSV {path} has no recognised joint columns. Expected either "
            "'<joint>_X/Y/Z' (short) or '<Joint>Logs_...Global..._1/2/3' (long)."
        )

    times = df["time"].astype(float).to_numpy()
    frames: list[Skeleton] = []
    n = len(df)
    for i in range(n):
        row = df.iloc[i]
        joints: dict[str, np.ndarray] = {}
        for short, xyz in resolved.items():
            v = np.array([float(row[c]) for c in xyz])
            if np.all(np.isfinite(v)):
                joints[short] = v
        if "lw" in joints and "rw" in joints:
            joints["mp"] = (joints["lw"] + joints["rw"]) / 2.0
            joints["butt"] = joints["mp"].copy()
        if "torso" not in joints and "spine" in joints and "hub" in joints:
            joints["torso"] = joints["spine"] + 0.2 * (joints["hub"] - joints["spine"])
        frames.append(
            Skeleton(
                name=f"trajectory[{i}]", joints=joints, segments=list(FALLBACK_SEGMENTS)
            )
        )

    return SkeletonTrajectory(times=times, frames=frames, source_path=str(path))


# ----------------------------------------------------------------------------
# Shaft-snap math
# ----------------------------------------------------------------------------


def solve_shaft_rz_deg(
    mp_target: np.ndarray,
    ch_target: np.ndarray,
    mp_skel: np.ndarray,
    ch_skel: np.ndarray,
) -> float:
    """Return the Rz angle (degrees, wrapped to [-180,180]) that maps the
    skeleton shaft direction (mp_skel→ch_skel) to the target shaft
    direction (mp_target→ch_target) projected onto the XY plane.

    Returns 0.0 if either projected shaft has near-zero magnitude.
    """
    shaft_t_xy = (np.asarray(ch_target) - np.asarray(mp_target))[:2]
    shaft_m_xy = (np.asarray(ch_skel) - np.asarray(mp_skel))[:2]
    nt = float(np.linalg.norm(shaft_t_xy))
    nm = float(np.linalg.norm(shaft_m_xy))
    if nt < 1e-9 or nm < 1e-9:
        return 0.0
    a_t = float(np.arctan2(shaft_t_xy[1], shaft_t_xy[0]))
    a_m = float(np.arctan2(shaft_m_xy[1], shaft_m_xy[0]))
    rz = float(np.degrees(a_t - a_m))
    return ((rz + 180.0) % 360.0) - 180.0


# ---------------------------------------------------------------------------
# Multi-source target dispatch (issue #4480)
# ---------------------------------------------------------------------------


def clubtarget_from_multi(target: MultiSourceTarget) -> ClubTarget | None:
    """Return a plain ``ClubTarget`` view of the club slot, if any.

    The matcher's existing rendering path takes a ``ClubTarget``-style
    object (.butt / .clubhead / .club_quat / .time / .impact_idx).  A
    ``ClubBallTarget`` exposes the same shape via composition (it has
    a ``club: ClubTarget`` attribute when the dependency lands).  When
    the dependency hasn't landed, the duck-typed slot itself satisfies
    the read-only contract, so we return it as-is.

    Returns ``None`` when the multi-source target has no club slot.
    """
    if not target.has_club():
        return None
    club_slot = target.club
    # ClubBallTarget composes a ClubTarget under .club; prefer that when present.
    inner = getattr(club_slot, "club", None)
    if isinstance(inner, ClubTarget):
        return inner
    return club_slot if isinstance(club_slot, ClubTarget) else club_slot


def dispatch_cost_inputs(target: MultiSourceTarget) -> dict[str, object]:
    """Adapter that exposes a per-source dict for cost-function callers.

    Cost terms today consume a single ``ClubTarget``.  As the body /
    ball-aware cost terms come online (issues #4476, #4479) they will
    select on ``has_body()`` / ``is_club_ball()``.  This helper centralises
    the dispatch so callers don't sprinkle ``hasattr`` checks across
    the codebase.
    """
    out: dict[str, object] = {"time": target.shared_time()}
    if target.has_club():
        out["club"] = clubtarget_from_multi(target)
        out["has_ball"] = target.is_club_ball()
    if target.has_body():
        out["body"] = target.body
    return out
