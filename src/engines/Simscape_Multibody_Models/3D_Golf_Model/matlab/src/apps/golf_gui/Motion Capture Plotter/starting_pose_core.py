"""Pure-data + math core for the Starting-Pose Matcher.

This module deliberately contains NO Qt / matplotlib imports so it can be
unit-tested in environments that don't have the GUI stack working.  The
matcher (`starting_pose_matcher.py`) imports everything from here.

Public API:

    Constants
    ---------
    CM_TO_M, SESSION_SCHEMA_VERSION
    EVENT_KEYS, EVENT_LABEL_PRESETS, DEFAULT_EVENT_PRESET
    PHASE_WINDOWS, DEFAULT_PHASE
    FALLBACK_IMPACT, FALLBACK_TOB, FALLBACK_SEGMENTS

    Dataclasses
    -----------
    MocapEvents, Skeleton, RigidTransform, PoseSlot

    Functions
    ---------
    load_mocap_xlsx(path, sheet) -> DataFrame  (cm → m units)
    read_event_header(path, sheet) -> MocapEvents
    load_skeleton(json_path, fallback_pose) -> Skeleton
    solve_shaft_rz_deg(mp_target, ch_target, mp_skel, ch_skel) -> float
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# Units: Wiffle xlsx positions are in CM.  See MATLAB_GOLF_MODEL_GUIDE.md.
# ----------------------------------------------------------------------------
CM_TO_M = 0.01

# Schema version for session JSON
SESSION_SCHEMA_VERSION = 2

# Event-label conventions.
EVENT_KEYS: tuple[str, ...] = ("A", "T", "I", "F")
EVENT_LABEL_PRESETS: dict[str, dict[str, str]] = {
    "Wiffle (A/T/I/F)": {
        "A": "Address", "T": "Top of Backswing",
        "I": "Impact",  "F": "Finish",
    },
    "Trackman P-system": {
        "A": "P1 Address", "T": "P4 Top",
        "I": "P7 Impact",  "F": "P10 Finish",
    },
    "Plain English": {
        "A": "Setup", "T": "Backswing top",
        "I": "Strike", "F": "Follow-through end",
    },
    "Sequence numbers": {
        "A": "Phase 1", "T": "Phase 2",
        "I": "Phase 3", "F": "Phase 4",
    },
}
DEFAULT_EVENT_PRESET = "Wiffle (A/T/I/F)"

# Phase windows.  Keys are LOGICAL identifiers (stable, not user-facing); the
# user-facing display label is built from the current event labels via
# `phase_display_label()` so the combo always shows fully spelled-out names
# like "Backswing (Address to Top of Backswing)" instead of "A → T".
PHASE_KEYS: tuple[str, ...] = (
    "none", "backswing", "downswing", "follow_through", "full_swing", "manual",
)
PHASE_BOUNDS: dict[str, tuple[str | None, str | None]] = {
    "none":           (None, None),
    "backswing":      ("A", "T"),
    "downswing":      ("T", "I"),
    "follow_through": ("I", "F"),
    "full_swing":     ("A", "F"),
    "manual":         ("manual", "manual"),
}
DEFAULT_PHASE = "full_swing"

# Backwards-compatible alias used by older sessions.  Maps the v1 display
# label back to the v2 logical key so a session.json from before this
# refactor still loads.
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
        "none":           "None - draw entire data range",
        "backswing":      f"Backswing ({a} to {t})",
        "downswing":      f"Downswing ({t} to {i})",
        "follow_through": f"Follow-through ({i} to {f})",
        "full_swing":     f"Full swing ({a} to {f})",
        "manual":         "Manual frame range",
    }
    return table.get(key, key)


def phase_key_from_label(label: str) -> str | None:
    """Look up a phase key from a (legacy or current) display label.

    Used by session-load to resolve old "Backswing (A → T)" strings to the
    new "backswing" key.  Returns None if no match.
    """
    if label in PHASE_LEGACY_LABELS:
        return PHASE_LEGACY_LABELS[label]
    if label in PHASE_BOUNDS:
        return label
    # Fuzzy match: try matching by leading word
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
        """Return 0-based frame index for label A/T/I/F, or None if NaN."""
        v = getattr(self, f"{label}_sample", float("nan"))
        if v != v:  # NaN check
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
    """Time series of skeleton joint positions (one Skeleton per frame).

    Loaded from a Simscape CSV via :func:`load_simscape_trajectory_csv`.
    Used by the matcher for skeleton playback (animating the model's
    forward dynamics output instead of just showing one static pose).
    """
    times: np.ndarray = field(default_factory=lambda: np.zeros(0))
    frames: list[Skeleton] = field(default_factory=list)
    source_path: str = ""

    def __len__(self) -> int:
        return len(self.frames)

    def frame_at_time(self, t: float) -> int:
        """Return the frame index closest to time ``t`` (clamped to range)."""
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
    trajectory_frame_index: int = 0  # used when playback target == "Skeleton"/"Both"


# ----------------------------------------------------------------------------
# Skeleton fallback
# ----------------------------------------------------------------------------


# The Simscape model body chain, per GolfSwing3D_Kinetic.mdl:
#
#   hip  --[gimbal X/Y/Z]-->  spine  --[universal X/Y tilt]-->  torso
#                                      --[revolute Z twist]-->  hub
#                                      --[universal X/Y]----->  lscap, rscap
#                                                            -->  ls, rs
#                                                            -->  le, re
#                                                            -->  lw, rw
#
# Geometric segment lengths from the model parameters:
#   UpperTorsoLength ≈ 0.305 m (12 in), split as 0.2/0.8 by UpperTorsoBase /
#   UpperTorsoTop cylinders.  We model the chain at three torso landmarks:
#     spine = hip + UpperTorsoLength/2 along (rotated) +Z   (lower back)
#     torso = spine + 0.2 * UpperTorsoLength * +Z           (between disks)
#     hub   = torso + 0.8 * UpperTorsoLength * +Z           (chest level)
#   HubtoSLength ≈ 0.254 m (10 in) shoulder offset.
#
# At Impact: shoulders roughly square to ball; at Top of Backswing: torso
# coiled ~90° clockwise (right-handed swing) so right shoulder is BEHIND
# the body and left shoulder is in FRONT.
FALLBACK_IMPACT: dict[str, list[float]] = {
    "hip":   [0.00, -0.30, 0.95],
    "spine": [0.00, -0.30, 1.20],
    "torso": [0.00, -0.30, 1.27],   # ~20% up the upper torso (twist joint)
    "hub":   [0.00, -0.30, 1.45],
    "ls":    [-0.25, -0.30, 1.42],   # left shoulder -X (target side)
    "rs":    [0.25, -0.30, 1.42],    # right shoulder +X
    "le":    [-0.20, -0.20, 1.15],
    "re":    [0.10, -0.20, 1.10],
    "lw":    [-0.05, -0.10, 0.85],
    "rw":    [0.05, -0.10, 0.85],
    "mp":    [0.00, -0.10, 0.85],
    "ch":    [0.10, 0.10, 0.05],     # clubhead near ball
}
# Top of Backswing: hip nearly square, but shoulders rotated ~90° about
# the body Z axis (torso disk twist).  Right shoulder pulled BACK (-Y),
# left shoulder pushed FORWARD (+Y).  Hands lifted high & behind.
FALLBACK_TOB: dict[str, list[float]] = {
    "hip":   [0.00, -0.30, 0.95],
    "spine": [0.00, -0.30, 1.20],
    "torso": [0.00, -0.30, 1.27],
    "hub":   [0.00, -0.30, 1.45],
    "ls":    [0.00, -0.05, 1.42],    # rotated +X shoulder line about Z by ~90°:
    "rs":    [0.00, -0.55, 1.42],    #   left now FORWARD, right now BACK
    "le":    [0.10, +0.05, 1.55],
    "re":    [-0.05, -0.55, 1.30],
    "lw":    [0.20,  0.10, 1.85],
    "rw":    [0.18,  0.05, 1.82],
    "mp":    [0.19,  0.08, 1.83],
    "ch":    [-0.30, 0.40, 1.65],
}
# Skeleton segments — connect joints into a body chain for plotting.
# Note that "torso" sits between spine and hub on the central column.
FALLBACK_SEGMENTS: list[tuple[str, str]] = [
    ("hip",   "spine"),
    ("spine", "torso"),
    ("torso", "hub"),
    ("hub",   "ls"),
    ("hub",   "rs"),
    ("ls",    "le"),
    ("rs",    "re"),
    ("le",    "lw"),
    ("re",    "rw"),
    ("lw",    "mp"),
    ("rw",    "mp"),
    ("mp",    "ch"),
]


# ----------------------------------------------------------------------------
# xlsx loaders (CORRECT units — bypasses buggy legacy mocap_data_loader)
# ----------------------------------------------------------------------------


def load_mocap_xlsx(xlsx_path: str | Path, sheet_name: str) -> pd.DataFrame:
    """Load a Wiffle xlsx sheet into a DataFrame in metres.

    Schema (subset): time (s), mid_X/Y/Z (m), club_X/Y/Z (m).
    """
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name, header=None)
    if len(df) <= 3:
        raise ValueError(f"Sheet '{sheet_name}' has no data rows")
    rows: list[dict[str, float]] = []
    for i in range(3, len(df)):
        row = df.iloc[i]
        if len(row) < 17:
            continue
        try:
            time = float(row[1]) if not pd.isna(row[1]) else float(i - 3)
            rec = {
                "time":   time,
                "mid_X":  _safe(row, 2)  * CM_TO_M,
                "mid_Y":  _safe(row, 3)  * CM_TO_M,
                "mid_Z":  _safe(row, 4)  * CM_TO_M,
                "club_X": _safe(row, 14) * CM_TO_M,
                "club_Y": _safe(row, 15) * CM_TO_M,
                "club_Z": _safe(row, 16) * CM_TO_M,
            }
        except (ValueError, TypeError):
            continue
        rows.append(rec)
    return pd.DataFrame(rows)


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
    """Parse the row-1 event-marker band: A=<n> T=<n> I=<n> F=<n> CHS=<mph>."""
    ev = MocapEvents()
    try:
        row1 = pd.read_excel(xlsx_path, sheet_name=sheet_name, header=None, nrows=1)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not read event header: %s", exc)
        return ev
    label_to_field = {"A": "A_sample", "T": "T_sample", "I": "I_sample",
                      "F": "F_sample", "CHS": "CHS_mph"}
    for c in range(row1.shape[1] - 1):
        cell = row1.iat[0, c]
        if pd.isna(cell):
            continue
        label = str(cell).strip()
        if label not in label_to_field:
            continue
        val = row1.iat[0, c + 1]
        if pd.isna(val):
            continue
        try:
            setattr(ev, label_to_field[label], float(val))
        except (ValueError, TypeError):
            continue
    return ev


# ----------------------------------------------------------------------------
# Skeleton loader
# ----------------------------------------------------------------------------


def load_skeleton(json_path: str | Path, fallback_pose: str = "Impact") -> Skeleton:
    """Load skeleton from JSON; fall back to hardcoded approximate pose."""
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
        return Skeleton(name=data.get("pose", fallback_pose),
                        joints=joints,
                        segments=segments or list(FALLBACK_SEGMENTS))
    logger.warning(
        "%s not found - using fallback %s pose. Run "
        "export_default_skeleton('%s') in MATLAB for real joints.",
        json_path, fallback_pose, fallback_pose,
    )
    pose = FALLBACK_TOB if fallback_pose.lower().startswith("top") else FALLBACK_IMPACT
    return Skeleton(
        name=fallback_pose,
        joints={k: np.array(v, dtype=float) for k, v in pose.items()},
        segments=list(FALLBACK_SEGMENTS),
    )


# ----------------------------------------------------------------------------
# Skeleton-trajectory loader (Simscape CSV)
# ----------------------------------------------------------------------------


# Map "<our short name>" -> list of CSV column-name candidates for X.  Each
# candidate's _Y/_Z (or _2/_3, _y/_z) is auto-derived.
_TRAJECTORY_COLUMN_MAP: dict[str, list[str]] = {
    # Short-form columns used by motion_capture_plotter_data.parse_simscape_csv
    "ch":    ["club_head_X",      "club_head_x"],
    "lw":    ["left_hand_X",      "left_hand_x"],
    "rw":    ["right_hand_X",     "right_hand_x"],
    "ls":    ["left_shoulder_X",  "left_shoulder_x"],
    "rs":    ["right_shoulder_X", "right_shoulder_x"],
    "le":    ["left_elbow_X",     "left_elbow_x"],
    "re":    ["right_elbow_X",    "right_elbow_x"],
    "hub":   ["hub_X",            "hub_x"],
    "torso": ["torso_X",          "torso_x"],   # may be missing — synthesized
    "spine": ["spine_X",          "spine_x"],
    "hip":   ["hip_X",            "hip_x"],
}

# Long-form (raw Simscape bus) columns.  Same short-name keys.
_TRAJECTORY_LONG_FORM: dict[str, str] = {
    "ch":    "ClubLogs_CHGlobalPosition_1",
    "lw":    "LWLogs_LHGlobalPosition_1",
    "rw":    "RWLogs_RHGlobalPosition_1",
    "ls":    "LSLogs_GlobalPosition_1",
    "rs":    "RSLogs_GlobalPosition_1",
    "le":    "LELogs_LArmonLForearmFGlobal_1",
    "re":    "RELogs_RArmonLForearmFGlobal_1",
    "hub":   "HipLogs_HUBGlobalPosition_1",
    "torso": "TorsoLogs_GlobalPosition_1",
    "spine": "SpineLogs_GlobalPosition_1",
    "hip":   "HipLogs_HipGlobalPosition_dim1",
}


def _xyz_columns_for(df_columns: list[str], stem: str) -> list[str] | None:
    """Resolve the X/Y/Z column names for a stem like 'club_head_X' or
    'ClubLogs_CHGlobalPosition_1'.  Returns None when any of the three is
    missing.
    """
    cols_set = set(df_columns)
    if stem.endswith("_X"):
        a, b, c = stem, stem[:-1] + "Y", stem[:-1] + "Z"
    elif stem.endswith("_x"):
        a, b, c = stem, stem[:-1] + "y", stem[:-1] + "z"
    elif stem.endswith("_1"):
        a, b, c = stem, stem[:-1] + "2", stem[:-1] + "3"
    elif stem.endswith("_dim1"):
        a, b, c = stem, stem[:-1] + "2", stem[:-1] + "3"
    else:
        return None
    if a in cols_set and b in cols_set and c in cols_set:
        return [a, b, c]
    return None


def load_simscape_trajectory_csv(path: str | Path) -> SkeletonTrajectory:
    """Load a Simscape forward-dynamics output CSV into a SkeletonTrajectory.

    Accepts both column conventions:
      * Short (motion_capture_plotter): ``club_head_X``, ``left_hand_X``, etc.
      * Long (raw Simscape bus): ``ClubLogs_CHGlobalPosition_1``, etc.

    The CSV must have a ``time`` column; rows are mapped to one
    :class:`Skeleton` each.  Any joint whose columns are missing is simply
    omitted from that frame's joint dict.

    Synthesizes ``mp`` (mid-hands) = (lw + rw) / 2 and aliases ``butt`` = mp
    when both hand columns are present.
    """
    path = Path(path)
    df = pd.read_csv(path)
    if "time" not in df.columns:
        raise ValueError(f"CSV missing 'time' column: {path}")
    cols = list(df.columns)

    # Resolve which stem to use for each joint.
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
            "'<joint>_X/Y/Z' (short) or '<Joint>Logs_...Global..._1/2/3' (long).")

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
        # Synthesize mp from lw + rw if available.
        if "lw" in joints and "rw" in joints:
            joints["mp"] = (joints["lw"] + joints["rw"]) / 2.0
            joints["butt"] = joints["mp"].copy()
        # Synthesize torso between spine and hub if missing — places the
        # twist-disk landmark roughly 20% up the upper torso (matching
        # UpperTorsoBase = 0.2 * UpperTorsoLength in the .mdl).
        if "torso" not in joints and "spine" in joints and "hub" in joints:
            joints["torso"] = joints["spine"] + 0.2 * (joints["hub"]
                                                       - joints["spine"])
        frames.append(Skeleton(name=f"trajectory[{i}]", joints=joints,
                               segments=list(FALLBACK_SEGMENTS)))

    return SkeletonTrajectory(times=times, frames=frames, source_path=str(path))


# ----------------------------------------------------------------------------
# Shaft-snap math
# ----------------------------------------------------------------------------


def solve_shaft_rz_deg(mp_target: np.ndarray, ch_target: np.ndarray,
                       mp_skel: np.ndarray, ch_skel: np.ndarray) -> float:
    """Return the Rz angle (degrees, wrapped to [-180,180]) that maps the
    skeleton shaft direction (mp_skel→ch_skel) to the target shaft
    direction (mp_target→ch_target) projected onto the XY plane.

    Returns 0.0 if either projected shaft has near-zero magnitude (which
    happens for a perfectly vertical shaft — Rz is undefined and the
    caller should warn / not adjust).
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
