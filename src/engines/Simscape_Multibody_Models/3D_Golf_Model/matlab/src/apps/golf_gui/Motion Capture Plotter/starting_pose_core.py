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

# Phase windows for trace display.
PHASE_WINDOWS: dict[str, tuple[str | None, str | None]] = {
    "None":                   (None, None),
    "Backswing (A → T)":      ("A", "T"),
    "Downswing (T → I)":      ("T", "I"),
    "Follow-through (I → F)": ("I", "F"),
    "Full swing (A → F)":     ("A", "F"),
    "Manual range":           ("manual", "manual"),
}
DEFAULT_PHASE = "Full swing (A → F)"


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
class PoseSlot:
    name: str
    skeleton: Skeleton
    color: str
    mocap_color: str
    target_event: str
    visible: bool = True


# ----------------------------------------------------------------------------
# Skeleton fallback
# ----------------------------------------------------------------------------


FALLBACK_IMPACT: dict[str, list[float]] = {
    "hip":   [0.00, -0.30, 0.95], "spine": [0.00, -0.30, 1.20],
    "hub":   [0.00, -0.30, 1.40], "ls":    [-0.20, -0.30, 1.40],
    "rs":    [0.20, -0.30, 1.40], "le":    [-0.10, -0.20, 1.10],
    "re":    [0.10, -0.20, 1.10], "lw":    [-0.05, -0.10, 0.80],
    "rw":    [0.05, -0.10, 0.80], "mp":    [0.00, -0.10, 0.80],
    "ch":    [0.00, 0.10, 0.10],
}
FALLBACK_TOB: dict[str, list[float]] = {
    "hip":   [0.00, -0.30, 0.95], "spine": [0.00, -0.30, 1.20],
    "hub":   [0.00, -0.30, 1.40], "ls":    [-0.20, -0.30, 1.40],
    "rs":    [0.20, -0.30, 1.40], "le":    [-0.05, -0.10, 1.55],
    "re":    [0.30, -0.10, 1.50], "lw":    [0.10,  0.10, 1.85],
    "rw":    [0.20,  0.10, 1.80], "mp":    [0.15,  0.10, 1.82],
    "ch":    [-0.40, 0.40, 1.60],
}
FALLBACK_SEGMENTS: list[tuple[str, str]] = [
    ("hip", "spine"), ("spine", "hub"), ("hub", "ls"), ("hub", "rs"),
    ("ls", "le"), ("rs", "re"), ("le", "lw"), ("re", "rw"),
    ("lw", "mp"), ("rw", "mp"), ("mp", "ch"),
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
