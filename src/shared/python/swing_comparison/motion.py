"""Engine-independent swing motion and capture-A marker schema (#11164).

Provides SwingMotion, the frozen motion record every metric reads, and
swing_motion_from_markers, which builds one from capture-A marker labels.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from src.shared.python.contracts import require
from src.shared.python.swing_comparison.events import _fill_nans_3d

# ---------------------------------------------------------------------------
# Marker Schema Constants (discovered from data/C3D_TA_Driver.c3d)
# ---------------------------------------------------------------------------

CAPTURE_A_MARKER_LABELS: tuple[str, ...] = (
    "Marker_0:0:0",
    "WaistLeft",
    "WaistRight",
    "WaistLBack",
    "WaistRBack",
    "BackTop",
    "BackLeft",
    "BackRight",
    "HeadTop",
    "HeadFront",
    "HeadSide",
    "LShoulderTop",
    "LShoulderBack",
    "LElbowOut",
    "LUArmHigh",
    "LWristTop",
    "RShoulderTop",
    "RShoulderBack",
    "RElbowOut",
    "RUArmHigh",
    "RWristTop",
    "LKneeOut",
    "LToeIn",
    "LToeOut",
    "LAnkleOut",
    "RKneeOut",
    "RToeIn",
    "RToeOut",
    "RAnkleOut",
    "Marker_2:2:1",
    "Marker_2:2:2",
    "Marker_2:2:3",
    "Marker_3:3:1",
    "Marker_3:3:2",
    "Marker_3:3:3",
    "Uname*36",
    "Uname*37",
    "Uname*38",
)

CAPTURE_A_CLUBHEAD_LABELS: tuple[str, ...] = (
    "Marker_2:2:1",
    "Marker_2:2:2",
    "Marker_2:2:3",
)

CAPTURE_A_GRIP_LABELS: tuple[str, ...] = (
    "Marker_3:3:1",
    "Marker_3:3:2",
    "Marker_3:3:3",
)

CAPTURE_A_PELVIS_LEFT_LABELS: tuple[str, ...] = ("WaistLeft", "WaistLBack")
CAPTURE_A_PELVIS_RIGHT_LABELS: tuple[str, ...] = ("WaistRight", "WaistRBack")
# Trunk (thorax) markers: fixed to the rib cage, unlike the acromion markers
# below whose line keeps rotating with the arms through follow-through.
CAPTURE_A_TRUNK_LEFT_LABELS: tuple[str, ...] = ("BackLeft",)
CAPTURE_A_TRUNK_RIGHT_LABELS: tuple[str, ...] = ("BackRight",)
CAPTURE_A_SHOULDER_LEFT_LABELS: tuple[str, ...] = ("LShoulderBack", "LShoulderTop")
CAPTURE_A_SHOULDER_RIGHT_LABELS: tuple[str, ...] = ("RShoulderBack", "RShoulderTop")


# ---------------------------------------------------------------------------
# SwingMotion Dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SwingMotion:
    """Engine-independent swing motion container.

    Attributes:
        t: Time array of shape (N,) in seconds. Must be strictly monotonically increasing.
        markers: Dictionary mapping marker name to (N, 3) 3D coordinate array in meters,
                 in a right-handed world frame with Z up.
        club_head: Optional clubhead position array of shape (N, 3) in meters.
        grip: Optional grip / butt position array of shape (N, 3) in meters.
        shaft_axis: Optional unit vector along the shaft of shape (N, 3).
        face_normal: Optional unit vector pointing outward from the clubface of shape (N, 3).
    """

    t: np.ndarray
    markers: dict[str, np.ndarray]
    club_head: np.ndarray | None = None
    grip: np.ndarray | None = None
    shaft_axis: np.ndarray | None = None
    face_normal: np.ndarray | None = None

    def __post_init__(self) -> None:
        require(
            isinstance(self.t, (np.ndarray, Sequence)), "t must be an array or sequence"
        )
        t_arr = np.asarray(self.t, dtype=np.float64)
        require(t_arr.ndim == 1, "t must be a 1-D array", t_arr.ndim)
        require(t_arr.size >= 4, "t must contain at least 4 frames", t_arr.size)
        require(bool(np.all(np.isfinite(t_arr))), "t must contain finite values")
        require(
            bool(np.all(np.diff(t_arr) > 0)),
            "t must be strictly monotonically increasing",
        )
        object.__setattr__(self, "t", t_arr)

        n_frames = t_arr.size
        require(isinstance(self.markers, dict), "markers must be a dictionary")

        clean_markers: dict[str, np.ndarray] = {}
        for name, m_pos in self.markers.items():
            arr = np.asarray(m_pos, dtype=np.float64)
            require(
                arr.shape == (n_frames, 3),
                f"Marker '{name}' must have shape ({n_frames}, 3), got {arr.shape}",
            )
            clean_markers[str(name)] = arr
        object.__setattr__(self, "markers", clean_markers)

        for opt_name in ("club_head", "grip", "shaft_axis", "face_normal"):
            val = getattr(self, opt_name)
            if val is not None:
                arr = np.asarray(val, dtype=np.float64)
                require(
                    arr.shape == (n_frames, 3),
                    f"{opt_name} must have shape ({n_frames}, 3), got {arr.shape}",
                )
                object.__setattr__(self, opt_name, arr)


def swing_motion_from_markers(
    t: np.ndarray | Sequence[float],
    markers: dict[str, np.ndarray],
    *,
    club_head: np.ndarray | None = None,
    grip: np.ndarray | None = None,
    shaft_axis: np.ndarray | None = None,
    face_normal: np.ndarray | None = None,
) -> SwingMotion:
    """Construct a SwingMotion from marker dict using capture-A naming.

    If club_head, grip, or shaft_axis are omitted, they are automatically
    derived from canonical capture-A marker clusters when present.

    Args:
        t: Time array in seconds, shape (N,).
        markers: Dictionary mapping marker name to (N, 3) coordinates in meters (Z up).
        club_head: Optional explicit clubhead position array of shape (N, 3).
        grip: Optional explicit grip position array of shape (N, 3).
        shaft_axis: Optional explicit shaft axis array of shape (N, 3).
        face_normal: Optional explicit face normal array of shape (N, 3).

    Returns:
        Configured SwingMotion instance.
    """
    derived_club_head = club_head
    if derived_club_head is None:
        head_pts = [
            markers[lbl]
            for lbl in CAPTURE_A_CLUBHEAD_LABELS
            if lbl in markers and not np.all(np.isnan(markers[lbl]))
        ]
        if head_pts:
            stacked = np.stack(head_pts, axis=0)  # (M, N, 3)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                derived_club_head = np.nanmean(stacked, axis=0)  # (N, 3)
            derived_club_head = _fill_nans_3d(derived_club_head)

    derived_grip = grip
    if derived_grip is None:
        grip_pts = [
            markers[lbl]
            for lbl in CAPTURE_A_GRIP_LABELS
            if lbl in markers and not np.all(np.isnan(markers[lbl]))
        ]
        if grip_pts:
            stacked = np.stack(grip_pts, axis=0)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                derived_grip = np.nanmean(stacked, axis=0)
            derived_grip = _fill_nans_3d(derived_grip)

    derived_shaft_axis = shaft_axis
    if (
        derived_shaft_axis is None
        and derived_club_head is not None
        and derived_grip is not None
    ):
        diff = derived_club_head - derived_grip
        norms = np.linalg.norm(diff, axis=-1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        derived_shaft_axis = diff / norms

    return SwingMotion(
        t=np.asarray(t, dtype=np.float64),
        markers=markers,
        club_head=derived_club_head,
        grip=derived_grip,
        shaft_axis=derived_shaft_axis,
        face_normal=face_normal,
    )
