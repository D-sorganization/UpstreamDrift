"""Starting-pose matcher: align Simscape golfer skeleton to mocap targets.

A focused, professional-grade alignment tool that lets you place the
Simscape model in the right starting pose BEFORE running any optimiser.

Why: starting fmincon at zero-theta dropped it into a bad local minimum
because the model started in the wrong pose. This tool produces the
seed (rigid transform + scale) that fit_swing_full_pipeline uses as
input_overrides for the model workspace.

Workflow:
    1. Loads the Wiffle ProV1 motion-capture xlsx.
       NOTE: Wiffle xlsx positions are in CENTIMETRES — see
       MATLAB_GOLF_MODEL_GUIDE.md.  We bypass the legacy
       mocap_data_loader.py (which uses the wrong inches→m factor).
    2. Reads the row-1 event header (A=address, T=top, I=impact, F=finish).
    3. Loads up to two pose skeletons (TopofBackswing + Impact) from
       simscape_skeleton_<pose>.json (produced by export_default_skeleton.m).
       Falls back to a hardcoded approximate pose if absent.
    4. A 7-DOF transform (Tx/Ty/Tz/Rx/Ry/Rz/Scale) applies to all visible
       skeletons.  Rx/Ry are LOCKED by default (both data and model use
       Z-up; the only physical DoF that matters is global heading via Rz
       plus a translation).  Unlock with the checkbox if needed.
    5. Two-point shaft snap: solves Rz + Tx/Ty/Tz so the SHAFT (mid-hands
       to clubhead vector) of the model pose aligns with the mocap shaft
       at the chosen event frame — not just the mid-hands point.
    6. Save offsets to JSON; later it seeds model-workspace overrides in
       fit_swing_full_pipeline.

Run:
    cd ".../Motion Capture Plotter"
    python -m starting_pose_matcher
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure
from PyQt6.QtCore import QSignalBlocker, Qt, QTimer
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QStyleFactory,
    QVBoxLayout,
    QWidget,
)

# Schema version for session JSON
_SESSION_SCHEMA_VERSION = 1

# Phase window options.  Maps display label -> (event_start, event_end) where
# either side may be None for manual / full-data.
_PHASE_WINDOWS: dict[str, tuple[str | None, str | None]] = {
    "None":                  (None, None),
    "Backswing (A → T)":     ("A", "T"),
    "Downswing (T → I)":     ("T", "I"),
    "Follow-through (I → F)":("I", "F"),
    "Full swing (A → F)":    ("A", "F"),
    "Manual range":          ("manual", "manual"),
}
_DEFAULT_PHASE = "Full swing (A → F)"

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")

# ----------------------------------------------------------------------------
# Units: Wiffle xlsx positions are in CM.  See MATLAB_GOLF_MODEL_GUIDE.md.
# ----------------------------------------------------------------------------
CM_TO_M = 0.01

# Default camera presets (elev, azim) for matplotlib's 3D view_init.
_CAMERA_PRESETS: dict[str, tuple[float, float]] = {
    "Face-On":     (10.0, -90.0),  # facing the golfer along +Y
    "Down-Line":   (10.0,   0.0),  # behind the ball, looking down +X target line
    "Top-Down":    (89.0, -90.0),
    "Isometric":   (20.0, -55.0),
    "Reset":       (15.0, -60.0),
}
_DEFAULT_CAMERA = "Reset"


# --------------------------------------------------------------------------- #
# Stylesheet                                                                  #
# --------------------------------------------------------------------------- #


_QSS = """
QMainWindow, QWidget {
    background: #2b2f36;
    color: #e6e6e6;
    font-family: "Segoe UI", "Helvetica Neue", sans-serif;
    font-size: 10pt;
}
QGroupBox {
    border: 1px solid #404652;
    border-radius: 6px;
    margin-top: 14px;
    padding: 10px 8px 8px 8px;
    background: #323742;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 2px 8px;
    margin-left: 8px;
    color: #f0c674;
    font-weight: bold;
    background: transparent;
}
QPushButton {
    background: #3b414e;
    border: 1px solid #535a69;
    border-radius: 4px;
    padding: 5px 10px;
    color: #e6e6e6;
}
QPushButton:hover { background: #475061; }
QPushButton:pressed { background: #2f3540; }
QPushButton:disabled { color: #6b7280; background: #2c303a; }
QPushButton#primary {
    background: #2563eb;
    border: 1px solid #1d4ed8;
    color: white;
    font-weight: bold;
}
QPushButton#primary:hover { background: #3b82f6; }
QPushButton#accent {
    background: #16a34a;
    border: 1px solid #15803d;
    color: white;
    font-weight: bold;
}
QPushButton#accent:hover { background: #22c55e; }
QPushButton#preset {
    background: #475261;
    border: 1px solid #5d6677;
    padding: 3px 6px;
    min-width: 48px;
}
QPushButton#preset:hover { background: #5a657a; }
QCheckBox { spacing: 6px; }
QCheckBox::indicator {
    width: 14px; height: 14px;
    border: 1px solid #6b7280;
    border-radius: 3px;
    background: #3b414e;
}
QCheckBox::indicator:checked {
    background: #2563eb; border-color: #1d4ed8;
}
QComboBox, QDoubleSpinBox {
    background: #1f242b;
    color: #f8fafc;
    border: 1px solid #535a69;
    border-radius: 4px;
    padding: 3px 6px;
    min-height: 20px;
    selection-background-color: #2563eb;
}
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {
    width: 14px;
}
QSlider::groove:horizontal {
    background: #1a1d23; height: 6px; border-radius: 3px;
}
QSlider::handle:horizontal {
    background: #2563eb;
    width: 12px; margin: -4px 0; border-radius: 6px;
}
QSlider::handle:horizontal:hover { background: #3b82f6; }
QSlider::handle:horizontal:disabled { background: #4b5563; }
QLabel#title {
    color: #f8fafc;
    font-size: 14pt;
    font-weight: bold;
    padding: 4px;
}
QLabel#status {
    color: #94a3b8;
    font-family: Consolas, "Courier New", monospace;
    font-size: 9pt;
}
QLabel#residual {
    color: #f8fafc;
    font-family: Consolas, "Courier New", monospace;
    font-size: 9pt;
    background: #1f242b;
    border: 1px solid #404652;
    border-radius: 4px;
    padding: 6px;
}
QFrame#sep {
    background: #404652;
    max-height: 1px; min-height: 1px;
}
QScrollArea { border: 0; }
"""


# --------------------------------------------------------------------------- #
# Data model                                                                  #
# --------------------------------------------------------------------------- #


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


# --------------------------------------------------------------------------- #
# xlsx loader (CORRECT units — bypasses buggy legacy mocap_data_loader)       #
# --------------------------------------------------------------------------- #


def load_mocap_xlsx(xlsx_path: str | Path, sheet_name: str) -> pd.DataFrame:
    """Load Wiffle xlsx -> DataFrame in METRES."""
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


# --------------------------------------------------------------------------- #
# Skeleton loader                                                             #
# --------------------------------------------------------------------------- #


_FALLBACK_IMPACT: dict[str, list[float]] = {
    "hip":   [0.00, -0.30, 0.95], "spine": [0.00, -0.30, 1.20],
    "hub":   [0.00, -0.30, 1.40], "ls":    [-0.20, -0.30, 1.40],
    "rs":    [0.20, -0.30, 1.40], "le":    [-0.10, -0.20, 1.10],
    "re":    [0.10, -0.20, 1.10], "lw":    [-0.05, -0.10, 0.80],
    "rw":    [0.05, -0.10, 0.80], "mp":    [0.00, -0.10, 0.80],
    "ch":    [0.00, 0.10, 0.10],
}
_FALLBACK_TOB: dict[str, list[float]] = {
    "hip":   [0.00, -0.30, 0.95], "spine": [0.00, -0.30, 1.20],
    "hub":   [0.00, -0.30, 1.40], "ls":    [-0.20, -0.30, 1.40],
    "rs":    [0.20, -0.30, 1.40], "le":    [-0.05, -0.10, 1.55],
    "re":    [0.30, -0.10, 1.50], "lw":    [0.10,  0.10, 1.85],
    "rw":    [0.20,  0.10, 1.80], "mp":    [0.15,  0.10, 1.82],
    "ch":    [-0.40, 0.40, 1.60],
}
_FALLBACK_SEGMENTS: list[tuple[str, str]] = [
    ("hip", "spine"), ("spine", "hub"), ("hub", "ls"), ("hub", "rs"),
    ("ls", "le"), ("rs", "re"), ("le", "lw"), ("re", "rw"),
    ("lw", "mp"), ("rw", "mp"), ("mp", "ch"),
]


def load_skeleton(json_path: str | Path, fallback_pose: str = "Impact") -> Skeleton:
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
                        segments=segments or list(_FALLBACK_SEGMENTS))
    logger.warning(
        "%s not found — using fallback %s pose. Run "
        "export_default_skeleton('%s') in MATLAB for real joints.",
        json_path, fallback_pose, fallback_pose,
    )
    pose = _FALLBACK_TOB if fallback_pose.lower().startswith("top") else _FALLBACK_IMPACT
    return Skeleton(
        name=fallback_pose,
        joints={k: np.array(v, dtype=float) for k, v in pose.items()},
        segments=list(_FALLBACK_SEGMENTS),
    )


# --------------------------------------------------------------------------- #
# UI helpers                                                                  #
# --------------------------------------------------------------------------- #


_T_RANGE = (-1500, 1500)
_R_RANGE = (-1800, 1800)
_S_RANGE = (50, 200)
_T_SCALE = 0.001
_R_SCALE = 0.1
_S_SCALE = 0.01


def _hsep() -> QFrame:
    f = QFrame()
    f.setObjectName("sep")
    f.setFrameShape(QFrame.Shape.HLine)
    return f


class LabelledControl(QWidget):
    """Spinbox + slider (slider follows spinbox).  Public API:
        .value()          -> float
        .set_value(v)
        .setEnabled(bool) -> grays out the whole row
        .valueChanged signal-like callback via spin.valueChanged
    """

    def __init__(self, label: str, units: str, slider_range: tuple[int, int],
                 scale: float, decimals: int, default: float = 0.0,
                 parent: QWidget | None = None):
        super().__init__(parent)
        self._scale = scale
        layout = QGridLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setHorizontalSpacing(8)

        lbl = QLabel(label)
        lbl.setMinimumWidth(28)
        layout.addWidget(lbl, 0, 0)

        self.spin = QDoubleSpinBox()
        self.spin.setDecimals(decimals)
        self.spin.setRange(slider_range[0] * scale, slider_range[1] * scale)
        self.spin.setSingleStep(scale * 10)  # arrow keys move by 10 ticks
        self.spin.setSuffix(f" {units}")
        self.spin.setMinimumWidth(110)
        self.spin.setKeyboardTracking(False)  # only commit on Enter / focus-out
        self.spin.setButtonSymbols(QDoubleSpinBox.ButtonSymbols.UpDownArrows)
        layout.addWidget(self.spin, 0, 1)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(*slider_range)
        self.slider.setMinimumWidth(160)
        layout.addWidget(self.slider, 0, 2)
        layout.setColumnStretch(2, 1)

        self.slider.valueChanged.connect(
            lambda v: self.spin.setValue(v * self._scale))
        self.spin.valueChanged.connect(
            lambda v: self.slider.setValue(int(round(v / self._scale))))
        self.spin.setValue(default)

    def value(self) -> float:
        return float(self.spin.value())

    def set_value(self, v: float) -> None:
        self.spin.setValue(v)

    def setEnabled(self, ok: bool) -> None:  # noqa: N802 (Qt name)
        self.spin.setEnabled(ok)
        self.slider.setEnabled(ok)
        super().setEnabled(True)  # keep label readable


# --------------------------------------------------------------------------- #
# Pose slot                                                                   #
# --------------------------------------------------------------------------- #


@dataclass
class PoseSlot:
    name: str
    skeleton: Skeleton
    color: str
    mocap_color: str
    target_event: str
    visible: bool = True


# --------------------------------------------------------------------------- #
# Main window                                                                 #
# --------------------------------------------------------------------------- #


class StartingPoseMatcher(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Starting-Pose Matcher")
        self.resize(1700, 1000)

        # Data
        self.df: pd.DataFrame | None = None
        self.events = MocapEvents()
        self._xlsx_path: str | None = None

        here = Path(__file__).parent
        self.poses: dict[str, PoseSlot] = {
            "TopofBackswing": PoseSlot(
                name="TopofBackswing",
                skeleton=load_skeleton(
                    here / "simscape_skeleton_TopofBackswing.json", "TopofBackswing"),
                color="#5b9eff", mocap_color="#ef4444",
                target_event="T",
            ),
            "Impact": PoseSlot(
                name="Impact",
                skeleton=load_skeleton(
                    here / "simscape_skeleton_Impact.json", "Impact"),
                color="#10b981", mocap_color="#f59e0b",
                target_event="I",
            ),
        }
        self.transform = RigidTransform()
        for slot in self.poses.values():
            if "hub" in slot.skeleton.joints:
                self.transform.pivot = tuple(slot.skeleton.joints["hub"])
                break

        self.show_clubhead_trace = False
        self.show_midhands_trace = False
        self.lock_xy_rotation = True   # Rx/Ry locked by default

        # Playback state
        self.current_frame: int = 0
        self.frame_override_active: bool = False  # use slider frame for mocap target?
        self.is_playing: bool = False
        self.loop_playback: bool = True
        self.event_overrides: dict[str, int] = {}  # user-set A/T/I/F sample numbers

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._advance_frame)

        # Phase window state
        self.phase_window: str = _DEFAULT_PHASE
        self.manual_window_start: int = 0
        self.manual_window_end: int = 0

        self._build_ui()
        self._apply_camera_preset(_DEFAULT_CAMERA)

        default_xlsx = Path(__file__).with_name("Wiffle_ProV1_club_3D_data.xlsx")
        if default_xlsx.exists():
            self._load_xlsx(str(default_xlsx))

    # ===================================================================== #
    # UI                                                                    #
    # ===================================================================== #

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        # ---------- LEFT: scrollable control column ---------------------- #
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMaximumWidth(490)
        scroll.setMinimumWidth(420)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        left = QWidget()
        scroll.setWidget(left)
        col = QVBoxLayout(left)
        col.setContentsMargins(6, 6, 6, 6)
        col.setSpacing(8)

        title = QLabel("Starting-Pose Matcher")
        title.setObjectName("title")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        col.addWidget(title)

        col.addWidget(self._build_file_box())
        col.addWidget(self._build_pose_box())
        col.addWidget(self._build_playback_box())
        col.addWidget(self._build_view_box())
        col.addWidget(self._build_align_box())
        col.addWidget(self._build_transform_box())
        col.addWidget(self._build_save_box())
        col.addStretch()

        root.addWidget(scroll)

        # ---------- RIGHT: plot column ----------------------------------- #
        plot_widget = QWidget()
        plot_layout = QVBoxLayout(plot_widget)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        plot_layout.setSpacing(0)

        self.fig = Figure(figsize=(10, 8), dpi=100, facecolor="#1f242b")
        self.ax = self.fig.add_subplot(111, projection="3d", facecolor="#1f242b")
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding,
                                  QSizePolicy.Policy.Expanding)
        self.toolbar = NavigationToolbar(self.canvas, plot_widget)
        self.toolbar.setStyleSheet("background:#2b2f36;color:#e6e6e6;")
        plot_layout.addWidget(self.toolbar)
        plot_layout.addWidget(self.canvas)

        root.addWidget(plot_widget, stretch=1)

        self._setup_axes()

    # ---------- builders --------------------------------------------------- #

    def _build_file_box(self) -> QGroupBox:
        box = QGroupBox("Mocap Source")
        gl = QGridLayout(box)
        gl.setVerticalSpacing(6)
        self.btn_load = QPushButton("Load xlsx…")
        self.btn_load.clicked.connect(self._on_load_clicked)
        gl.addWidget(self.btn_load, 0, 0, 1, 2)
        gl.addWidget(QLabel("Sheet:"), 1, 0)
        self.sheet_combo = QComboBox()
        self.sheet_combo.addItems(["TW_ProV1", "TW_wiffle", "GW_wiffle", "GW_ProV11"])
        self.sheet_combo.currentTextChanged.connect(self._on_sheet_changed)
        gl.addWidget(self.sheet_combo, 1, 1)
        self.lbl_file = QLabel("(no file loaded)")
        self.lbl_file.setObjectName("status")
        self.lbl_file.setWordWrap(True)
        gl.addWidget(self.lbl_file, 2, 0, 1, 2)
        self.lbl_event_info = QLabel("Events: (none)")
        self.lbl_event_info.setObjectName("status")
        self.lbl_event_info.setWordWrap(True)
        gl.addWidget(self.lbl_event_info, 3, 0, 1, 2)
        return box

    def _build_pose_box(self) -> QGroupBox:
        box = QGroupBox("Pose Slots")
        gl = QGridLayout(box)
        gl.setVerticalSpacing(4)
        gl.addWidget(QLabel("Show"), 0, 0)
        gl.addWidget(QLabel("Pose"),  0, 1)
        gl.addWidget(QLabel("Event"), 0, 2)
        gl.addWidget(QLabel("Reload"), 0, 3)
        self._pose_visible_checks: dict[str, QCheckBox] = {}
        self._pose_event_combos: dict[str, QComboBox] = {}
        for r, (key, slot) in enumerate(self.poses.items(), start=1):
            cb = QCheckBox()
            cb.setChecked(slot.visible)
            cb.stateChanged.connect(self._on_pose_toggled)
            self._pose_visible_checks[key] = cb
            gl.addWidget(cb, r, 0)
            color = slot.color
            tag = QLabel(f'<span style="color:{color};">●</span>  {key}')
            gl.addWidget(tag, r, 1)
            ec = QComboBox()
            ec.addItems(["A", "T", "I", "F"])
            ec.setCurrentText(slot.target_event)
            ec.currentTextChanged.connect(self._on_pose_event_changed)
            self._pose_event_combos[key] = ec
            gl.addWidget(ec, r, 2)
            btn = QPushButton("⟳")
            btn.setObjectName("preset")
            btn.setMaximumWidth(40)
            btn.setToolTip(f"Reload simscape_skeleton_{key}.json")
            btn.clicked.connect(lambda _checked, k=key: self._reload_pose(k))
            gl.addWidget(btn, r, 3)
        return box

    def _build_view_box(self) -> QGroupBox:
        box = QGroupBox("View / Mocap Traces")
        v = QVBoxLayout(box)
        v.setSpacing(6)

        # Camera presets
        cam_row = QHBoxLayout()
        cam_row.setSpacing(4)
        for name in _CAMERA_PRESETS:
            b = QPushButton(name)
            b.setObjectName("preset")
            b.clicked.connect(lambda _checked, n=name: self._apply_camera_preset(n))
            cam_row.addWidget(b)
        v.addLayout(cam_row)

        v.addWidget(_hsep())

        # Trace toggles
        self.cb_clubhead_trace = QCheckBox("Show mocap clubhead path")
        self.cb_clubhead_trace.stateChanged.connect(self._on_traces_toggled)
        v.addWidget(self.cb_clubhead_trace)
        self.cb_midhands_trace = QCheckBox("Show mocap mid-hands path")
        self.cb_midhands_trace.stateChanged.connect(self._on_traces_toggled)
        v.addWidget(self.cb_midhands_trace)

        # Phase window combo (replaces the old simple "swing window" checkbox)
        ph_row = QHBoxLayout()
        ph_row.addWidget(QLabel("Phase:"))
        self.phase_combo = QComboBox()
        for label in _PHASE_WINDOWS:
            self.phase_combo.addItem(label)
        self.phase_combo.setCurrentText(_DEFAULT_PHASE)
        self.phase_combo.currentTextChanged.connect(self._on_phase_changed)
        ph_row.addWidget(self.phase_combo, stretch=1)
        v.addLayout(ph_row)

        # Manual range (hidden until "Manual range" selected)
        self.manual_range_widget = QWidget()
        mr = QHBoxLayout(self.manual_range_widget)
        mr.setContentsMargins(0, 0, 0, 0)
        mr.addWidget(QLabel("From:"))
        self.spin_phase_start = QSpinBox()
        self.spin_phase_start.setRange(0, 0)
        self.spin_phase_start.valueChanged.connect(self._on_manual_range_changed)
        mr.addWidget(self.spin_phase_start)
        mr.addWidget(QLabel("To:"))
        self.spin_phase_end = QSpinBox()
        self.spin_phase_end.setRange(0, 0)
        self.spin_phase_end.valueChanged.connect(self._on_manual_range_changed)
        mr.addWidget(self.spin_phase_end)
        self.manual_range_widget.setVisible(False)
        v.addWidget(self.manual_range_widget)

        # Show current-frame marker on traces
        self.cb_frame_marker = QCheckBox("Show current-frame marker on traces")
        self.cb_frame_marker.setChecked(True)
        self.cb_frame_marker.stateChanged.connect(lambda _: self._redraw())
        v.addWidget(self.cb_frame_marker)
        return box

    def _build_playback_box(self) -> QGroupBox:
        box = QGroupBox("Playback")
        v = QVBoxLayout(box)
        v.setSpacing(6)

        # Frame slider + spinbox row
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Frame:"))
        self.spin_frame = QSpinBox()
        self.spin_frame.setRange(0, 0)
        self.spin_frame.setMinimumWidth(80)
        self.spin_frame.setKeyboardTracking(False)
        self.spin_frame.valueChanged.connect(self._on_frame_changed_spin)
        row1.addWidget(self.spin_frame)
        self.lbl_time = QLabel("t = — s")
        self.lbl_time.setObjectName("status")
        self.lbl_time.setMinimumWidth(110)
        row1.addWidget(self.lbl_time)
        v.addLayout(row1)

        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setRange(0, 0)
        self.frame_slider.valueChanged.connect(self._on_frame_changed_slider)
        v.addWidget(self.frame_slider)

        # Step buttons
        step_row = QHBoxLayout()
        step_row.setSpacing(2)
        for label, delta, tip in [
            ("⏮", -10**9, "First frame"),
            ("⏪", -10, "−10 frames"),
            ("◀", -1, "−1 frame"),
            ("▶", +1, "+1 frame"),
            ("⏩", +10, "+10 frames"),
            ("⏭", +10**9, "Last frame"),
        ]:
            b = QPushButton(label)
            b.setObjectName("preset")
            b.setToolTip(tip)
            b.setMaximumWidth(46)
            b.clicked.connect(lambda _checked, d=delta: self._step_frame(d))
            step_row.addWidget(b)
        v.addLayout(step_row)

        # Play/pause + speed
        play_row = QHBoxLayout()
        self.btn_play = QPushButton("▶ Play")
        self.btn_play.setObjectName("primary")
        self.btn_play.clicked.connect(self._toggle_play)
        play_row.addWidget(self.btn_play)
        play_row.addWidget(QLabel("Speed:"))
        self.spin_speed = QSpinBox()
        self.spin_speed.setRange(1, 240)
        self.spin_speed.setValue(30)
        self.spin_speed.setSuffix(" fps")
        play_row.addWidget(self.spin_speed)
        self.cb_loop = QCheckBox("Loop")
        self.cb_loop.setChecked(True)
        self.cb_loop.stateChanged.connect(
            lambda _: setattr(self, "loop_playback", self.cb_loop.isChecked()))
        play_row.addWidget(self.cb_loop)
        v.addLayout(play_row)

        # Use-current-frame override
        self.cb_use_current_frame = QCheckBox(
            "Use current frame for mocap target (override pose-slot events)")
        self.cb_use_current_frame.stateChanged.connect(self._on_frame_override_toggled)
        v.addWidget(self.cb_use_current_frame)

        # "Set as event" row
        ev_row = QHBoxLayout()
        ev_row.addWidget(QLabel("Mark current frame as event:"))
        self.combo_set_event = QComboBox()
        self.combo_set_event.addItems(["A", "T", "I", "F"])
        ev_row.addWidget(self.combo_set_event)
        b_set = QPushButton("Set")
        b_set.setObjectName("preset")
        b_set.clicked.connect(self._set_event_to_current_frame)
        ev_row.addWidget(b_set)
        b_clear = QPushButton("Clear overrides")
        b_clear.setObjectName("preset")
        b_clear.clicked.connect(self._clear_event_overrides)
        ev_row.addWidget(b_clear)
        v.addLayout(ev_row)
        return box

    def _build_align_box(self) -> QGroupBox:
        box = QGroupBox("Auto-Align")
        v = QVBoxLayout(box)
        v.setSpacing(6)

        hint = QLabel(
            "Solves Rz + Tx/Ty/Tz so the model SHAFT (mid-hands → clubhead) "
            "lines up with the mocap shaft at the chosen frame.")
        hint.setObjectName("status")
        hint.setWordWrap(True)
        v.addWidget(hint)

        self.cb_fit_scale = QCheckBox("Also fit scale (|shaft_target| / |shaft_model|)")
        v.addWidget(self.cb_fit_scale)

        # One snap button per pose-slot
        for key, slot in self.poses.items():
            btn = QPushButton(f"Snap {key} pose → mocap @ {slot.target_event} (shaft-aligned)")
            btn.setObjectName("primary")
            btn.clicked.connect(lambda _checked, k=key: self._snap_shaft(k))
            v.addWidget(btn)

        v.addWidget(_hsep())
        # Convenience: snap mid-hands only (legacy quick-snap)
        self.btn_snap_mid = QPushButton("Snap mid-hands only (no rotation)")
        self.btn_snap_mid.setToolTip("Set Tx/Ty/Tz so the FIRST visible skeleton's "
                                     "mid-hands lands on its mocap target.  "
                                     "Rotations preserved.")
        self.btn_snap_mid.clicked.connect(self._snap_mid_first_visible)
        v.addWidget(self.btn_snap_mid)
        return box

    def _build_transform_box(self) -> QGroupBox:
        box = QGroupBox("Rigid Transform + Scale")
        v = QVBoxLayout(box)
        v.setSpacing(4)

        self.s_tx = LabelledControl("Tx", "m", _T_RANGE, _T_SCALE, 3)
        self.s_ty = LabelledControl("Ty", "m", _T_RANGE, _T_SCALE, 3)
        self.s_tz = LabelledControl("Tz", "m", _T_RANGE, _T_SCALE, 3)
        v.addWidget(self.s_tx)
        v.addWidget(self.s_ty)
        v.addWidget(self.s_tz)

        v.addWidget(_hsep())

        # Rz + presets (always enabled — Z is the heading axis)
        self.s_rz = LabelledControl("Rz", "°", _R_RANGE, _R_SCALE, 1)
        v.addWidget(self.s_rz)
        rz_row = QHBoxLayout()
        rz_row.setSpacing(4)
        rz_row.addWidget(QLabel("Presets:"))
        for label, deg in [("-90°", -90), ("-45°", -45), ("0°", 0),
                           ("+45°", 45), ("+90°", 90), ("180°", 180)]:
            b = QPushButton(label)
            b.setObjectName("preset")
            b.clicked.connect(lambda _checked, d=deg: self.s_rz.set_value(d))
            rz_row.addWidget(b)
        v.addLayout(rz_row)

        v.addWidget(_hsep())

        # X/Y rotation lock
        self.cb_lock_xy = QCheckBox("Allow Rx/Ry rotations (off by default — Z is up in both data and model)")
        self.cb_lock_xy.setChecked(False)
        self.cb_lock_xy.stateChanged.connect(self._on_lock_xy_toggled)
        v.addWidget(self.cb_lock_xy)

        self.s_rx = LabelledControl("Rx", "°", _R_RANGE, _R_SCALE, 1)
        self.s_ry = LabelledControl("Ry", "°", _R_RANGE, _R_SCALE, 1)
        v.addWidget(self.s_rx)
        v.addWidget(self.s_ry)
        self.s_rx.setEnabled(False)
        self.s_ry.setEnabled(False)

        v.addWidget(_hsep())

        # Scale + presets
        self.s_scale = LabelledControl(
            "Scale", "×", _S_RANGE, _S_SCALE, 2, default=1.0)
        v.addWidget(self.s_scale)
        sc_row = QHBoxLayout()
        sc_row.setSpacing(4)
        sc_row.addWidget(QLabel("Presets:"))
        for label, val in [("0.85", 0.85), ("0.95", 0.95), ("1.00", 1.00),
                           ("1.05", 1.05), ("1.15", 1.15)]:
            b = QPushButton(label)
            b.setObjectName("preset")
            b.clicked.connect(lambda _checked, x=val: self.s_scale.set_value(x))
            sc_row.addWidget(b)
        v.addLayout(sc_row)

        # Pivot info
        pi = QLabel(
            "Pivot @ first-pose hub: ({:.3f}, {:.3f}, {:.3f}) m".format(
                *self.transform.pivot))
        pi.setObjectName("status")
        v.addWidget(pi)

        # Wire all the changes
        for s in (self.s_tx, self.s_ty, self.s_tz, self.s_rx, self.s_ry, self.s_rz,
                  self.s_scale):
            s.spin.valueChanged.connect(self._on_transform_changed)

        # Reset row
        reset_row = QHBoxLayout()
        self.btn_reset_t = QPushButton("Reset translations")
        self.btn_reset_t.clicked.connect(self._reset_translations)
        reset_row.addWidget(self.btn_reset_t)
        self.btn_reset_r = QPushButton("Reset rotations")
        self.btn_reset_r.clicked.connect(self._reset_rotations)
        reset_row.addWidget(self.btn_reset_r)
        self.btn_reset_all = QPushButton("Reset all")
        self.btn_reset_all.clicked.connect(self._reset_all)
        reset_row.addWidget(self.btn_reset_all)
        v.addLayout(reset_row)
        return box

    def _build_save_box(self) -> QGroupBox:
        box = QGroupBox("Output")
        v = QVBoxLayout(box)
        v.setSpacing(6)

        self.btn_save = QPushButton("Save offsets to JSON…")
        self.btn_save.setObjectName("accent")
        self.btn_save.clicked.connect(self._on_save_clicked)
        v.addWidget(self.btn_save)

        ses_row = QHBoxLayout()
        self.btn_save_session = QPushButton("Save session…")
        self.btn_save_session.clicked.connect(self._on_save_session_clicked)
        ses_row.addWidget(self.btn_save_session)
        self.btn_load_session = QPushButton("Load session…")
        self.btn_load_session.clicked.connect(self._on_load_session_clicked)
        ses_row.addWidget(self.btn_load_session)
        v.addLayout(ses_row)

        self.lbl_residual = QLabel("Residuals: (no data)")
        self.lbl_residual.setObjectName("residual")
        self.lbl_residual.setWordWrap(True)
        v.addWidget(self.lbl_residual)
        return box

    # ---------- axis setup ----------------------------------------------- #

    def _setup_axes(self) -> None:
        ax = self.ax
        ax.set_xlabel("X (target line)", color="#cbd5e1")
        ax.set_ylabel("Y (ball direction)", color="#cbd5e1")
        ax.set_zlabel("Z (vertical)", color="#cbd5e1")
        ax.set_xlim(-2.0, 2.0)
        ax.set_ylim(-1.5, 2.0)
        ax.set_zlim(-1.5, 2.5)
        try:
            ax.set_box_aspect((4, 3.5, 4))
        except AttributeError:
            pass
        # Dark-theme tick & pane colours
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.set_pane_color((0.16, 0.18, 0.22, 0.85))
            axis.label.set_color("#cbd5e1")
            for t in axis.get_ticklabels():
                t.set_color("#a3a8b3")
            axis._axinfo['grid']['color'] = (0.35, 0.40, 0.48, 0.45)

    # ===================================================================== #
    # Event handlers                                                        #
    # ===================================================================== #

    def _apply_camera_preset(self, name: str) -> None:
        elev, azim = _CAMERA_PRESETS.get(name, _CAMERA_PRESETS[_DEFAULT_CAMERA])
        self.ax.view_init(elev=elev, azim=azim)
        self.canvas.draw()

    def _on_load_clicked(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Wiffle xlsx", str(Path(__file__).parent),
            "Excel files (*.xlsx *.xls)")
        if path:
            self._load_xlsx(path)

    def _on_sheet_changed(self, _: str) -> None:
        if self._xlsx_path:
            self._load_xlsx(self._xlsx_path)

    def _on_pose_toggled(self, _state: int) -> None:
        for key, cb in self._pose_visible_checks.items():
            self.poses[key].visible = cb.isChecked()
        self._redraw()

    def _on_pose_event_changed(self, _: str) -> None:
        for key, combo in self._pose_event_combos.items():
            self.poses[key].target_event = combo.currentText()
        self._redraw()

    def _on_traces_toggled(self, _: int) -> None:
        self.show_clubhead_trace = self.cb_clubhead_trace.isChecked()
        self.show_midhands_trace = self.cb_midhands_trace.isChecked()
        self._redraw()

    def _on_phase_changed(self, label: str) -> None:
        self.phase_window = label
        is_manual = (label == "Manual range")
        self.manual_range_widget.setVisible(is_manual)
        self._redraw()

    def _on_manual_range_changed(self, _: int) -> None:
        self.manual_window_start = int(self.spin_phase_start.value())
        self.manual_window_end = int(self.spin_phase_end.value())
        if self.manual_window_end < self.manual_window_start:
            self.manual_window_end = self.manual_window_start
            with QSignalBlocker(self.spin_phase_end):
                self.spin_phase_end.setValue(self.manual_window_end)
        self._redraw()

    # ---- playback handlers ----
    def _on_frame_changed_slider(self, frame: int) -> None:
        with QSignalBlocker(self.spin_frame):
            self.spin_frame.setValue(int(frame))
        self.current_frame = int(frame)
        self._update_time_label()
        self._redraw()

    def _on_frame_changed_spin(self, frame: int) -> None:
        with QSignalBlocker(self.frame_slider):
            self.frame_slider.setValue(int(frame))
        self.current_frame = int(frame)
        self._update_time_label()
        self._redraw()

    def _step_frame(self, delta: int) -> None:
        if self.df is None:
            return
        n = len(self.df)
        if delta <= -10**8:
            self.spin_frame.setValue(0)
        elif delta >= 10**8:
            self.spin_frame.setValue(n - 1)
        else:
            new = max(0, min(n - 1, self.current_frame + delta))
            self.spin_frame.setValue(new)

    def _toggle_play(self) -> None:
        if self.is_playing:
            self._timer.stop()
            self.is_playing = False
            self.btn_play.setText("▶ Play")
        else:
            if self.df is None or len(self.df) == 0:
                return
            fps = max(1, int(self.spin_speed.value()))
            self._timer.start(int(round(1000.0 / fps)))
            self.is_playing = True
            self.btn_play.setText("⏸ Pause")

    def _advance_frame(self) -> None:
        if self.df is None:
            return
        n = len(self.df)
        nxt = self.current_frame + 1
        if nxt >= n:
            if self.loop_playback:
                nxt = 0
            else:
                self._toggle_play()
                return
        self.spin_frame.setValue(nxt)

    def _on_frame_override_toggled(self, _state: int) -> None:
        self.frame_override_active = self.cb_use_current_frame.isChecked()
        self._redraw()

    def _set_event_to_current_frame(self) -> None:
        if self.df is None:
            return
        ev = self.combo_set_event.currentText()
        # Store as "absolute sample number" (1-based) so it round-trips with
        # MocapEvents; current_frame is 0-based in the loaded data.
        self.event_overrides[ev] = self.current_frame + 1
        # Reflect in events struct for in-session use
        setattr(self.events, f"{ev}_sample", float(self.current_frame + 1))
        self.lbl_event_info.setText(self._events_summary() + "  (overrides active)")
        self._redraw()

    def _clear_event_overrides(self) -> None:
        if not self.event_overrides:
            return
        # Re-read events from the xlsx to undo overrides
        if self._xlsx_path:
            self.events = read_event_header(self._xlsx_path,
                                            self.sheet_combo.currentText())
        self.event_overrides = {}
        self.lbl_event_info.setText(self._events_summary())
        self._redraw()

    def _update_time_label(self) -> None:
        if self.df is None or self.current_frame >= len(self.df):
            self.lbl_time.setText("t = — s")
            return
        t = float(self.df.iloc[self.current_frame]["time"])
        self.lbl_time.setText(f"t = {t:+.3f} s   (frame {self.current_frame})")

    def _on_lock_xy_toggled(self, _state: int) -> None:
        self.lock_xy_rotation = not self.cb_lock_xy.isChecked()
        if self.lock_xy_rotation:
            self.s_rx.set_value(0.0)
            self.s_ry.set_value(0.0)
        self.s_rx.setEnabled(not self.lock_xy_rotation)
        self.s_ry.setEnabled(not self.lock_xy_rotation)

    def _on_transform_changed(self, _: float) -> None:
        self.transform.tx = self.s_tx.value()
        self.transform.ty = self.s_ty.value()
        self.transform.tz = self.s_tz.value()
        if self.lock_xy_rotation:
            self.transform.rx = 0.0
            self.transform.ry = 0.0
        else:
            self.transform.rx = self.s_rx.value()
            self.transform.ry = self.s_ry.value()
        self.transform.rz = self.s_rz.value()
        self.transform.scale = max(1e-3, self.s_scale.value())
        self._redraw()

    # ---------- resets ---------------------------------------------------- #

    def _reset_translations(self) -> None:
        for s in (self.s_tx, self.s_ty, self.s_tz):
            s.set_value(0.0)

    def _reset_rotations(self) -> None:
        for s in (self.s_rx, self.s_ry, self.s_rz):
            s.set_value(0.0)

    def _reset_all(self) -> None:
        self._reset_translations()
        self._reset_rotations()
        self.s_scale.set_value(1.0)

    def _reload_pose(self, key: str) -> None:
        path = Path(__file__).parent / f"simscape_skeleton_{key}.json"
        self.poses[key].skeleton = load_skeleton(path, key)
        self._redraw()

    # ---------- snaps ----------------------------------------------------- #

    def _snap_mid_first_visible(self) -> None:
        slot = self._first_visible_pose()
        if slot is None or "mp" not in slot.skeleton.joints:
            return
        target = self._mocap_pos_for(slot, "mid")
        if target is None:
            return
        # Apply current rotation+scale (no translation) and compute delta.
        no_t = RigidTransform(
            rx=0.0 if self.lock_xy_rotation else self.s_rx.value(),
            ry=0.0 if self.lock_xy_rotation else self.s_ry.value(),
            rz=self.s_rz.value(),
            scale=max(1e-3, self.s_scale.value()),
            pivot=self.transform.pivot)
        rotated_mp = no_t.apply(slot.skeleton.joints["mp"][None, :])[0]
        delta = target - rotated_mp
        self.s_tx.set_value(float(delta[0]))
        self.s_ty.set_value(float(delta[1]))
        self.s_tz.set_value(float(delta[2]))

    def _snap_shaft(self, slot_key: str) -> None:
        """Two-point shaft alignment for one pose.

        Keeps Z up (Rx=Ry=0).  Solves Rz so the model shaft (mp→ch) in the
        XY plane points the same way as the mocap shaft, then sets Tx/Ty/Tz
        so the model mid-hands lands on the mocap mid-hands.  Optionally
        sets scale = |shaft_target| / |shaft_model|.
        """
        slot = self.poses.get(slot_key)
        if slot is None:
            return
        sk = slot.skeleton
        if "mp" not in sk.joints or "ch" not in sk.joints:
            self._notify("Pose lacks mp/ch joints — cannot shaft-snap.")
            return
        mp_target = self._mocap_pos_for(slot, "mid")
        ch_target = self._mocap_pos_for(slot, "club")
        if mp_target is None or ch_target is None:
            self._notify(f"No mocap row for event '{slot.target_event}'.")
            return

        mp_skel = sk.joints["mp"]
        ch_skel = sk.joints["ch"]

        # Optional scale: ratio of shaft lengths.
        if self.cb_fit_scale.isChecked():
            shaft_t = ch_target - mp_target
            shaft_m = ch_skel - mp_skel
            len_t = float(np.linalg.norm(shaft_t))
            len_m = float(np.linalg.norm(shaft_m))
            if len_m > 1e-6 and len_t > 1e-6:
                new_scale = float(np.clip(len_t / len_m, _S_RANGE[0] * _S_SCALE,
                                          _S_RANGE[1] * _S_SCALE))
                self.s_scale.set_value(new_scale)

        scale = max(1e-3, self.s_scale.value())

        # Solve Rz from XY-plane shaft directions.
        shaft_t_xy = (ch_target - mp_target)[:2]
        shaft_m_xy = (ch_skel - mp_skel)[:2]
        nt = float(np.linalg.norm(shaft_t_xy))
        nm = float(np.linalg.norm(shaft_m_xy))
        if nt < 1e-6 or nm < 1e-6:
            self._notify("Shaft projection onto XY plane is degenerate (vertical "
                         "shaft) — Rz cannot be solved.  Adjust manually.")
            return
        a_t = float(np.arctan2(shaft_t_xy[1], shaft_t_xy[0]))
        a_m = float(np.arctan2(shaft_m_xy[1], shaft_m_xy[0]))
        rz_deg = float(np.degrees(a_t - a_m))
        # Wrap to [-180, 180]
        rz_deg = ((rz_deg + 180.0) % 360.0) - 180.0

        # Lock Rx/Ry to 0 for this snap (Z-up).
        if not self.lock_xy_rotation:
            self.cb_lock_xy.setChecked(False)  # leave as-is for user; we just zero
        self.s_rx.set_value(0.0)
        self.s_ry.set_value(0.0)
        self.s_rz.set_value(rz_deg)

        # Translation: rotate+scale mp_skel about pivot, then offset to land on mp_target.
        rotated = RigidTransform(
            rx=0.0, ry=0.0, rz=rz_deg, scale=scale,
            pivot=self.transform.pivot)
        rotated_mp = rotated.apply(mp_skel[None, :])[0]
        delta = mp_target - rotated_mp
        self.s_tx.set_value(float(delta[0]))
        self.s_ty.set_value(float(delta[1]))
        self.s_tz.set_value(float(delta[2]))
        self._notify(f"Snapped {slot_key}: Rz={rz_deg:+.1f}°, "
                     f"|shaft_target|={nt:.3f}m, |shaft_model|={nm:.3f}m")

    # ---------- file load ------------------------------------------------- #

    def _load_xlsx(self, path: str) -> None:
        sheet = self.sheet_combo.currentText()
        try:
            df = load_mocap_xlsx(path, sheet)
        except Exception as exc:  # noqa: BLE001
            logger.error("Load failed: %s", exc)
            self.lbl_file.setText(f"Load failed: {exc}")
            return
        if df is None or df.empty:
            self.lbl_file.setText(f"No data in sheet '{sheet}'")
            return
        self.df = df
        self._xlsx_path = path
        self.events = read_event_header(path, sheet)
        # Re-apply any event-override that survived from the previous load
        for ev, sample in list(self.event_overrides.items()):
            setattr(self.events, f"{ev}_sample", float(sample))
        n = len(df)
        self.lbl_file.setText(f"{Path(path).name}\nsheet={sheet}  frames={n}")
        self.lbl_event_info.setText(self._events_summary())
        # Configure playback widgets to the new range
        with QSignalBlocker(self.spin_frame):
            self.spin_frame.setRange(0, n - 1)
        with QSignalBlocker(self.frame_slider):
            self.frame_slider.setRange(0, n - 1)
        with QSignalBlocker(self.spin_phase_start):
            self.spin_phase_start.setRange(0, n - 1)
        with QSignalBlocker(self.spin_phase_end):
            self.spin_phase_end.setRange(0, n - 1)
            self.spin_phase_end.setValue(n - 1)
        self.manual_window_end = n - 1
        # Default initial frame to T (top of backswing) if available
        t_frame = self._frame_for("T")
        if t_frame is not None:
            with QSignalBlocker(self.spin_frame):
                self.spin_frame.setValue(t_frame)
            with QSignalBlocker(self.frame_slider):
                self.frame_slider.setValue(t_frame)
            self.current_frame = t_frame
        self._update_time_label()
        self._redraw()

    def _events_summary(self) -> str:
        e = self.events
        parts = []
        for label in ("A", "T", "I", "F"):
            v = getattr(e, f"{label}_sample")
            parts.append(f"{label}={'?' if v != v else int(v)}")
        if e.CHS_mph == e.CHS_mph:
            parts.append(f"CHS={e.CHS_mph:.1f}mph")
        return "Events:  " + "  ".join(parts)

    # ---------- save ------------------------------------------------------ #

    def _on_save_clicked(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save offsets",
            str(Path(__file__).parent / "starting_pose_offsets.json"),
            "JSON (*.json)")
        if not path:
            return
        out = {
            "transform": {
                "tx": self.transform.tx, "ty": self.transform.ty,
                "tz": self.transform.tz, "rx": self.transform.rx,
                "ry": self.transform.ry, "rz": self.transform.rz,
                "scale": self.transform.scale,
                "pivot": list(self.transform.pivot),
                "lock_xy_rotation": self.lock_xy_rotation,
                "units": {"translation": "metres", "rotation": "degrees",
                          "rotation_order": "Rz @ Ry @ Rx (intrinsic XYZ)"}},
            "poses": {key: {
                "visible": slot.visible, "event": slot.target_event,
                "skeleton_source": str(Path(__file__).parent /
                                       f"simscape_skeleton_{key}.json"),
            } for key, slot in self.poses.items()},
            "events": {
                "A_sample": self.events.A_sample, "T_sample": self.events.T_sample,
                "I_sample": self.events.I_sample, "F_sample": self.events.F_sample,
                "CHS_mph": self.events.CHS_mph},
            "residuals_mm": self._compute_residuals_mm(),
        }
        with open(path, "w") as f:
            json.dump(out, f, indent=2, default=float)
        self._notify(f"Saved: {Path(path).name}")
        logger.info("Wrote %s", path)

    # ---------- session save / load --------------------------------------- #

    def _serialize_session(self) -> dict[str, Any]:
        """Snapshot the entire UI state to a JSON-serialisable dict."""
        return {
            "schema_version": _SESSION_SCHEMA_VERSION,
            "saved_at": pd.Timestamp.now().isoformat(),
            "xlsx_path": self._xlsx_path,
            "sheet": self.sheet_combo.currentText(),
            "transform": {
                "tx": self.transform.tx, "ty": self.transform.ty,
                "tz": self.transform.tz, "rx": self.transform.rx,
                "ry": self.transform.ry, "rz": self.transform.rz,
                "scale": self.transform.scale,
                "pivot": list(self.transform.pivot),
            },
            "lock_xy_rotation": self.lock_xy_rotation,
            "poses": {key: {"visible": slot.visible,
                            "event": slot.target_event,
                            "skeleton_path":
                                str(Path(__file__).parent /
                                    f"simscape_skeleton_{key}.json")}
                      for key, slot in self.poses.items()},
            "view": {"elev": float(self.ax.elev), "azim": float(self.ax.azim)},
            "traces": {
                "clubhead": self.show_clubhead_trace,
                "midhands": self.show_midhands_trace,
                "phase": self.phase_window,
                "manual_start": self.manual_window_start,
                "manual_end": self.manual_window_end,
                "frame_marker": self.cb_frame_marker.isChecked(),
            },
            "playback": {
                "current_frame": self.current_frame,
                "frame_override_active": self.frame_override_active,
                "loop": self.loop_playback,
                "fps": int(self.spin_speed.value()),
            },
            "event_overrides": dict(self.event_overrides),
        }

    def _on_save_session_clicked(self) -> None:
        ses_dir = Path(__file__).parent / "sessions"
        ses_dir.mkdir(exist_ok=True)
        sheet = self.sheet_combo.currentText() or "session"
        ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        path, _ = QFileDialog.getSaveFileName(
            self, "Save session", str(ses_dir / f"{sheet}_{ts}.session.json"),
            "JSON (*.json)")
        if not path:
            return
        with open(path, "w") as f:
            json.dump(self._serialize_session(), f, indent=2, default=float)
        self._notify(f"Saved session: {Path(path).name}")
        logger.info("Wrote session %s", path)

    def _on_load_session_clicked(self) -> None:
        ses_dir = Path(__file__).parent / "sessions"
        start = str(ses_dir) if ses_dir.exists() else str(Path(__file__).parent)
        path, _ = QFileDialog.getOpenFileName(
            self, "Load session", start, "JSON (*.json)")
        if not path:
            return
        try:
            with open(path) as f:
                d = json.load(f)
        except Exception as exc:  # noqa: BLE001
            self._notify(f"Load failed: {exc}")
            return
        self._apply_session(d)
        self._notify(f"Loaded session: {Path(path).name}")

    def _apply_session(self, d: dict[str, Any]) -> None:
        """Restore UI state from a session dict.  Forward-compatible: missing
        keys keep current values, unknown keys are ignored.
        """
        ver = d.get("schema_version", 1)
        if ver > _SESSION_SCHEMA_VERSION:
            logger.warning("Session schema_version=%s newer than supported %s "
                           "— ignoring unknown keys.", ver, _SESSION_SCHEMA_VERSION)

        # 1. Re-load xlsx + sheet (this resets a lot of widgets, so do it first).
        xlsx = d.get("xlsx_path")
        sheet = d.get("sheet")
        if sheet:
            with QSignalBlocker(self.sheet_combo):
                self.sheet_combo.setCurrentText(sheet)
        if xlsx and Path(xlsx).exists():
            self._load_xlsx(xlsx)
        elif xlsx:
            logger.warning("Saved xlsx not found: %s", xlsx)

        # 2. Event overrides (applied on top of the freshly-loaded events).
        evo = d.get("event_overrides") or {}
        for ev, sample in evo.items():
            self.event_overrides[ev] = int(sample)
            setattr(self.events, f"{ev}_sample", float(sample))
        if evo:
            self.lbl_event_info.setText(self._events_summary() + "  (overrides active)")

        # 3. Pose visibility + events.
        for key, slot_d in (d.get("poses") or {}).items():
            if key not in self.poses:
                continue
            cb = self._pose_visible_checks.get(key)
            ec = self._pose_event_combos.get(key)
            if cb is not None:
                with QSignalBlocker(cb):
                    cb.setChecked(bool(slot_d.get("visible", True)))
                self.poses[key].visible = cb.isChecked()
            if ec is not None and slot_d.get("event") in ("A", "T", "I", "F"):
                with QSignalBlocker(ec):
                    ec.setCurrentText(slot_d["event"])
                self.poses[key].target_event = slot_d["event"]

        # 4. Transform sliders.
        tf = d.get("transform") or {}
        for attr, widget in [("tx", self.s_tx), ("ty", self.s_ty), ("tz", self.s_tz),
                             ("rx", self.s_rx), ("ry", self.s_ry), ("rz", self.s_rz),
                             ("scale", self.s_scale)]:
            if attr in tf:
                with QSignalBlocker(widget.spin):
                    widget.set_value(float(tf[attr]))
                with QSignalBlocker(widget.slider):
                    widget.slider.setValue(int(round(float(tf[attr]) / widget._scale)))
                setattr(self.transform, attr, float(tf[attr]))
        if "pivot" in tf:
            self.transform.pivot = tuple(tf["pivot"])

        # 5. Lock-XY rotation.
        if "lock_xy_rotation" in d:
            allow_xy = not bool(d["lock_xy_rotation"])
            with QSignalBlocker(self.cb_lock_xy):
                self.cb_lock_xy.setChecked(allow_xy)
            self.lock_xy_rotation = not allow_xy
            self.s_rx.setEnabled(allow_xy)
            self.s_ry.setEnabled(allow_xy)

        # 6. Camera.
        view = d.get("view") or {}
        if "elev" in view and "azim" in view:
            self.ax.view_init(elev=float(view["elev"]), azim=float(view["azim"]))

        # 7. Traces / phase.
        tr = d.get("traces") or {}
        if "clubhead" in tr:
            with QSignalBlocker(self.cb_clubhead_trace):
                self.cb_clubhead_trace.setChecked(bool(tr["clubhead"]))
            self.show_clubhead_trace = bool(tr["clubhead"])
        if "midhands" in tr:
            with QSignalBlocker(self.cb_midhands_trace):
                self.cb_midhands_trace.setChecked(bool(tr["midhands"]))
            self.show_midhands_trace = bool(tr["midhands"])
        if tr.get("phase") in _PHASE_WINDOWS:
            with QSignalBlocker(self.phase_combo):
                self.phase_combo.setCurrentText(tr["phase"])
            self.phase_window = tr["phase"]
            self.manual_range_widget.setVisible(self.phase_window == "Manual range")
        if "manual_start" in tr:
            with QSignalBlocker(self.spin_phase_start):
                self.spin_phase_start.setValue(int(tr["manual_start"]))
            self.manual_window_start = int(tr["manual_start"])
        if "manual_end" in tr:
            with QSignalBlocker(self.spin_phase_end):
                self.spin_phase_end.setValue(int(tr["manual_end"]))
            self.manual_window_end = int(tr["manual_end"])
        if "frame_marker" in tr:
            with QSignalBlocker(self.cb_frame_marker):
                self.cb_frame_marker.setChecked(bool(tr["frame_marker"]))

        # 8. Playback.
        pb = d.get("playback") or {}
        if "current_frame" in pb:
            with QSignalBlocker(self.spin_frame):
                self.spin_frame.setValue(int(pb["current_frame"]))
            with QSignalBlocker(self.frame_slider):
                self.frame_slider.setValue(int(pb["current_frame"]))
            self.current_frame = int(pb["current_frame"])
            self._update_time_label()
        if "frame_override_active" in pb:
            with QSignalBlocker(self.cb_use_current_frame):
                self.cb_use_current_frame.setChecked(bool(pb["frame_override_active"]))
            self.frame_override_active = bool(pb["frame_override_active"])
        if "loop" in pb:
            with QSignalBlocker(self.cb_loop):
                self.cb_loop.setChecked(bool(pb["loop"]))
            self.loop_playback = bool(pb["loop"])
        if "fps" in pb:
            with QSignalBlocker(self.spin_speed):
                self.spin_speed.setValue(int(pb["fps"]))

        self._redraw()

    # ---------- helpers --------------------------------------------------- #

    def _frame_for(self, label: str) -> int | None:
        f = self.events.frame_for(label)
        if f is None or self.df is None:
            return None
        return max(0, min(f, len(self.df) - 1))

    def _first_visible_pose(self) -> PoseSlot | None:
        for slot in self.poses.values():
            if slot.visible:
                return slot
        return None

    def _mocap_pos_for(self, slot: PoseSlot, kind: str) -> np.ndarray | None:
        if self.df is None:
            return None
        if self.frame_override_active:
            f: int | None = self.current_frame
        else:
            f = self._frame_for(slot.target_event)
        if f is None:
            return None
        f = max(0, min(int(f), len(self.df) - 1))
        row = self.df.iloc[f]
        if kind == "mid":
            return np.array([-row["mid_X"], row["mid_Y"], row["mid_Z"]])
        return np.array([-row["club_X"], row["club_Y"], row["club_Z"]])

    def _compute_residuals_mm(self) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for key, slot in self.poses.items():
            if "mp" not in slot.skeleton.joints:
                continue
            target = self._mocap_pos_for(slot, "mid")
            if target is None:
                continue
            moved = self.transform.apply(slot.skeleton.joints["mp"][None, :])[0]
            d_mid = (moved - target) * 1000.0
            entry = {"dx_mm": float(d_mid[0]), "dy_mm": float(d_mid[1]),
                     "dz_mm": float(d_mid[2]),
                     "norm_mm": float(np.linalg.norm(d_mid))}
            ch_target = self._mocap_pos_for(slot, "club")
            if ch_target is not None and "ch" in slot.skeleton.joints:
                moved_ch = self.transform.apply(
                    slot.skeleton.joints["ch"][None, :])[0]
                entry["clubhead_norm_mm"] = float(
                    np.linalg.norm((moved_ch - ch_target) * 1000.0))
            out[key] = entry
        return out

    def _notify(self, msg: str) -> None:
        # Reuse the residual line as a status display
        self.lbl_residual.setText(msg + "\n" + self._residual_text())

    def _residual_text(self) -> str:
        residuals = self._compute_residuals_mm()
        if not residuals:
            return "Residuals: (no data)"
        lines = []
        for key, r in residuals.items():
            line = (f"{key}:  |Δmid|={r['norm_mm']:5.0f} mm  "
                    f"(Δ=[{r['dx_mm']:+5.0f}, {r['dy_mm']:+5.0f}, "
                    f"{r['dz_mm']:+5.0f}])")
            if "clubhead_norm_mm" in r:
                line += f"   |Δclub|={r['clubhead_norm_mm']:5.0f} mm"
            lines.append(line)
        return "\n".join(lines)

    # ===================================================================== #
    # Drawing                                                               #
    # ===================================================================== #

    def _redraw(self) -> None:
        elev, azim = self.ax.elev, self.ax.azim
        self.ax.clear()
        self._setup_axes()
        self.ax.view_init(elev=elev, azim=azim)

        self._draw_floor_and_ball()
        self._draw_traces()
        self._draw_visible_poses()

        self.lbl_residual.setText(self._residual_text())

        leg = self.ax.legend(loc="upper right", fontsize=8, ncol=1,
                             framealpha=0.85)
        if leg is not None:
            for text in leg.get_texts():
                text.set_color("#e6e6e6")
            leg.get_frame().set_facecolor("#1f242b")
            leg.get_frame().set_edgecolor("#404652")
        self.canvas.draw()

    def _draw_floor_and_ball(self) -> None:
        x = np.linspace(-1.5, 1.5, 5)
        y = np.linspace(-1.5, 1.5, 5)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        self.ax.plot_surface(X, Y, Z, alpha=0.10, color="#22c55e")
        self.ax.scatter([0], [0], [0.021], c="white", edgecolor="black", s=40,
                        label="ball")

    def _trace_window(self) -> tuple[int, int]:
        """Return [start, end) frame indices for trace drawing per phase setting."""
        if self.df is None:
            return (0, 0)
        n = len(self.df)
        bounds = _PHASE_WINDOWS.get(self.phase_window, (None, None))
        # "None" -> draw across full data
        if bounds == (None, None):
            return (0, n)
        # "Manual range"
        if bounds == ("manual", "manual"):
            i0 = max(0, min(self.manual_window_start, n - 1))
            i1 = max(0, min(self.manual_window_end + 1, n))
            if i1 <= i0:
                i1 = i0 + 1
            return (i0, i1)
        # Event-bounded
        a_label, b_label = bounds
        a = self._frame_for(str(a_label)) if a_label is not None else 0
        b = self._frame_for(str(b_label)) if b_label is not None else (n - 1)
        if a is None:
            a = 0
        if b is None:
            b = n - 1
        if b < a:
            a, b = 0, n - 1
        return (a, b + 1)

    def _draw_traces(self) -> None:
        if self.df is None:
            return
        i0, i1 = self._trace_window()
        sub = self.df.iloc[i0:i1]
        if self.show_midhands_trace and len(sub) > 1:
            self.ax.plot(-sub["mid_X"].values, sub["mid_Y"].values,
                         sub["mid_Z"].values, color="#7dd3fc", linestyle="--",
                         linewidth=1.2, alpha=0.85,
                         label="mocap mid-hands trace")
        if self.show_clubhead_trace and len(sub) > 1:
            self.ax.plot(-sub["club_X"].values, sub["club_Y"].values,
                         sub["club_Z"].values, color="#fb7185", linestyle="--",
                         linewidth=1.2, alpha=0.85,
                         label="mocap clubhead trace")
        # Phase boundary markers (start / end of selected window)
        if (self.show_midhands_trace or self.show_clubhead_trace) and len(sub) > 0:
            for idx, marker_label, color in [
                (i0, "start", "#22c55e"),
                (min(i1 - 1, len(self.df) - 1), "end", "#a855f7"),
            ]:
                row = self.df.iloc[idx]
                if self.show_midhands_trace:
                    self.ax.scatter(-row["mid_X"], row["mid_Y"], row["mid_Z"],
                                    color=color, s=40, marker="^",
                                    edgecolor="black", linewidth=0.5)
                if self.show_clubhead_trace:
                    self.ax.scatter(-row["club_X"], row["club_Y"], row["club_Z"],
                                    color=color, s=40, marker="^",
                                    edgecolor="black", linewidth=0.5)
        # Current-frame marker (cross)
        if (getattr(self, "cb_frame_marker", None) is not None and
                self.cb_frame_marker.isChecked() and
                self.df is not None and 0 <= self.current_frame < len(self.df)):
            row = self.df.iloc[self.current_frame]
            if self.show_midhands_trace:
                self.ax.scatter(-row["mid_X"], row["mid_Y"], row["mid_Z"],
                                color="#fde047", s=120, marker="x", linewidth=2,
                                label="current frame (mid)")
            if self.show_clubhead_trace:
                self.ax.scatter(-row["club_X"], row["club_Y"], row["club_Z"],
                                color="#fde047", s=140, marker="x", linewidth=2,
                                label="current frame (clubhead)")

    def _draw_visible_poses(self) -> None:
        for slot in self.poses.values():
            if not slot.visible:
                continue
            self._draw_one_pose(slot)

    def _draw_one_pose(self, slot: PoseSlot) -> None:
        if not slot.skeleton.joints:
            return

        mp = self._mocap_pos_for(slot, "mid")
        ch = self._mocap_pos_for(slot, "club")
        if mp is not None and ch is not None:
            pts = np.array([mp, ch])
            self.ax.plot(pts[:, 0], pts[:, 1], pts[:, 2],
                         color=slot.mocap_color, linewidth=4.5,
                         label=f"mocap {slot.name}")
            self.ax.scatter(*mp, color=slot.mocap_color, s=70, marker="o",
                            edgecolor="black", linewidth=0.6)
            self.ax.scatter(*ch, color=slot.mocap_color, s=130, marker="s",
                            edgecolor="black", linewidth=0.6)

        names = list(slot.skeleton.joints.keys())
        pts = np.array([slot.skeleton.joints[n] for n in names])
        moved = self.transform.apply(pts)
        pos = {n: moved[i] for i, n in enumerate(names)}

        for parent, child in slot.skeleton.segments:
            if parent in pos and child in pos:
                a, b = pos[parent], pos[child]
                width = 4.5 if (parent, child) == ("mp", "ch") else 2.6
                self.ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]],
                             color=slot.color, linewidth=width)
        self.ax.scatter(moved[:, 0], moved[:, 1], moved[:, 2],
                        color=slot.color, s=24,
                        label=f"sim {slot.name}")
        if "mp" in pos:
            self.ax.scatter(*pos["mp"], color=slot.color, s=70, marker="o",
                            edgecolor="black", linewidth=0.6)
        if "ch" in pos:
            self.ax.scatter(*pos["ch"], color=slot.color, s=130, marker="s",
                            edgecolor="black", linewidth=0.6)


# --------------------------------------------------------------------------- #
# Entrypoint                                                                  #
# --------------------------------------------------------------------------- #


def main() -> int:
    app = QApplication(sys.argv)
    if "Fusion" in QStyleFactory.keys():
        app.setStyle("Fusion")
    app.setStyleSheet(_QSS)
    base_font = QFont("Segoe UI", 10)
    app.setFont(base_font)
    win = StartingPoseMatcher()
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
