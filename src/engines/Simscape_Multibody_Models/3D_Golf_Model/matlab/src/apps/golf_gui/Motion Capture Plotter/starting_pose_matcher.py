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
from PyQt6.QtCore import Qt
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
    QStyleFactory,
    QVBoxLayout,
    QWidget,
)

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
        self.show_full_swing_window = True
        self.lock_xy_rotation = True   # Rx/Ry locked by default

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
        self.cb_swing_window = QCheckBox("Limit traces to swing window (A→F)")
        self.cb_swing_window.setChecked(True)
        self.cb_swing_window.stateChanged.connect(self._on_traces_toggled)
        v.addWidget(self.cb_swing_window)
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
        self.show_full_swing_window = self.cb_swing_window.isChecked()
        self._redraw()

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
        n = len(df)
        self.lbl_file.setText(f"{Path(path).name}\nsheet={sheet}  frames={n}")
        self.lbl_event_info.setText(self._events_summary())
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
        f = self._frame_for(slot.target_event)
        if f is None:
            return None
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
        if self.df is None:
            return (0, 0)
        n = len(self.df)
        if not self.show_full_swing_window:
            return (0, n)
        a = self._frame_for("A") or 0
        f = self._frame_for("F") or (n - 1)
        if f < a:
            a, f = 0, n - 1
        return (a, f + 1)

    def _draw_traces(self) -> None:
        if self.df is None:
            return
        if not (self.show_clubhead_trace or self.show_midhands_trace):
            return
        i0, i1 = self._trace_window()
        sub = self.df.iloc[i0:i1]
        if self.show_midhands_trace:
            self.ax.plot(-sub["mid_X"].values, sub["mid_Y"].values,
                         sub["mid_Z"].values, color="#7dd3fc", linestyle="--",
                         linewidth=1.2, alpha=0.8,
                         label="mocap mid-hands trace")
        if self.show_clubhead_trace:
            self.ax.plot(-sub["club_X"].values, sub["club_Y"].values,
                         sub["club_Z"].values, color="#fb7185", linestyle="--",
                         linewidth=1.2, alpha=0.8,
                         label="mocap clubhead trace")

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
