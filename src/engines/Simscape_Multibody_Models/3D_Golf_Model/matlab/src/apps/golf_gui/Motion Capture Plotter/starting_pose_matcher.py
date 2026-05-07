"""Starting-pose matcher: align Simscape golfer skeleton to mocap target frames.

This is a focused, iterative tool that lets you place the Simscape model in
the right starting pose BEFORE running any optimiser.  The pre-optimisation
flow that previously sent fmincon spinning into bad local minima was caused
by zero-theta starting the model far from the mocap target — this tool is
the fix.

Workflow:

    1. Loads the Wiffle ProV1 motion-capture xlsx (TW_ProV1 sheet by default).
    2. Reads the row-1 event header (A=address, T=top, I=impact, F=finish).
       NOTE: positions in the xlsx are in CENTIMETRES despite the
       "Definitions" tab claiming inches — see MATLAB_GOLF_MODEL_GUIDE.md
       § "Wiffle xlsx Definitions tab claims 'inches' but data is centimetres".
    3. Loads up to two pose skeletons (TopofBackswing + Impact) from
       simscape_skeleton_<pose>.json (produced by export_default_skeleton.m).
       Falls back to a hardcoded approximate pose so the tool runs without
       running MATLAB first.
    4. A single 7-DOF transform (Tx/Ty/Tz/Rx/Ry/Rz/Scale) is applied to
       every visible skeleton — toggle them on/off to verify the SAME
       global transform aligns both poses to their respective mocap frames.
    5. Save the transform to JSON; later it seeds model-workspace overrides
       in fit_swing_full_pipeline.

Run:
    cd ".../Motion Capture Plotter"
    python -m starting_pose_matcher
"""

from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")

# ----------------------------------------------------------------------------
# Units: Wiffle ProV1 xlsx positions are in CM (NOT inches) per the MATLAB
# loader (load_club_target_excel.m line 53: CM_TO_METRES = 0.01) and per
# matlab/MATLAB_GOLF_MODEL_GUIDE.md.  We deliberately bypass the legacy
# mocap_data_loader.py here because it uses the wrong INCHES_TO_METERS.
# ----------------------------------------------------------------------------
CM_TO_M = 0.01


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
    """7-DOF transform: Tx/Ty/Tz (m), Rx/Ry/Rz (deg), Scale (unitless).

    Applied as: P' = scale * R * (P - pivot) + pivot + t
    where R = Rz @ Ry @ Rx (intrinsic XYZ Euler).
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
        """Return (R_scaled, t) such that P' = R_scaled @ (P - pivot) + pivot + t."""
        cx, cy, cz = (np.cos(np.deg2rad(a)) for a in (self.rx, self.ry, self.rz))
        sx, sy, sz = (np.sin(np.deg2rad(a)) for a in (self.rx, self.ry, self.rz))
        Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
        R = Rz @ Ry @ Rx * float(self.scale)
        return R, np.array([self.tx, self.ty, self.tz])

    def apply(self, points: np.ndarray) -> np.ndarray:
        """Apply transform to (N, 3) array of points."""
        R, t = self.matrix()
        pivot = np.array(self.pivot)
        return (points - pivot) @ R.T + pivot + t


# --------------------------------------------------------------------------- #
# xlsx loader (CORRECT units — bypasses buggy legacy mocap_data_loader)       #
# --------------------------------------------------------------------------- #


def load_mocap_xlsx(xlsx_path: str | Path, sheet_name: str) -> pd.DataFrame:
    """Load Wiffle xlsx -> DataFrame in METRES.

    Schema (subset of what we need):
        time (s), mid_X/Y/Z (m), club_X/Y/Z (m), and orientation cols.
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
            rec: dict[str, float] = {
                "time":    time,
                "mid_X":   _safe(row, 2)  * CM_TO_M,
                "mid_Y":   _safe(row, 3)  * CM_TO_M,
                "mid_Z":   _safe(row, 4)  * CM_TO_M,
                "club_X":  _safe(row, 14) * CM_TO_M,
                "club_Y":  _safe(row, 15) * CM_TO_M,
                "club_Z":  _safe(row, 16) * CM_TO_M,
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


# Approximate Impact-pose joint positions (metres) — used until the user
# runs export_default_skeleton.m.  These are deliberately rough.
_FALLBACK_IMPACT: dict[str, list[float]] = {
    "hip":   [0.00, -0.30, 0.95],
    "spine": [0.00, -0.30, 1.20],
    "hub":   [0.00, -0.30, 1.40],
    "ls":    [-0.20, -0.30, 1.40],
    "rs":    [0.20, -0.30, 1.40],
    "le":    [-0.10, -0.20, 1.10],
    "re":    [0.10, -0.20, 1.10],
    "lw":    [-0.05, -0.10, 0.80],
    "rw":    [0.05, -0.10, 0.80],
    "mp":    [0.00, -0.10, 0.80],
    "ch":    [0.00, 0.10, 0.10],
}
# Approximate Top-of-Backswing pose (arms raised, club back).
_FALLBACK_TOB: dict[str, list[float]] = {
    "hip":   [0.00, -0.30, 0.95],
    "spine": [0.00, -0.30, 1.20],
    "hub":   [0.00, -0.30, 1.40],
    "ls":    [-0.20, -0.30, 1.40],
    "rs":    [0.20, -0.30, 1.40],
    "le":    [-0.05, -0.10, 1.55],
    "re":    [0.30, -0.10, 1.50],
    "lw":    [0.10,  0.10, 1.85],
    "rw":    [0.20,  0.10, 1.80],
    "mp":    [0.15,  0.10, 1.82],
    "ch":    [-0.40, 0.40, 1.60],
}
_FALLBACK_SEGMENTS: list[tuple[str, str]] = [
    ("hip", "spine"), ("spine", "hub"), ("hub", "ls"), ("hub", "rs"),
    ("ls", "le"), ("rs", "re"), ("le", "lw"), ("re", "rw"),
    ("lw", "mp"), ("rw", "mp"), ("mp", "ch"),
]


def load_skeleton(json_path: str | Path,
                  fallback_pose: str = "Impact") -> Skeleton:
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
        if not segments:
            segments = _FALLBACK_SEGMENTS
        return Skeleton(name=data.get("pose", fallback_pose),
                        joints=joints, segments=segments)

    logger.warning(
        "%s not found — using hardcoded fallback %s pose. Run "
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
# UI widgets                                                                  #
# --------------------------------------------------------------------------- #


_SLIDER_T_RANGE = (-1500, 1500)   # ±1.5 m, mm steps
_SLIDER_R_RANGE = (-720, 720)     # ±360 deg, 0.5-deg steps
_SLIDER_S_RANGE = (50, 200)       # 0.5x .. 2.0x, 0.01 steps
_SLIDER_T_SCALE = 0.001
_SLIDER_R_SCALE = 0.5
_SLIDER_S_SCALE = 0.01


class _LabelledSlider(QWidget):
    """Slider + spinbox, kept in sync. valueChanged exposed via spinbox."""

    def __init__(self, label: str, units: str, slider_range: tuple[int, int],
                 scale: float, decimals: int, default: float = 0.0,
                 parent: QWidget | None = None):
        super().__init__(parent)
        self._scale = scale
        layout = QGridLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)

        layout.addWidget(QLabel(label), 0, 0)
        self.spin = QDoubleSpinBox()
        self.spin.setDecimals(decimals)
        self.spin.setRange(slider_range[0] * scale, slider_range[1] * scale)
        self.spin.setSingleStep(scale)
        self.spin.setSuffix(f" {units}")
        self.spin.setMinimumWidth(110)
        layout.addWidget(self.spin, 0, 1)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(*slider_range)
        layout.addWidget(self.slider, 0, 2)

        self.slider.valueChanged.connect(
            lambda v: self.spin.setValue(v * self._scale))
        self.spin.valueChanged.connect(
            lambda v: self.slider.setValue(int(round(v / self._scale))))
        self.spin.setValue(default)

    def value(self) -> float:
        return float(self.spin.value())

    def set_value(self, v: float) -> None:
        self.spin.setValue(v)


# --------------------------------------------------------------------------- #
# Pose slot                                                                   #
# --------------------------------------------------------------------------- #


@dataclass
class PoseSlot:
    """One model pose + its mocap target frame."""
    name: str                            # display name e.g. "TopofBackswing"
    skeleton: Skeleton
    color: str                           # skeleton-line colour
    mocap_color: str                     # mocap-target colour
    target_event: str                    # default event label "A"/"T"/"I"/"F"
    visible: bool = True
    target_frame_override: int | None = None


# --------------------------------------------------------------------------- #
# Main window                                                                 #
# --------------------------------------------------------------------------- #


class StartingPoseMatcher(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Starting-Pose Matcher")
        self.resize(1600, 950)

        # Data
        self.df: pd.DataFrame | None = None
        self.events = MocapEvents()
        self._xlsx_path: str | None = None

        here = Path(__file__).parent
        self.poses: dict[str, PoseSlot] = {
            "TopofBackswing": PoseSlot(
                name="TopofBackswing",
                skeleton=load_skeleton(here / "simscape_skeleton_TopofBackswing.json",
                                       "TopofBackswing"),
                color="blue",
                mocap_color="red",
                target_event="T",
            ),
            "Impact": PoseSlot(
                name="Impact",
                skeleton=load_skeleton(here / "simscape_skeleton_Impact.json",
                                       "Impact"),
                color="darkgreen",
                mocap_color="orange",
                target_event="I",
            ),
        }

        self.transform = RigidTransform()
        # Pivot defaults to first available skeleton's hub.
        for slot in self.poses.values():
            if "hub" in slot.skeleton.joints:
                self.transform.pivot = tuple(slot.skeleton.joints["hub"])
                break

        # Trace toggles
        self.show_clubhead_trace = False
        self.show_midhands_trace = False
        self.show_full_swing_window = True   # only draw the swing window (A..F)

        self._build_ui()

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

        left = QVBoxLayout()
        title = QLabel("Starting-Pose Matcher")
        title.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        left.addWidget(title)

        left.addWidget(self._build_file_box())
        left.addWidget(self._build_pose_box())
        left.addWidget(self._build_traces_box())
        left.addWidget(self._build_transform_box())
        left.addWidget(self._build_save_box())
        left.addStretch()

        lwidget = QWidget()
        lwidget.setLayout(left)
        lwidget.setMaximumWidth(450)
        root.addWidget(lwidget)

        plot = QWidget()
        pl = QVBoxLayout(plot)
        self.fig = Figure(figsize=(10, 8), dpi=100)
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.canvas = FigureCanvas(self.fig)
        pl.addWidget(self.canvas)
        root.addWidget(plot, stretch=1)

        self._setup_axes()

    def _build_file_box(self) -> QGroupBox:
        box = QGroupBox("Mocap source (units: cm in xlsx → m here)")
        gl = QGridLayout(box)
        self.btn_load = QPushButton("Load xlsx…")
        self.btn_load.clicked.connect(self._on_load_clicked)
        gl.addWidget(self.btn_load, 0, 0, 1, 2)
        gl.addWidget(QLabel("Sheet:"), 1, 0)
        self.sheet_combo = QComboBox()
        self.sheet_combo.addItems(["TW_ProV1", "TW_wiffle", "GW_wiffle", "GW_ProV11"])
        self.sheet_combo.currentTextChanged.connect(self._on_sheet_changed)
        gl.addWidget(self.sheet_combo, 1, 1)
        self.lbl_file = QLabel("(no file loaded)")
        self.lbl_file.setWordWrap(True)
        gl.addWidget(self.lbl_file, 2, 0, 1, 2)
        self.lbl_event_info = QLabel("Events: (none)")
        self.lbl_event_info.setWordWrap(True)
        gl.addWidget(self.lbl_event_info, 3, 0, 1, 2)
        return box

    def _build_pose_box(self) -> QGroupBox:
        box = QGroupBox("Pose slots (visible × event)")
        gl = QGridLayout(box)
        self._pose_visible_checks: dict[str, QCheckBox] = {}
        self._pose_event_combos: dict[str, QComboBox] = {}
        gl.addWidget(QLabel("Show"), 0, 0)
        gl.addWidget(QLabel("Pose"),  0, 1)
        gl.addWidget(QLabel("Event"), 0, 2)
        gl.addWidget(QLabel("Reload"), 0, 3)
        for r, (key, slot) in enumerate(self.poses.items(), start=1):
            cb = QCheckBox()
            cb.setChecked(slot.visible)
            cb.stateChanged.connect(self._on_pose_toggled)
            self._pose_visible_checks[key] = cb
            gl.addWidget(cb, r, 0)
            lbl = QLabel(f"{key} ({slot.color})")
            gl.addWidget(lbl, r, 1)
            ec = QComboBox()
            ec.addItems(["A", "T", "I", "F"])
            ec.setCurrentText(slot.target_event)
            ec.currentTextChanged.connect(self._on_pose_event_changed)
            self._pose_event_combos[key] = ec
            gl.addWidget(ec, r, 2)
            btn = QPushButton("⟳")
            btn.setToolTip(f"Reload simscape_skeleton_{key}.json")
            btn.clicked.connect(lambda _checked, k=key: self._reload_pose(k))
            gl.addWidget(btn, r, 3)
        return box

    def _build_traces_box(self) -> QGroupBox:
        box = QGroupBox("Mocap traces (full timeline)")
        gl = QGridLayout(box)
        self.cb_clubhead_trace = QCheckBox("Show clubhead path")
        self.cb_clubhead_trace.stateChanged.connect(self._on_traces_toggled)
        gl.addWidget(self.cb_clubhead_trace, 0, 0)
        self.cb_midhands_trace = QCheckBox("Show mid-hands path")
        self.cb_midhands_trace.stateChanged.connect(self._on_traces_toggled)
        gl.addWidget(self.cb_midhands_trace, 1, 0)
        self.cb_swing_window = QCheckBox("Limit to swing window (A→F)")
        self.cb_swing_window.setChecked(True)
        self.cb_swing_window.stateChanged.connect(self._on_traces_toggled)
        gl.addWidget(self.cb_swing_window, 2, 0)
        return box

    def _build_transform_box(self) -> QGroupBox:
        box = QGroupBox("Rigid transform + scale (applied to all visible skeletons)")
        v = QVBoxLayout(box)
        self.s_tx = _LabelledSlider("Tx", "m", _SLIDER_T_RANGE, _SLIDER_T_SCALE, 3)
        self.s_ty = _LabelledSlider("Ty", "m", _SLIDER_T_RANGE, _SLIDER_T_SCALE, 3)
        self.s_tz = _LabelledSlider("Tz", "m", _SLIDER_T_RANGE, _SLIDER_T_SCALE, 3)
        self.s_rx = _LabelledSlider("Rx", "°", _SLIDER_R_RANGE, _SLIDER_R_SCALE, 1)
        self.s_ry = _LabelledSlider("Ry", "°", _SLIDER_R_RANGE, _SLIDER_R_SCALE, 1)
        self.s_rz = _LabelledSlider("Rz", "°", _SLIDER_R_RANGE, _SLIDER_R_SCALE, 1)
        self.s_scale = _LabelledSlider("Scale", "×", _SLIDER_S_RANGE, _SLIDER_S_SCALE, 2,
                                       default=1.0)
        for s in (self.s_tx, self.s_ty, self.s_tz, self.s_rx, self.s_ry, self.s_rz,
                  self.s_scale):
            v.addWidget(s)
            s.spin.valueChanged.connect(self._on_transform_changed)
        v.addWidget(QLabel(
            "Pivot @ hub of first visible pose: ({:.3f}, {:.3f}, {:.3f})".format(
                *self.transform.pivot)))
        btn_row = QHBoxLayout()
        self.btn_snap_visible = QPushButton("Snap grip → mocap mid (visible pose)")
        self.btn_snap_visible.setToolTip("Solve Tx/Ty/Tz so the FIRST visible "
                                         "skeleton's mid-hands lands on its mocap "
                                         "target.")
        self.btn_snap_visible.clicked.connect(self._snap_first_visible)
        btn_row.addWidget(self.btn_snap_visible)
        self.btn_reset = QPushButton("Reset")
        self.btn_reset.clicked.connect(self._reset_transforms)
        btn_row.addWidget(self.btn_reset)
        v.addLayout(btn_row)
        return box

    def _build_save_box(self) -> QGroupBox:
        box = QGroupBox("Output / status")
        v = QVBoxLayout(box)
        self.btn_save = QPushButton("Save offsets to JSON…")
        self.btn_save.clicked.connect(self._on_save_clicked)
        v.addWidget(self.btn_save)
        self.lbl_residual = QLabel("Residuals: (no data)")
        self.lbl_residual.setWordWrap(True)
        v.addWidget(self.lbl_residual)
        return box

    def _setup_axes(self) -> None:
        self.ax.set_xlabel("X (target line)")
        self.ax.set_ylabel("Y (ball direction)")
        self.ax.set_zlabel("Z (vertical)")
        # Wiffle ProV1 mocap data spans roughly:
        #   X = ±1.6 m  (clubhead arc)
        #   Y =  ±1.6 m
        #   Z = -1.0 to +1.4 m  (mocap origin is NOT at ground; positions
        #                        are relative to the recorder's reference)
        self.ax.set_xlim(-2.0, 2.0)
        self.ax.set_ylim(-1.5, 2.0)
        self.ax.set_zlim(-1.5, 2.5)
        try:
            self.ax.set_box_aspect((4, 3.5, 4))
        except AttributeError:
            pass

    # ===================================================================== #
    # Event handlers                                                        #
    # ===================================================================== #

    def _on_load_clicked(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Wiffle xlsx",
            str(Path(__file__).parent),
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

    def _on_transform_changed(self, _: float) -> None:
        self.transform.tx = self.s_tx.value()
        self.transform.ty = self.s_ty.value()
        self.transform.tz = self.s_tz.value()
        self.transform.rx = self.s_rx.value()
        self.transform.ry = self.s_ry.value()
        self.transform.rz = self.s_rz.value()
        self.transform.scale = max(1e-3, self.s_scale.value())
        self._redraw()

    def _reset_transforms(self) -> None:
        for s in (self.s_tx, self.s_ty, self.s_tz, self.s_rx, self.s_ry, self.s_rz):
            s.set_value(0.0)
        self.s_scale.set_value(1.0)

    def _snap_first_visible(self) -> None:
        slot = self._first_visible_pose()
        if slot is None or "mp" not in slot.skeleton.joints:
            return
        target = self._mocap_pos_for(slot, "mid")
        if target is None:
            return
        # Apply current rotation+scale (but no translation) and compute the
        # translation needed to land mp on target.
        dummy = RigidTransform(
            rx=self.s_rx.value(), ry=self.s_ry.value(), rz=self.s_rz.value(),
            scale=max(1e-3, self.s_scale.value()), pivot=self.transform.pivot)
        rotated_mp = dummy.apply(slot.skeleton.joints["mp"][None, :])[0]
        delta = target - rotated_mp
        self.s_tx.set_value(delta[0])
        self.s_ty.set_value(delta[1])
        self.s_tz.set_value(delta[2])

    def _reload_pose(self, key: str) -> None:
        path = Path(__file__).parent / f"simscape_skeleton_{key}.json"
        self.poses[key].skeleton = load_skeleton(path, key)
        # Reset pivot to the hub of this pose if it was the source.
        if self._first_visible_pose() and self._first_visible_pose().name == key:
            hub = self.poses[key].skeleton.joints.get("hub")
            if hub is not None:
                self.transform.pivot = tuple(hub)
        self._redraw()

    # ===================================================================== #
    # Loading                                                               #
    # ===================================================================== #

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
        parts: list[str] = []
        for label in ("A", "T", "I", "F"):
            v = getattr(e, f"{label}_sample")
            parts.append(f"{label}={'?' if v != v else int(v)}")
        if e.CHS_mph == e.CHS_mph:
            parts.append(f"CHS={e.CHS_mph:.1f}mph")
        return "Events: " + " ".join(parts)

    # ===================================================================== #
    # Save                                                                  #
    # ===================================================================== #

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
                "units": {"translation": "metres", "rotation": "degrees",
                          "rotation_order": "Rz @ Ry @ Rx (intrinsic XYZ)"}},
            "poses": {key: {
                "visible": slot.visible,
                "event": slot.target_event,
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
        logger.info("Wrote %s", path)

    # ===================================================================== #
    # Helpers                                                               #
    # ===================================================================== #

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
        """kind ∈ {'mid','club'} — return the mocap point at the slot's event frame."""
        if self.df is None:
            return None
        f = (slot.target_frame_override
             if slot.target_frame_override is not None
             else self._frame_for(slot.target_event))
        if f is None:
            return None
        row = self.df.iloc[f]
        if kind == "mid":
            return np.array([-row["mid_X"], row["mid_Y"], row["mid_Z"]])
        return np.array([-row["club_X"], row["club_Y"], row["club_Z"]])

    def _compute_residuals_mm(self) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for key, slot in self.poses.items():
            mp = slot.skeleton.joints.get("mp")
            if mp is None:
                continue
            target = self._mocap_pos_for(slot, "mid")
            if target is None:
                continue
            moved = self.transform.apply(mp[None, :])[0]
            delta = (moved - target) * 1000.0
            out[key] = {
                "dx_mm": float(delta[0]), "dy_mm": float(delta[1]),
                "dz_mm": float(delta[2]),
                "norm_mm": float(np.linalg.norm(delta)),
            }
        return out

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
        self._update_residual_label()
        self.ax.legend(loc="upper right", fontsize=7, ncol=2)
        self.canvas.draw()

    def _draw_floor_and_ball(self) -> None:
        x = np.linspace(-1.5, 1.5, 5)
        y = np.linspace(-1.5, 1.5, 5)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        self.ax.plot_surface(X, Y, Z, alpha=0.12, color="green")
        self.ax.scatter([0], [0], [0.021], c="white", edgecolor="black", s=40,
                        label="ball")

    def _trace_window(self) -> tuple[int, int]:
        """Slice indices for clubhead/mid traces."""
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
                         sub["mid_Z"].values, "b--", linewidth=1.0, alpha=0.6,
                         label="mocap mid-hands trace")
        if self.show_clubhead_trace:
            self.ax.plot(-sub["club_X"].values, sub["club_Y"].values,
                         sub["club_Z"].values, "r--", linewidth=1.0, alpha=0.6,
                         label="mocap clubhead trace")

    def _draw_visible_poses(self) -> None:
        for slot in self.poses.values():
            if not slot.visible:
                continue
            self._draw_one_pose(slot)

    def _draw_one_pose(self, slot: PoseSlot) -> None:
        if not slot.skeleton.joints:
            return

        # Mocap target (red/orange + frame label)
        mp = self._mocap_pos_for(slot, "mid")
        ch = self._mocap_pos_for(slot, "club")
        if mp is not None and ch is not None:
            pts = np.array([mp, ch])
            self.ax.plot(pts[:, 0], pts[:, 1], pts[:, 2],
                         color=slot.mocap_color, linewidth=4,
                         label=f"mocap {slot.name}")
            self.ax.scatter(*mp, color=slot.mocap_color, s=70, marker="o")
            self.ax.scatter(*ch, color=slot.mocap_color, s=120, marker="s")

        # Skeleton (transformed)
        names = list(slot.skeleton.joints.keys())
        pts = np.array([slot.skeleton.joints[n] for n in names])
        moved = self.transform.apply(pts)
        pos = {n: moved[i] for i, n in enumerate(names)}

        for parent, child in slot.skeleton.segments:
            if parent in pos and child in pos:
                a, b = pos[parent], pos[child]
                width = 4 if (parent, child) == ("mp", "ch") else 2.4
                self.ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]],
                             color=slot.color, linewidth=width)
        self.ax.scatter(moved[:, 0], moved[:, 1], moved[:, 2],
                        color=slot.color, s=24, label=f"sim {slot.name}")
        if "mp" in pos:
            self.ax.scatter(*pos["mp"], color=slot.color, s=70, marker="o",
                            edgecolor="black")
        if "ch" in pos:
            self.ax.scatter(*pos["ch"], color=slot.color, s=120, marker="s",
                            edgecolor="black")

    def _update_residual_label(self) -> None:
        residuals = self._compute_residuals_mm()
        if not residuals:
            self.lbl_residual.setText("Residuals: (no data)")
            return
        lines = []
        for key, r in residuals.items():
            lines.append(
                f"{key}: |Δ|={r['norm_mm']:.0f}mm  "
                f"(Δ=[{r['dx_mm']:+.0f}, {r['dy_mm']:+.0f}, {r['dz_mm']:+.0f}])"
            )
        self.lbl_residual.setText("\n".join(lines))


# --------------------------------------------------------------------------- #
# Entrypoint                                                                  #
# --------------------------------------------------------------------------- #


def main() -> int:
    app = QApplication(sys.argv)
    win = StartingPoseMatcher()
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
