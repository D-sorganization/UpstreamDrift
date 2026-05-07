"""Starting-pose matcher: align Simscape golfer skeleton to mocap target.

Workflow:
    1. Loads the Wiffle ProV1 motion-capture xlsx (TW_ProV1 sheet by default).
    2. Reads the row-1 event header (A=address, T=top of backswing, I=impact, F=finish).
    3. Loads the Simscape default-pose skeleton from simscape_default_skeleton.json
       (produced by export_default_skeleton.m) — falls back to a hardcoded
       approximate pose if that file isn't there yet.
    4. Plots both in 3D, with the Simscape skeleton drawn under a 6-DOF rigid
       transform that the user controls live with sliders.
    5. "Save Offsets" writes the chosen Tx/Ty/Tz/Rx/Ry/Rz to a JSON file that
       can later seed the model-workspace overrides for fit_swing_full_pipeline.

Usage:
    cd ".../Motion Capture Plotter"
    python3 -m starting_pose_matcher
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

# Allow running as a script too
try:
    from .mocap_data_loader import process_excel_sheet
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from mocap_data_loader import process_excel_sheet

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")

# Conversion factor: inches to metres (mocap data is in inches)
INCHES_TO_M = 0.0254


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
        # Excel sample numbers in row 1 are 1-based — convert.
        return max(0, int(v) - 1)


@dataclass
class Skeleton:
    """Default Simscape skeleton at t=0 (joint world positions in metres)."""

    joints: dict[str, np.ndarray] = field(default_factory=dict)
    # List of (parent, child) joint-name pairs to draw as line segments.
    segments: list[tuple[str, str]] = field(default_factory=list)


@dataclass
class RigidTransform:
    """6-DOF rigid transform applied around a pivot point.

    Translation in metres, rotation Euler angles in degrees (XYZ intrinsic).
    """

    tx: float = 0.0
    ty: float = 0.0
    tz: float = 0.0
    rx: float = 0.0
    ry: float = 0.0
    rz: float = 0.0
    pivot: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def matrix(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (R, t) where each transformed point P' = R @ (P - pivot) + pivot + t."""
        cx, cy, cz = (np.cos(np.deg2rad(a)) for a in (self.rx, self.ry, self.rz))
        sx, sy, sz = (np.sin(np.deg2rad(a)) for a in (self.rx, self.ry, self.rz))
        Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
        R = Rz @ Ry @ Rx
        return R, np.array([self.tx, self.ty, self.tz])

    def apply(self, points: np.ndarray) -> np.ndarray:
        """Apply transform to (N, 3) array of points."""
        R, t = self.matrix()
        pivot = np.array(self.pivot)
        return (points - pivot) @ R.T + pivot + t


# --------------------------------------------------------------------------- #
# Excel events loader                                                         #
# --------------------------------------------------------------------------- #


def read_event_header(xlsx_path: str | Path, sheet_name: str) -> MocapEvents:
    """Parse the row-1 event-marker band of a Wiffle xlsx sheet.

    Mirrors load_club_target_excel.m's local_read_event_header — row 1 has
    label/value pairs: A=<n> T=<n> I=<n> F=<n> CHS=<mph>.
    """
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
# Skeleton loader (with hardcoded fallback)                                   #
# --------------------------------------------------------------------------- #


# Approximate joint positions for a typical impact-pose Simscape golfer,
# used when simscape_default_skeleton.json hasn't been exported yet.
# Coordinates: X = target line (towards target), Y = ball-direction depth,
# Z = vertical up. Ball at origin. (Matches motion_capture_plotter conventions.)
_FALLBACK_SKELETON: dict[str, list[float]] = {
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
_FALLBACK_SEGMENTS: list[tuple[str, str]] = [
    ("hip", "spine"), ("spine", "hub"), ("hub", "ls"), ("hub", "rs"),
    ("ls", "le"), ("rs", "re"), ("le", "lw"), ("re", "rw"),
    ("lw", "mp"), ("rw", "mp"), ("mp", "ch"),
]


def load_skeleton(json_path: str | Path | None = None) -> Skeleton:
    """Load skeleton from JSON; fall back to hardcoded approximate pose."""
    if json_path is None:
        json_path = Path(__file__).with_name("simscape_default_skeleton.json")
    json_path = Path(json_path)
    if not json_path.exists():
        logger.warning(
            "%s not found — using hardcoded approximate skeleton.\n"
            "Run export_default_skeleton.m in MATLAB to generate the real one.",
            json_path,
        )
        return Skeleton(
            joints={k: np.array(v, dtype=float) for k, v in _FALLBACK_SKELETON.items()},
            segments=_FALLBACK_SEGMENTS,
        )
    with open(json_path) as f:
        data = json.load(f)
    joints = {k: np.array(v, dtype=float) for k, v in data["joints"].items()}
    raw_segments = data.get("segments", [])
    segments: list[tuple[str, str]] = []
    for s in raw_segments:
        # JSON cell-arrays from MATLAB serialize as lists of lists.
        if isinstance(s, list) and len(s) == 2:
            segments.append((str(s[0]), str(s[1])))
    if not segments:
        segments = _FALLBACK_SEGMENTS
    return Skeleton(joints=joints, segments=segments)


# --------------------------------------------------------------------------- #
# UI                                                                          #
# --------------------------------------------------------------------------- #


# Slider granularity: 1 unit = 1 mm (translation) or 0.5 deg (rotation).
_SLIDER_T_RANGE = (-1500, 1500)   # ±1.5 m in mm steps
_SLIDER_R_RANGE = (-720, 720)     # ±360 deg in 0.5-deg steps
_SLIDER_T_SCALE = 0.001            # mm -> m
_SLIDER_R_SCALE = 0.5              # 0.5 deg per tick


class _LabelledSlider(QWidget):
    """Slider + label + double-spinbox, all kept in sync.

    Emits valueChanged(float) via the spinbox.
    """

    def __init__(self, label: str, units: str, slider_range: tuple[int, int],
                 scale: float, decimals: int, parent: QWidget | None = None):
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
        layout.addWidget(self.spin, 0, 1)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(*slider_range)
        layout.addWidget(self.slider, 0, 2)

        # Two-way binding
        self.slider.valueChanged.connect(
            lambda v: self.spin.setValue(v * self._scale))
        self.spin.valueChanged.connect(
            lambda v: self.slider.setValue(int(round(v / self._scale))))

    def value(self) -> float:
        return float(self.spin.value())

    def set_value(self, v: float) -> None:
        self.spin.setValue(v)


class StartingPoseMatcher(QMainWindow):
    """Standalone tool to align the Simscape skeleton to a mocap target frame."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Starting-Pose Matcher")
        self.resize(1500, 900)

        # Data
        self.df: pd.DataFrame | None = None
        self.events = MocapEvents()
        self.skeleton: Skeleton = load_skeleton()
        self.transform = RigidTransform()
        # Pivot defaults to skeleton hub (so rotations feel natural)
        if "hub" in self.skeleton.joints:
            self.transform.pivot = tuple(self.skeleton.joints["hub"])

        # UI
        self._build_ui()

        # Auto-load default xlsx in the same directory.
        default_xlsx = Path(__file__).with_name("Wiffle_ProV1_club_3D_data.xlsx")
        if default_xlsx.exists():
            self._load_xlsx(str(default_xlsx))

    # -- UI building -------------------------------------------------------- #

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)

        # -- Left control column ------------------------------------------- #
        left = QVBoxLayout()
        title = QLabel("Starting-Pose Matcher")
        title.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        left.addWidget(title)

        # File / sheet
        file_box = QGroupBox("Mocap source")
        fl = QGridLayout(file_box)
        self.btn_load = QPushButton("Load xlsx…")
        self.btn_load.clicked.connect(self._on_load_clicked)
        fl.addWidget(self.btn_load, 0, 0, 1, 2)
        fl.addWidget(QLabel("Sheet:"), 1, 0)
        self.sheet_combo = QComboBox()
        self.sheet_combo.addItems(["TW_ProV1", "TW_wiffle", "GW_wiffle", "GW_ProV11"])
        self.sheet_combo.currentTextChanged.connect(self._on_sheet_changed)
        fl.addWidget(self.sheet_combo, 1, 1)
        self.lbl_file = QLabel("(no file loaded)")
        self.lbl_file.setWordWrap(True)
        fl.addWidget(self.lbl_file, 2, 0, 1, 2)
        left.addWidget(file_box)

        # Frame jump
        jump_box = QGroupBox("Jump to event")
        jl = QGridLayout(jump_box)
        button_titles = {"A": "A\nAddress", "T": "T\nTop", "I": "I\nImpact", "F": "F\nFinish"}
        for col, label in enumerate(["A", "T", "I", "F"]):
            btn = QPushButton(button_titles[label])
            btn.clicked.connect(lambda _checked, ev=label: self._jump_to_event(ev))
            jl.addWidget(btn, 0, col)
        # Frame slider + spin
        jl.addWidget(QLabel("Frame:"), 1, 0)
        self.frame_spin = QDoubleSpinBox()
        self.frame_spin.setDecimals(0)
        self.frame_spin.setRange(0, 0)
        self.frame_spin.setSingleStep(1)
        jl.addWidget(self.frame_spin, 1, 1)
        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setRange(0, 0)
        jl.addWidget(self.frame_slider, 1, 2, 1, 2)
        self.frame_slider.valueChanged.connect(
            lambda v: self.frame_spin.setValue(float(v)))
        self.frame_spin.valueChanged.connect(
            lambda v: self.frame_slider.setValue(int(v)))
        self.frame_spin.valueChanged.connect(self._on_frame_changed)
        self.lbl_event_info = QLabel("Events: (none)")
        self.lbl_event_info.setWordWrap(True)
        jl.addWidget(self.lbl_event_info, 2, 0, 1, 4)
        left.addWidget(jump_box)

        # Rigid-transform sliders
        tx_box = QGroupBox("Rigid transform (skeleton overlay)")
        txl = QVBoxLayout(tx_box)
        self.s_tx = _LabelledSlider("Tx", "m", _SLIDER_T_RANGE, _SLIDER_T_SCALE, 3)
        self.s_ty = _LabelledSlider("Ty", "m", _SLIDER_T_RANGE, _SLIDER_T_SCALE, 3)
        self.s_tz = _LabelledSlider("Tz", "m", _SLIDER_T_RANGE, _SLIDER_T_SCALE, 3)
        self.s_rx = _LabelledSlider("Rx", "°", _SLIDER_R_RANGE, _SLIDER_R_SCALE, 1)
        self.s_ry = _LabelledSlider("Ry", "°", _SLIDER_R_RANGE, _SLIDER_R_SCALE, 1)
        self.s_rz = _LabelledSlider("Rz", "°", _SLIDER_R_RANGE, _SLIDER_R_SCALE, 1)
        for s in (self.s_tx, self.s_ty, self.s_tz, self.s_rx, self.s_ry, self.s_rz):
            txl.addWidget(s)
            s.spin.valueChanged.connect(self._on_transform_changed)
        # Pivot info
        pivot_str = ("Pivot @ hub: ({:.3f}, {:.3f}, {:.3f})".format(*self.transform.pivot)
                     if "hub" in self.skeleton.joints else "Pivot @ origin")
        txl.addWidget(QLabel(pivot_str))
        # Auto-snap + reset buttons
        btn_row = QHBoxLayout()
        self.btn_snap = QPushButton("Snap grip → mocap mid")
        self.btn_snap.setToolTip("Set Tx/Ty/Tz so the skeleton's mid-hands sits on "
                                 "the mocap mid-hands of the current frame.")
        self.btn_snap.clicked.connect(self._snap_to_mocap_grip)
        btn_row.addWidget(self.btn_snap)
        self.btn_reset = QPushButton("Reset transforms")
        self.btn_reset.clicked.connect(self._reset_transforms)
        btn_row.addWidget(self.btn_reset)
        txl.addLayout(btn_row)
        left.addWidget(tx_box)

        # Save
        save_box = QGroupBox("Output")
        sl = QVBoxLayout(save_box)
        self.btn_save = QPushButton("Save offsets to JSON…")
        self.btn_save.clicked.connect(self._on_save_clicked)
        sl.addWidget(self.btn_save)
        self.lbl_residual = QLabel("Grip residual: (no data)")
        sl.addWidget(self.lbl_residual)
        left.addWidget(save_box)

        left.addStretch()

        left_widget = QWidget()
        left_widget.setLayout(left)
        left_widget.setMaximumWidth(420)
        root.addWidget(left_widget)

        # -- Right plot column --------------------------------------------- #
        plot_widget = QWidget()
        pl = QVBoxLayout(plot_widget)
        self.fig = Figure(figsize=(10, 8), dpi=100)
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.canvas = FigureCanvas(self.fig)
        pl.addWidget(self.canvas)
        root.addWidget(plot_widget, stretch=1)

        self._setup_axes()

    def _setup_axes(self) -> None:
        self.ax.set_xlabel("X (target line)")
        self.ax.set_ylabel("Y (ball direction)")
        self.ax.set_zlabel("Z (vertical)")
        self.ax.set_xlim(-1.5, 1.5)
        self.ax.set_ylim(-1.5, 1.5)
        self.ax.set_zlim(0.0, 2.5)

    # -- Loading ------------------------------------------------------------ #

    def _on_load_clicked(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Wiffle xlsx",
            os.path.dirname(os.path.abspath(__file__)),
            "Excel files (*.xlsx *.xls)")
        if path:
            self._load_xlsx(path)

    def _on_sheet_changed(self, _: str) -> None:
        if hasattr(self, "_xlsx_path"):
            self._load_xlsx(self._xlsx_path)

    def _load_xlsx(self, path: str) -> None:
        sheet = self.sheet_combo.currentText()
        try:
            df = process_excel_sheet(path, sheet)
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
        self.frame_slider.setRange(0, n - 1)
        self.frame_spin.setRange(0, n - 1)
        self.lbl_file.setText(f"{Path(path).name}\nsheet={sheet}  frames={n}")
        self.lbl_event_info.setText(self._events_summary())
        # Default jump: top of backswing if available, else address, else 0
        for ev in ("T", "A", "I"):
            f = self.events.frame_for(ev)
            if f is not None:
                self.frame_spin.setValue(float(f))
                break
        else:
            self.frame_spin.setValue(0.0)
        self._redraw()

    def _events_summary(self) -> str:
        e = self.events
        parts: list[str] = []
        for label in ("A", "T", "I", "F"):
            v = getattr(e, f"{label}_sample")
            parts.append(f"{label}={'?' if v != v else int(v)}")
        if e.CHS_mph == e.CHS_mph:
            parts.append(f"CHS={e.CHS_mph:.1f} mph")
        return "Events: " + "  ".join(parts)

    # -- Frame / events ----------------------------------------------------- #

    def _jump_to_event(self, label: str) -> None:
        f = self.events.frame_for(label)
        if f is None:
            self.lbl_event_info.setText(f"{self._events_summary()}\n"
                                        f"(no {label} marker in sheet)")
            return
        f = max(0, min(f, int(self.frame_spin.maximum())))
        self.frame_spin.setValue(float(f))

    def _on_frame_changed(self, _: float) -> None:
        self._redraw()

    # -- Transform ---------------------------------------------------------- #

    def _on_transform_changed(self, _: float) -> None:
        self.transform.tx = self.s_tx.value()
        self.transform.ty = self.s_ty.value()
        self.transform.tz = self.s_tz.value()
        self.transform.rx = self.s_rx.value()
        self.transform.ry = self.s_ry.value()
        self.transform.rz = self.s_rz.value()
        self._redraw()

    def _reset_transforms(self) -> None:
        for s in (self.s_tx, self.s_ty, self.s_tz, self.s_rx, self.s_ry, self.s_rz):
            s.set_value(0.0)

    def _snap_to_mocap_grip(self) -> None:
        """Set translation so skeleton mp lands on mocap mid-hands; keep rotations."""
        target = self._mocap_grip()
        if target is None or "mp" not in self.skeleton.joints:
            return
        # Apply current rotation (with no translation) and snap residual.
        no_t = RigidTransform(rx=self.s_rx.value(), ry=self.s_ry.value(),
                              rz=self.s_rz.value(), pivot=self.transform.pivot)
        rotated_mp = no_t.apply(self.skeleton.joints["mp"][None, :])[0]
        delta = target - rotated_mp
        self.s_tx.set_value(delta[0])
        self.s_ty.set_value(delta[1])
        self.s_tz.set_value(delta[2])

    # -- Save --------------------------------------------------------------- #

    def _on_save_clicked(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save offsets",
            os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "starting_pose_offsets.json"),
            "JSON (*.json)")
        if not path:
            return
        out = {
            "transform": {
                "tx": self.transform.tx, "ty": self.transform.ty,
                "tz": self.transform.tz, "rx": self.transform.rx,
                "ry": self.transform.ry, "rz": self.transform.rz,
                "pivot": list(self.transform.pivot),
                "units": {"translation": "metres", "rotation": "degrees",
                          "rotation_order": "Rz @ Ry @ Rx (intrinsic XYZ)"}},
            "frame": int(self.frame_spin.value()),
            "events": {
                "A_sample": self.events.A_sample, "T_sample": self.events.T_sample,
                "I_sample": self.events.I_sample, "F_sample": self.events.F_sample,
                "CHS_mph": self.events.CHS_mph},
            "mocap_target": self._mocap_target_summary(),
        }
        with open(path, "w") as f:
            json.dump(out, f, indent=2)
        logger.info("Wrote %s", path)
        self.lbl_residual.setText(self.lbl_residual.text() + f"\nSaved → {Path(path).name}")

    def _mocap_target_summary(self) -> dict[str, Any]:
        mp = self._mocap_grip()
        ch = self._mocap_clubhead()
        return {"mid_hands": mp.tolist() if mp is not None else None,
                "club_head": ch.tolist() if ch is not None else None}

    # -- Mocap helpers ------------------------------------------------------ #

    def _current_row(self) -> pd.Series | None:
        if self.df is None or self.df.empty:
            return None
        i = int(self.frame_spin.value())
        i = max(0, min(i, len(self.df) - 1))
        return self.df.iloc[i]

    def _mocap_grip(self) -> np.ndarray | None:
        row = self._current_row()
        if row is None:
            return None
        # Note: process_excel_sheet already converts inches -> metres.
        # Match motion_capture_plotter convention: flip X for right-handed.
        return np.array([-row["mid_X"], row["mid_Y"], row["mid_Z"]])

    def _mocap_clubhead(self) -> np.ndarray | None:
        row = self._current_row()
        if row is None:
            return None
        return np.array([-row["club_X"], row["club_Y"], row["club_Z"]])

    # -- Drawing ------------------------------------------------------------ #

    def _redraw(self) -> None:
        elev, azim = self.ax.elev, self.ax.azim
        self.ax.clear()
        self._setup_axes()
        self.ax.view_init(elev=elev, azim=azim)

        self._draw_floor_and_ball()
        self._draw_mocap_target()
        self._draw_skeleton_overlay()
        self._update_residual()

        self.ax.legend(loc="upper right", fontsize=8)
        self.canvas.draw()

    def _draw_floor_and_ball(self) -> None:
        x = np.linspace(-1.5, 1.5, 5)
        y = np.linspace(-1.5, 1.5, 5)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        self.ax.plot_surface(X, Y, Z, alpha=0.15, color="green")
        self.ax.scatter([0], [0], [0.021], c="white", edgecolor="black", s=40,
                        label="ball")

    def _draw_mocap_target(self) -> None:
        mp = self._mocap_grip()
        ch = self._mocap_clubhead()
        if mp is None or ch is None:
            return
        # Mocap club: red, thick
        pts = np.array([mp, ch])
        self.ax.plot(pts[:, 0], pts[:, 1], pts[:, 2],
                     color="red", linewidth=4, label="mocap club")
        self.ax.scatter(*mp, color="red", s=80, marker="o", label="mocap mid-hands")
        self.ax.scatter(*ch, color="darkred", s=120, marker="s", label="mocap clubhead")

    def _draw_skeleton_overlay(self) -> None:
        if not self.skeleton.joints:
            return
        names = list(self.skeleton.joints.keys())
        pts = np.array([self.skeleton.joints[n] for n in names])
        # Apply rigid transform.
        moved = self.transform.apply(pts)
        pos = {n: moved[i] for i, n in enumerate(names)}

        # Draw segments.
        for parent, child in self.skeleton.segments:
            if parent in pos and child in pos:
                a, b = pos[parent], pos[child]
                color = "blue" if (parent, child) != ("mp", "ch") else "darkblue"
                width = 4 if (parent, child) == ("mp", "ch") else 2.5
                self.ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]],
                             color=color, linewidth=width)
        # Draw joints.
        self.ax.scatter(moved[:, 0], moved[:, 1], moved[:, 2],
                        color="blue", s=30, label="sim skeleton")
        # Highlight mp + ch
        if "mp" in pos:
            self.ax.scatter(*pos["mp"], color="cyan", s=80, marker="o",
                            edgecolor="navy", label="sim mid-hands")
        if "ch" in pos:
            self.ax.scatter(*pos["ch"], color="cyan", s=120, marker="s",
                            edgecolor="navy", label="sim clubhead")

    def _update_residual(self) -> None:
        if "mp" not in self.skeleton.joints:
            self.lbl_residual.setText("Grip residual: (no skeleton mp)")
            return
        target = self._mocap_grip()
        if target is None:
            self.lbl_residual.setText("Grip residual: (no mocap)")
            return
        moved_mp = self.transform.apply(self.skeleton.joints["mp"][None, :])[0]
        delta = moved_mp - target
        d_mm = float(np.linalg.norm(delta) * 1000.0)
        self.lbl_residual.setText(
            f"Grip residual: {d_mm:.1f} mm  "
            f"(Δ = [{delta[0]*1000:+.0f}, {delta[1]*1000:+.0f}, "
            f"{delta[2]*1000:+.0f}] mm)")


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
