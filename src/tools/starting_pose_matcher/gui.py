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

Run::

    python -m src.tools.starting_pose_matcher

Or, from the GolfLauncher tile (registered in ``src/config/models.yaml``).
"""

from __future__ import annotations

from contextlib import suppress
import json
import logging
import os
import sys
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
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QSplitter,
    QStyle,
    QStyleFactory,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

# Pure-data + math core.  Split out of this file so it can be unit-tested
# in environments where the Qt stack isn't fully working.
# Pure-data + math layer (split out so it can be unit-tested without Qt).
from src.tools.starting_pose_matcher.core import (
    CM_TO_M,
    DEFAULT_EVENT_PRESET as _DEFAULT_EVENT_PRESET,
    DEFAULT_PHASE as _DEFAULT_PHASE,
    EVENT_KEYS as _EVENT_KEYS,
    EVENT_LABEL_PRESETS as _EVENT_LABEL_PRESETS,
    MocapEvents,
    PHASE_BOUNDS as _PHASE_BOUNDS,
    PHASE_KEYS as _PHASE_KEYS,
    PoseSlot,
    RigidTransform,
    SESSION_SCHEMA_VERSION as _SESSION_SCHEMA_VERSION,
    Skeleton,
    SkeletonTrajectory,
    load_mocap_xlsx,
    load_simscape_trajectory_csv,
    load_skeleton,
    phase_display_label as _phase_display_label,
    phase_key_from_label as _phase_key_from_label,
    read_event_header,
    solve_shaft_rz_deg,
)
from src.tools.starting_pose_matcher.skeleton_extractor import (
    JsonSkeletonExtractor,
    SkeletonExtractor,
)
from src.tools.starting_pose_matcher.gui_source_panel import DataSourcesPanel
from src.tools.starting_pose_matcher.session_schema import (
    DataSourcesBlock,
    parse_data_sources,
    serialize_data_sources,
)

# Shared 3D-rendering helpers (per #4376 — DRY with the rest of the
# motion-matching diagnostics).  Used by ``_setup_axes`` to fit the view
# tightly around the current data when a target/skeleton is loaded.
from src.shared.python.motion_matching.diagnostics._skeleton_render import (
    equalize_3d_axes as _shared_equalize_3d_axes,
)
from src.tools.starting_pose_matcher.live_view_controller import LiveViewController

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")

# Default camera presets (elev, azim) for matplotlib's 3D view_init.
_CAMERA_PRESETS: dict[str, tuple[float, float]] = {
    "Face-On": (10.0, -90.0),  # facing the golfer along +Y
    "Down-Line": (10.0, 0.0),  # behind the ball, looking down +X target line
    "Top-Down": (89.0, -90.0),
    "Isometric": (20.0, -55.0),
    "Reset": (15.0, -60.0),
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
QToolButton#help {
    background: #475261;
    border: 1px solid #5d6677;
    border-radius: 10px;
    color: #f0c674;
    font-weight: bold;
    padding: 0;
}
QToolButton#help:hover { background: #5a657a; }
QSplitter::handle:horizontal { background: #404652; width: 4px; }
QSplitter::handle:vertical   { background: #404652; height: 4px; }
QScrollArea { border: 0; }
"""


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


# ----------------------------------------------------------------------------
# Help-button helper                                                          #
# ----------------------------------------------------------------------------
# The help-text registry: keyed by section name, value is a multi-line
# explanation shown in a QMessageBox when the user clicks the "?" button.
_HELP_TEXT: dict[str, str] = {
    "Mocap Source": (
        "Loads a Wiffle-style xlsx motion-capture file.\n\n"
        "• xlsx positions are in CENTIMETRES (we convert to metres on load).\n"
        "• Sheet selects which trial in the workbook to read.\n"
        "• Row 1 of the sheet defines event sample numbers (A/T/I/F + CHS).\n"
        "  These appear here and feed the per-pose event combos."
    ),
    "Event Labels": (
        "Pick the naming convention for the four swing events:\n\n"
        "  Wiffle (A/T/I/F)        — Address, Top of Backswing, Impact, Finish\n"
        "  Trackman P-system       — P1, P4, P7, P10\n"
        "  Plain English           — Setup / Backswing top / Strike / End\n"
        "  Sequence numbers        — Phase 1..4\n\n"
        "Edit any of the four text boxes to write your own labels.  The\n"
        "preset combo flips to 'Custom' automatically when you do.\n"
        "Labels propagate to phase windows, pose-slot combos, and the\n"
        "current-frame readout.  They are saved with the session."
    ),
    "Pose Slots": (
        "The two model poses overlaid on the mocap target.\n\n"
        "• 'Show' toggles the skeleton overlay for that pose.\n"
        "• 'Event' picks which mocap frame the pose-target lines up against.\n"
        "• Reload (⟳) re-reads the corresponding\n"
        "  simscape_skeleton_<Pose>.json (regenerate via\n"
        "  export_default_skeleton.m in MATLAB).\n"
        "• 'Trajectory…' loads a Simscape forward-dynamics CSV so the\n"
        "  skeleton can play back through its motion (instead of being\n"
        "  static).  Use the Playback group's target combo to choose."
    ),
    "Playback": (
        "Animate the mocap (always available), the skeleton (when a\n"
        "trajectory CSV is loaded for a visible pose), or both.\n\n"
        "• Frame slider scrubs through the mocap timeline.\n"
        "• Step buttons jump first / -10 / -1 / +1 / +10 / last.\n"
        "• Play/Pause runs the timer at the chosen FPS; Loop wraps when\n"
        "  the end is reached.\n"
        "• 'Playback target' chooses what plays back:\n"
        "    Mocap     — animate the mocap target (default).\n"
        "    Skeleton  — animate the skeleton through its trajectory.\n"
        "    Both      — animate both, time-aligned by impact.\n"
        "• 'Use current frame for mocap target' overrides the per-pose\n"
        "  events and pins the mocap target to the slider.\n"
        "• 'Mark current frame as event' records an in-session override\n"
        "  without modifying the xlsx."
    ),
    "View / Mocap Traces": (
        "Camera presets jump the 3D view to a known angle (Reset returns\n"
        "to the default).  Use the Matplotlib toolbar above the plot for\n"
        "free pan / zoom / rotate.\n\n"
        "Trace toggles draw the mocap clubhead and / or mid-hands path\n"
        "across the chosen swing phase.  Phase boundaries are marked by\n"
        "green / purple triangles; the current frame is marked with a\n"
        "yellow ✕ when traces are visible.\n\n"
        "Scene toggles:\n"
        "  • Show golf ball — white ball at the world origin.\n"
        "  • Show ground plane — green semi-transparent floor at Z=0.\n"
        "  • Torso-twist indicator — small disc at the model's torso\n"
        "    revolute joint (between spine and hub).  The disc plane is\n"
        "    perpendicular to the spine direction; an arrow points along\n"
        "    the LS-RS shoulder line so you can see the body coil at a\n"
        "    glance.  This matches the rotating-disk geometry in\n"
        "    GolfSwing3D_Kinetic.mdl ('Torso Kinetically Driven' block\n"
        "    between UpperTorsoBase and UpperTorsoTop)."
    ),
    "Auto-Align": (
        "Solve a transform automatically.\n\n"
        "• Snap … (shaft-aligned): finds the Rz that aligns the model\n"
        "  shaft (mid-hands → clubhead) with the mocap shaft at the\n"
        "  pose's event frame, then sets Tx/Ty/Tz so the model mid-hands\n"
        "  lands on the mocap mid-hands.  Rx/Ry are forced to 0 (Z-up).\n"
        "• 'Also fit scale' additionally sets scale =\n"
        "  |shaft_target| / |shaft_model| so the shaft length matches.\n"
        "• Snap mid-hands only: just translates the first visible pose's\n"
        "  mid-hands onto the mocap mid-hands without changing rotation."
    ),
    "Rigid Transform + Scale": (
        "Manual 7-DOF transform.\n\n"
        "Rx/Ry are LOCKED by default because both the mocap data and the\n"
        "Simscape model use Z-up — only the heading (Rz) and translation\n"
        "are physically meaningful for global alignment.  Tick 'Allow\n"
        "Rx/Ry rotations' if you really need them.\n\n"
        "Rotations pivot around the first pose's hub joint so they feel\n"
        "like body rotations.  Scale is isotropic about the same pivot."
    ),
    "Output": (
        "• Save offsets to JSON: writes only the transform + residuals.\n"
        "  This is what fit_swing_full_pipeline reads as input_overrides.\n"
        "• Save / Load session: full UI snapshot — every slider, label,\n"
        "  pose visibility, camera angle, trace toggles, current frame,\n"
        "  fps, etc.  Re-opens to the exact state you saved."
    ),
}


def _help_button(section_title: str, parent: QWidget | None = None) -> QToolButton:
    """Make a small info-icon button that pops up the help text for a section."""
    from src.shared.python.ui.info_button import make_info_button

    def _show() -> None:
        text = _HELP_TEXT.get(section_title, "(no help text registered)")
        QMessageBox.information(parent, f"Help — {section_title}", text)

    btn = make_info_button(
        parent,
        tooltip=f"Help: {section_title}",
        accessible_name=f"Help: {section_title}",
        on_click=_show,
    )
    btn.setObjectName("help")
    return btn


def _group_with_help(title: str, content: QWidget) -> QGroupBox:
    """Wrap a content widget in a QGroupBox with a help button in the title row."""
    box = QGroupBox()
    box.setTitle("")
    outer = QVBoxLayout(box)
    outer.setContentsMargins(8, 8, 8, 8)
    outer.setSpacing(4)
    head = QHBoxLayout()
    lbl = QLabel(title)
    lbl.setObjectName("groupTitle")
    head.addWidget(lbl)
    head.addStretch()
    head.addWidget(_help_button(title, box))
    outer.addLayout(head)
    outer.addWidget(content)
    return box


class LabelledControl(QWidget):
    """Spinbox + slider (slider follows spinbox).  Public API:
    .value()          -> float
    .set_value(v)
    .setEnabled(bool) -> grays out the whole row
    .valueChanged signal-like callback via spin.valueChanged
    """

    def __init__(
        self,
        label: str,
        units: str,
        slider_range: tuple[int, int],
        scale: float,
        decimals: int,
        default: float = 0.0,
        parent: QWidget | None = None,
    ):
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

        self.slider.valueChanged.connect(lambda v: self.spin.setValue(v * self._scale))
        self.spin.valueChanged.connect(
            lambda v: self.slider.setValue(int(round(v / self._scale)))
        )
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
        # Default extractor: JSON-based Simscape skeleton loader
        self.skeleton_extractor: SkeletonExtractor = JsonSkeletonExtractor(
            here, poses=("TopofBackswing", "Impact")
        )
        self.poses: dict[str, PoseSlot] = {
            "TopofBackswing": PoseSlot(
                name="TopofBackswing",
                skeleton=self.skeleton_extractor.get_skeleton("TopofBackswing"),
                color="#5b9eff",
                mocap_color="#ef4444",
                target_event="T",
            ),
            "Impact": PoseSlot(
                name="Impact",
                skeleton=self.skeleton_extractor.get_skeleton("Impact"),
                color="#10b981",
                mocap_color="#f59e0b",
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
        self.show_ball = True
        self.show_ground = True
        self.show_torso_disk = True  # disc indicator at torso joint
        self.lock_xy_rotation = True  # Rx/Ry locked by default
        self.auto_fit_axes = True  # use shared equalize_3d_axes per redraw

        # Playback state
        self.current_frame: int = 0
        self.frame_override_active: bool = False  # use slider frame for mocap target?
        self.is_playing: bool = False
        self.loop_playback: bool = True
        self.event_overrides: dict[str, int] = {}  # user-set A/T/I/F sample numbers
        # Playback speed multiplier and marker-trail length for the animated
        # full-trajectory preview (issue #4482).
        self.playback_speed: float = 1.0
        self.trail_frames: int = 30
        self.show_trail: bool = True

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._advance_frame)

        # Phase window state keeps the user/session display label for legacy
        # compatibility; drawing paths normalize it back to a logical key.
        self.phase_window: str = _DEFAULT_PHASE
        self.manual_window_start: int = 0
        self.manual_window_end: int = 0

        # Playback target — what advances when the timer fires:
        #   "Mocap"     animate the mocap target only (skeleton stays static)
        #   "Skeleton"  animate the skeleton through its trajectory CSV
        #   "Both"      animate both, time-aligned at the impact frame
        self.playback_target: str = "Mocap"

        # Event labels (Address / Top of Backswing / Impact / Finish, or
        # author-specific conventions).  Mutated via the Event-Labels
        # group; persisted to session JSON.
        self.event_label_preset: str = _DEFAULT_EVENT_PRESET
        self.event_labels: dict[str, str] = dict(
            _EVENT_LABEL_PRESETS[_DEFAULT_EVENT_PRESET]
        )

        self._build_ui()
        self._apply_camera_preset(_DEFAULT_CAMERA)

        default_xlsx = Path(__file__).with_name("Wiffle_ProV1_club_3D_data.xlsx")
        if default_xlsx.exists():
            self._load_xlsx(str(default_xlsx))

    # ===================================================================== #
    # UI                                                                    #
    # ===================================================================== #

    def _build_ui(self) -> None:
        """Build the main window with QSplitters so the user can resize the
        control panel vs. the plot AND each section independently."""
        central = QWidget()
        self.setCentralWidget(central)
        outer = QHBoxLayout(central)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.setSpacing(0)

        # Outer horizontal splitter: control panel | plot
        self.h_splitter = QSplitter(Qt.Orientation.Horizontal)
        outer.addWidget(self.h_splitter)

        # ---------- LEFT: scrollable column with vertical splitter ------- #
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(360)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        left_widget = QWidget()
        scroll.setWidget(left_widget)
        left_col = QVBoxLayout(left_widget)
        left_col.setContentsMargins(4, 4, 4, 4)
        left_col.setSpacing(4)

        title = QLabel("Starting-Pose Matcher")
        title.setObjectName("title")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        left_col.addWidget(title)

        # Vertical splitter: every group can be resized.  Sections in order.
        self.v_splitter = QSplitter(Qt.Orientation.Vertical)
        self.v_splitter.setChildrenCollapsible(True)
        # Multi-source toggle panel (issue #4480).  Lives alongside the
        # legacy Mocap-Source group; either may be used to drive the view.
        self.source_panel = DataSourcesPanel()
        self.source_panel.targets_changed.connect(self._on_multi_source_changed)
        self._latest_multi_source: object | None = None

        self._sections: dict[str, QGroupBox] = {
            "Mocap Source": self._build_file_box(),
            "Data sources": self.source_panel,
            "Event Labels": self._build_event_labels_box(),
            "Pose Slots": self._build_pose_box(),
            "Playback": self._build_playback_box(),
            "View / Mocap Traces": self._build_view_box(),
            "Auto-Align": self._build_align_box(),
            "Rigid Transform + Scale": self._build_transform_box(),
            "Output": self._build_save_box(),
        }
        for name, box in self._sections.items():
            self._attach_help_button(box, name)
            self.v_splitter.addWidget(box)
        # Reasonable starting heights (px) so big sections aren't squished.
        self.v_splitter.setSizes([90, 200, 160, 180, 220, 200, 180, 280, 140])
        left_col.addWidget(self.v_splitter, stretch=1)

        self.h_splitter.addWidget(scroll)

        # ---------- RIGHT: plot column ----------------------------------- #
        plot_widget = QWidget()
        plot_layout = QVBoxLayout(plot_widget)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        plot_layout.setSpacing(0)

        self.fig = Figure(figsize=(10, 8), dpi=100, facecolor="#1f242b")
        self.ax = self.fig.add_subplot(111, projection="3d", facecolor="#1f242b")
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.toolbar = NavigationToolbar(self.canvas, plot_widget)
        self.toolbar.setStyleSheet("background:#2b2f36;color:#e6e6e6;")
        plot_layout.addWidget(self.toolbar)
        plot_layout.addWidget(self.canvas)

        self.h_splitter.addWidget(plot_widget)
        self.h_splitter.setSizes([440, 1200])
        self.h_splitter.setStretchFactor(0, 0)
        self.h_splitter.setStretchFactor(1, 1)

        self._setup_axes()

        # Live multi-source view controller (issue #4512). Owns the layer
        # stack that renders BodyTarget / ClubTarget / BallImpact data on
        # the same axes the static-pose path uses.
        self._live_view = LiveViewController(self.ax, self.canvas)
        self._live_body_target: Any | None = None

    def _attach_help_button(self, box: QGroupBox, section: str) -> None:
        """Place a small '?' help button at the top-right corner of a QGroupBox.

        Repositions itself on resize so it always tracks the corner.
        """
        btn = _help_button(section, box)
        btn.setParent(box)
        btn.show()
        btn.raise_()

        def _reposition() -> None:
            btn.move(max(0, box.width() - 30), 4)

        _reposition()
        # Chain into the existing resizeEvent without losing it
        original = box.resizeEvent

        def _on_resize(event):  # noqa: ANN001
            _reposition()
            original(event)

        box.resizeEvent = _on_resize  # type: ignore[method-assign]

    # ---------- builders --------------------------------------------------- #

    def _build_file_box(self) -> QGroupBox:
        box = QGroupBox("Mocap Source")
        gl = QGridLayout(box)
        gl.setVerticalSpacing(6)
        self.btn_load = QPushButton("Load xlsx…")
        self.btn_load.setToolTip("Load a motion-capture xlsx target file")
        self.btn_load.setStatusTip("Loads motion-capture xlsx")
        self.btn_load.clicked.connect(self._on_load_clicked)
        gl.addWidget(self.btn_load, 0, 0, 1, 2)
        gl.addWidget(QLabel("Sheet:"), 1, 0)
        self.sheet_combo = QComboBox()
        self.sheet_combo.addItems(["TW_ProV1", "TW_wiffle", "GW_wiffle", "GW_ProV11"])
        self.sheet_combo.setToolTip("Select which sheet of the xlsx to load")
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
        # Live C3D body source row (issue #4512). Loads a BodyTarget and
        # routes per-frame marker positions to the matcher's existing 3D
        # axes via the LiveViewController.
        self.btn_load_c3d_body = QPushButton("Browse C3D Body…")
        self.btn_load_c3d_body.setToolTip(
            "Load a .c3d body-marker file and render its markers live on the 3D axes."
        )
        self.btn_load_c3d_body.setStatusTip(
            "Loads a C3D body target and wires it to the timeline slider."
        )
        self.btn_load_c3d_body.clicked.connect(self._on_load_c3d_body_clicked)
        gl.addWidget(self.btn_load_c3d_body, 4, 0, 1, 2)
        self.lbl_c3d_body = QLabel("Live body: (none)")
        self.lbl_c3d_body.setObjectName("status")
        self.lbl_c3d_body.setWordWrap(True)
        gl.addWidget(self.lbl_c3d_body, 5, 0, 1, 2)
        # Layer toggles for the live view.
        self.cb_show_body_markers = QCheckBox("Show body markers")
        self.cb_show_body_markers.setChecked(True)
        self.cb_show_body_markers.toggled.connect(
            lambda on: self._live_view.set_layer_visible("body_markers", bool(on))
            if getattr(self, "_live_view", None) is not None
            else None
        )
        gl.addWidget(self.cb_show_body_markers, 6, 0, 1, 1)
        self.cb_show_body_skeleton = QCheckBox("Show body skeleton")
        self.cb_show_body_skeleton.setChecked(True)
        self.cb_show_body_skeleton.toggled.connect(
            lambda on: self._live_view.set_layer_visible("body_skeleton", bool(on))
            if getattr(self, "_live_view", None) is not None
            else None
        )
        gl.addWidget(self.cb_show_body_skeleton, 6, 1, 1, 1)
        return box

    def _build_event_labels_box(self) -> QGroupBox:
        box = QGroupBox("Event Labels")
        gl = QGridLayout(box)
        gl.setVerticalSpacing(4)

        gl.addWidget(QLabel("Convention:"), 0, 0)
        self.event_preset_combo = QComboBox()
        for preset in _EVENT_LABEL_PRESETS:
            self.event_preset_combo.addItem(preset)
        self.event_preset_combo.addItem("Custom…")
        self.event_preset_combo.setCurrentText(self.event_label_preset)
        self.event_preset_combo.setToolTip(
            "Event-label naming convention used in the legend and pose-event combos"
        )
        self.event_preset_combo.currentTextChanged.connect(
            self._on_event_preset_changed
        )
        gl.addWidget(self.event_preset_combo, 0, 1, 1, 3)

        # Editable entries for each event key
        from PyQt6.QtWidgets import QLineEdit

        self._event_label_edits: dict[str, QLineEdit] = {}
        for r, k in enumerate(_EVENT_KEYS, start=1):
            gl.addWidget(QLabel(f"{k}:"), r, 0)
            le = QLineEdit(self.event_labels[k])
            le.setMinimumWidth(160)
            le.editingFinished.connect(lambda key=k: self._on_event_label_edited(key))
            self._event_label_edits[k] = le
            gl.addWidget(le, r, 1, 1, 3)

        hint = QLabel(
            "Custom labels are saved with the session and shown in the legend / "
            "current-frame indicator."
        )
        hint.setObjectName("status")
        hint.setWordWrap(True)
        gl.addWidget(hint, len(_EVENT_KEYS) + 1, 0, 1, 4)
        return box

    def _on_event_preset_changed(self, preset: str) -> None:
        if preset in _EVENT_LABEL_PRESETS:
            self.event_label_preset = preset
            self.event_labels = dict(_EVENT_LABEL_PRESETS[preset])
            for k, le in self._event_label_edits.items():
                with QSignalBlocker(le):
                    le.setText(self.event_labels[k])
        else:
            self.event_label_preset = "Custom"
        self._refresh_event_label_dependents()

    def _on_event_label_edited(self, key: str) -> None:
        text = self._event_label_edits[key].text().strip() or key
        self.event_labels[key] = text
        # Switch preset to Custom unless the new map matches a preset exactly.
        for preset, mapping in _EVENT_LABEL_PRESETS.items():
            if mapping == self.event_labels:
                self.event_label_preset = preset
                with QSignalBlocker(self.event_preset_combo):
                    self.event_preset_combo.setCurrentText(preset)
                break
        else:
            self.event_label_preset = "Custom"
            with QSignalBlocker(self.event_preset_combo):
                self.event_preset_combo.setCurrentText("Custom…")
        self._refresh_event_label_dependents()

    def _refresh_event_label_dependents(self) -> None:
        """Re-render anything that displays event labels."""
        # Pose-slot 'Event' combos: we keep the underlying key (A/T/I/F)
        # but show the display label.  Done via combo items.
        for combo in getattr(self, "_pose_event_combos", {}).values():
            current = combo.currentText().split()[0]  # original key
            with QSignalBlocker(combo):
                combo.clear()
                for k in _EVENT_KEYS:
                    combo.addItem(f"{k} - {self.event_labels[k]}")
                # restore selection
                idx = next(
                    (
                        i
                        for i in range(combo.count())
                        if combo.itemText(i).startswith(current + " ")
                    ),
                    0,
                )
                combo.setCurrentIndex(idx)
        # "Mark current frame as event" combo
        if hasattr(self, "combo_set_event"):
            current = self.combo_set_event.currentText().split(" ", 1)[0] or "T"
            with QSignalBlocker(self.combo_set_event):
                self.combo_set_event.clear()
                for k in _EVENT_KEYS:
                    self.combo_set_event.addItem(f"{k} - {self.event_labels[k]}")
                for i in range(self.combo_set_event.count()):
                    if self.combo_set_event.itemText(i).startswith(current + " "):
                        self.combo_set_event.setCurrentIndex(i)
                        break
        # Phase combo — re-render display labels (preserve selected key).
        if hasattr(self, "phase_combo"):
            current_key = self.phase_combo.currentData() or self.phase_window
            with QSignalBlocker(self.phase_combo):
                self.phase_combo.clear()
                for k in _PHASE_KEYS:
                    self.phase_combo.addItem(
                        _phase_display_label(k, self.event_labels), k
                    )
                for i in range(self.phase_combo.count()):
                    if self.phase_combo.itemData(i) == current_key:
                        self.phase_combo.setCurrentIndex(i)
                        break
        # Refresh events summary line
        self.lbl_event_info.setText(self._events_summary())
        self._redraw()

    def _build_pose_box(self) -> QGroupBox:
        box = QGroupBox("Pose Slots")
        gl = QGridLayout(box)
        gl.setVerticalSpacing(4)
        gl.addWidget(QLabel("Show"), 0, 0)
        gl.addWidget(QLabel("Pose"), 0, 1)
        gl.addWidget(QLabel("Event"), 0, 2)
        gl.addWidget(QLabel("Reload"), 0, 3)
        gl.addWidget(QLabel("Trajectory"), 0, 4)
        self._pose_visible_checks: dict[str, QCheckBox] = {}
        self._pose_event_combos: dict[str, QComboBox] = {}
        self._pose_trajectory_buttons: dict[str, QPushButton] = {}
        for r, (key, slot) in enumerate(self.poses.items(), start=1):
            cb = QCheckBox()
            cb.setChecked(slot.visible)
            cb.setToolTip(
                f"Show or hide the {key} skeleton overlay (color {slot.color})"
            )
            cb.stateChanged.connect(self._on_pose_toggled)
            self._pose_visible_checks[key] = cb
            gl.addWidget(cb, r, 0)
            color = slot.color
            tag = QLabel(f'<span style="color:{color};">●</span>  {key}')
            gl.addWidget(tag, r, 1)
            ec = QComboBox()
            ec.setToolTip(f"Mocap event the {key} pose snaps to when Auto-Align is run")
            for k in _EVENT_KEYS:
                ec.addItem(f"{k} - {self.event_labels[k]}")
            # Pick the item whose first token matches the slot's key
            for i in range(ec.count()):
                if ec.itemText(i).startswith(slot.target_event + " "):
                    ec.setCurrentIndex(i)
                    break
            ec.currentTextChanged.connect(self._on_pose_event_changed)
            self._pose_event_combos[key] = ec
            gl.addWidget(ec, r, 2)
            rbtn = QPushButton("⟳")
            rbtn.setObjectName("preset")
            rbtn.setMaximumWidth(40)
            rbtn.setToolTip(f"Reload simscape_skeleton_{key}.json")
            rbtn.clicked.connect(lambda _checked, k=key: self._reload_pose(k))
            gl.addWidget(rbtn, r, 3)
            tbtn = QPushButton("Load…")
            tbtn.setObjectName("preset")
            tbtn.setMaximumWidth(80)
            tbtn.setToolTip(
                "Load a Simscape forward-dynamics CSV so the\n"
                "skeleton can play back through its motion."
            )
            tbtn.clicked.connect(lambda _checked, k=key: self._load_trajectory(k))
            self._pose_trajectory_buttons[key] = tbtn
            gl.addWidget(tbtn, r, 4)
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
        self.cb_clubhead_trace.setToolTip(
            "Overlay the clubhead path from the mocap file"
        )
        self.cb_clubhead_trace.stateChanged.connect(self._on_traces_toggled)
        v.addWidget(self.cb_clubhead_trace)
        self.cb_midhands_trace = QCheckBox("Show mocap mid-hands path")
        self.cb_midhands_trace.setToolTip(
            "Overlay the mid-hands path from the mocap file"
        )
        self.cb_midhands_trace.stateChanged.connect(self._on_traces_toggled)
        v.addWidget(self.cb_midhands_trace)

        # Phase window combo (replaces the old simple "swing window" checkbox)
        ph_row = QHBoxLayout()
        ph_row.addWidget(QLabel("Phase:"))
        self.phase_combo = QComboBox()
        self.phase_combo.setToolTip(
            "Phase window for trace overlays; choose Manual range to set frames"
        )
        for key in _PHASE_KEYS:
            self.phase_combo.addItem(_phase_display_label(key, self.event_labels), key)
        # Select the default by KEY (currentData() lookup)
        for i in range(self.phase_combo.count()):
            if self.phase_combo.itemData(i) == _DEFAULT_PHASE:
                self.phase_combo.setCurrentIndex(i)
                break
        self.phase_combo.currentIndexChanged.connect(self._on_phase_changed)
        ph_row.addWidget(self.phase_combo, stretch=1)
        v.addLayout(ph_row)

        # Manual range (hidden until "Manual range" selected)
        self.manual_range_widget = QWidget()
        mr = QHBoxLayout(self.manual_range_widget)
        mr.setContentsMargins(0, 0, 0, 0)
        mr.addWidget(QLabel("From:"))
        self.spin_phase_start = QSpinBox()
        self.spin_phase_start.setRange(0, 0)
        self.spin_phase_start.setToolTip("First frame index of the manual phase window")
        self.spin_phase_start.valueChanged.connect(self._on_manual_range_changed)
        mr.addWidget(self.spin_phase_start)
        mr.addWidget(QLabel("To:"))
        self.spin_phase_end = QSpinBox()
        self.spin_phase_end.setRange(0, 0)
        self.spin_phase_end.setToolTip("Last frame index of the manual phase window")
        self.spin_phase_end.valueChanged.connect(self._on_manual_range_changed)
        mr.addWidget(self.spin_phase_end)
        self.manual_range_widget.setVisible(False)
        v.addWidget(self.manual_range_widget)

        # Show current-frame marker on traces
        self.cb_frame_marker = QCheckBox("Show current-frame marker on traces")
        self.cb_frame_marker.setChecked(True)
        self.cb_frame_marker.setToolTip(
            "Render a marker at the current playback frame on each trace"
        )
        self.cb_frame_marker.stateChanged.connect(lambda _: self._redraw())
        v.addWidget(self.cb_frame_marker)

        v.addWidget(_hsep())

        # Scene element toggles
        self.cb_show_ball = QCheckBox("Show golf ball")
        self.cb_show_ball.setChecked(self.show_ball)
        self.cb_show_ball.setToolTip("Render the ball glyph at the address position")
        self.cb_show_ball.stateChanged.connect(self._on_scene_toggled)
        v.addWidget(self.cb_show_ball)

        self.cb_show_ground = QCheckBox("Show ground plane")
        self.cb_show_ground.setChecked(self.show_ground)
        self.cb_show_ground.setToolTip("Render a translucent ground plane at z=0")
        self.cb_show_ground.stateChanged.connect(self._on_scene_toggled)
        v.addWidget(self.cb_show_ground)

        self.cb_show_torso_disk = QCheckBox(
            "Show torso-twist indicator (disk at torso joint)"
        )
        self.cb_show_torso_disk.setChecked(self.show_torso_disk)
        self.cb_show_torso_disk.setToolTip(
            "Draws a small disc at the torso revolute joint between the\n"
            "spine and the hub.  The disc orientation reflects the body\n"
            "twist (LS-RS line direction) so the rotating-disk action of\n"
            "the model is visually obvious."
        )
        self.cb_show_torso_disk.stateChanged.connect(self._on_scene_toggled)
        v.addWidget(self.cb_show_torso_disk)

        self.cb_auto_fit_axes = QCheckBox("Auto-fit axes to data")
        self.cb_auto_fit_axes.setChecked(self.auto_fit_axes)
        self.cb_auto_fit_axes.setToolTip(
            "Re-fit the 3D axis bounds to whatever skeleton + mocap target\n"
            "is currently visible.  Uses the shared\n"
            "src/shared/python/motion_matching/diagnostics/\n"
            "_skeleton_render.equalize_3d_axes helper so the view always\n"
            "stays cropped tightly around the body.  Untick to keep fixed\n"
            "[-2, 2] x [-1.5, 2] x [-1.5, 2.5] m bounds (useful for\n"
            "comparing scale across loads)."
        )
        self.cb_auto_fit_axes.stateChanged.connect(self._on_scene_toggled)
        v.addWidget(self.cb_auto_fit_axes)
        return box

    def _on_scene_toggled(self, _: int) -> None:
        self.show_ball = self.cb_show_ball.isChecked()
        self.show_ground = self.cb_show_ground.isChecked()
        self.show_torso_disk = self.cb_show_torso_disk.isChecked()
        self.auto_fit_axes = self.cb_auto_fit_axes.isChecked()
        self._redraw()

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
        self.spin_frame.setToolTip("Current playback frame index")
        self.spin_frame.valueChanged.connect(self._on_frame_changed_spin)
        row1.addWidget(self.spin_frame)
        self.lbl_time = QLabel("t = — s")
        self.lbl_time.setObjectName("status")
        self.lbl_time.setMinimumWidth(110)
        row1.addWidget(self.lbl_time)
        v.addLayout(row1)

        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setRange(0, 0)
        self.frame_slider.setToolTip("Scrub through the loaded mocap or trajectory")
        self.frame_slider.valueChanged.connect(self._on_frame_changed_slider)
        v.addWidget(self.frame_slider)

        # Step buttons
        step_row = QHBoxLayout()
        step_row.setSpacing(2)
        for label, delta, tip in [
            ("⏮", -(10**9), "First frame"),
            ("⏪", -10, "−10 frames"),
            ("◀", -1, "−1 frame"),
            ("▶", +1, "+1 frame"),
            ("⏩", +10, "+10 frames"),
            ("⏭", +(10**9), "Last frame"),
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
        self.btn_play.setToolTip("Start or stop animated playback (Space)")
        self.btn_play.setStatusTip("Toggles playback")
        self.btn_play.clicked.connect(self._toggle_play)
        play_row.addWidget(self.btn_play)
        play_row.addWidget(QLabel("Speed:"))
        self.spin_speed = QSpinBox()
        self.spin_speed.setRange(1, 240)
        self.spin_speed.setValue(30)
        self.spin_speed.setSuffix(" fps")
        self.spin_speed.setToolTip("Playback speed in frames per second (1-240)")
        play_row.addWidget(self.spin_speed)
        self.cb_loop = QCheckBox("Loop")
        self.cb_loop.setChecked(True)
        self.cb_loop.setToolTip("Restart playback from the first frame on overflow")
        self.cb_loop.stateChanged.connect(
            lambda _: setattr(self, "loop_playback", self.cb_loop.isChecked())
        )
        play_row.addWidget(self.cb_loop)
        v.addLayout(play_row)

        # Speed multiplier combo + frame counter (issue #4482).
        scale_row = QHBoxLayout()
        scale_row.addWidget(QLabel("× Speed:"))
        self.combo_speed = QComboBox()
        from .session_schema import ALLOWED_SPEEDS as _ALLOWED_SPEEDS

        for s in _ALLOWED_SPEEDS:
            self.combo_speed.addItem(f"{s}×", float(s))
        self.combo_speed.setCurrentText("1.0×")
        self.combo_speed.currentIndexChanged.connect(
            lambda _i: setattr(
                self,
                "playback_speed",
                float(self.combo_speed.currentData() or 1.0),
            )
        )
        scale_row.addWidget(self.combo_speed)
        scale_row.addStretch(1)
        self.lbl_frame_counter = QLabel("0 / 0")
        self.lbl_frame_counter.setObjectName("status")
        scale_row.addWidget(self.lbl_frame_counter)
        v.addLayout(scale_row)

        # Show-trail toggle (default on, fading polylines for last N frames).
        trail_row = QHBoxLayout()
        self.cb_show_trail = QCheckBox("Show trail")
        self.cb_show_trail.setChecked(True)
        self.cb_show_trail.stateChanged.connect(
            lambda _: setattr(self, "show_trail", self.cb_show_trail.isChecked())
        )
        trail_row.addWidget(self.cb_show_trail)
        trail_row.addWidget(QLabel("frames:"))
        self.spin_trail = QSpinBox()
        self.spin_trail.setRange(0, 600)
        self.spin_trail.setValue(int(self.trail_frames))
        self.spin_trail.valueChanged.connect(
            lambda v: setattr(self, "trail_frames", int(v))
        )
        trail_row.addWidget(self.spin_trail)
        trail_row.addStretch(1)
        v.addLayout(trail_row)

        # Playback target selector — what advances when Play is pressed.
        target_row = QHBoxLayout()
        target_row.addWidget(QLabel("Playback target:"))
        self.combo_playback_target = QComboBox()
        self.combo_playback_target.addItems(["Mocap", "Skeleton", "Both"])
        self.combo_playback_target.setCurrentText(self.playback_target)
        self.combo_playback_target.currentTextChanged.connect(
            self._on_playback_target_changed
        )
        self.combo_playback_target.setToolTip(
            "Mocap: animate the mocap target.\n"
            "Skeleton: animate the model skeleton through its loaded\n"
            "  trajectory CSV (Pose Slot → Trajectory…).\n"
            "Both: animate both, time-aligned at impact."
        )
        target_row.addWidget(self.combo_playback_target, stretch=1)
        v.addLayout(target_row)

        # Use-current-frame override
        self.cb_use_current_frame = QCheckBox(
            "Use current frame for mocap target (override pose-slot events)"
        )
        self.cb_use_current_frame.setToolTip(
            "Use the slider's current frame as the mocap target rather than event keys"
        )
        self.cb_use_current_frame.stateChanged.connect(self._on_frame_override_toggled)
        v.addWidget(self.cb_use_current_frame)

        # "Set as event" row
        ev_row = QHBoxLayout()
        ev_row.addWidget(QLabel("Mark current frame as event:"))
        self.combo_set_event = QComboBox()
        self.combo_set_event.setToolTip(
            "Event key to assign to the current frame when Set is pressed"
        )
        for k in _EVENT_KEYS:
            self.combo_set_event.addItem(f"{k} - {self.event_labels[k]}")
        ev_row.addWidget(self.combo_set_event)
        b_set = QPushButton("Set")
        b_set.setObjectName("preset")
        b_set.setToolTip("Mark the current frame as the selected event")
        b_set.clicked.connect(self._set_event_to_current_frame)
        ev_row.addWidget(b_set)
        b_clear = QPushButton("Clear overrides")
        b_clear.setObjectName("preset")
        b_clear.setToolTip("Drop all user-assigned event-frame overrides")
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
            "lines up with the mocap shaft at the chosen frame."
        )
        hint.setObjectName("status")
        hint.setWordWrap(True)
        v.addWidget(hint)

        self.cb_fit_scale = QCheckBox("Also fit scale (|shaft_target| / |shaft_model|)")
        self.cb_fit_scale.setToolTip(
            "Also solve a uniform scale so the model shaft length matches the target"
        )
        v.addWidget(self.cb_fit_scale)

        # One snap button per pose-slot
        for key, slot in self.poses.items():
            btn = QPushButton(
                f"Snap {key} pose → mocap @ {slot.target_event} (shaft-aligned)"
            )
            btn.setObjectName("primary")
            btn.clicked.connect(lambda _checked, k=key: self._snap_shaft(k))
            v.addWidget(btn)

        v.addWidget(_hsep())
        # Convenience: snap mid-hands only (legacy quick-snap)
        self.btn_snap_mid = QPushButton("Snap mid-hands only (no rotation)")
        self.btn_snap_mid.setToolTip(
            "Set Tx/Ty/Tz so the FIRST visible skeleton's "
            "mid-hands lands on its mocap target.  "
            "Rotations preserved."
        )
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
        for label, deg in [
            ("-90°", -90),
            ("-45°", -45),
            ("0°", 0),
            ("+45°", 45),
            ("+90°", 90),
            ("180°", 180),
        ]:
            b = QPushButton(label)
            b.setObjectName("preset")
            b.clicked.connect(lambda _checked, d=deg: self.s_rz.set_value(d))
            rz_row.addWidget(b)
        v.addLayout(rz_row)

        v.addWidget(_hsep())

        # X/Y rotation lock
        self.cb_lock_xy = QCheckBox(
            "Allow Rx/Ry rotations (off by default — Z is up in both data and model)"
        )
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
        self.s_scale = LabelledControl("Scale", "×", _S_RANGE, _S_SCALE, 2, default=1.0)
        v.addWidget(self.s_scale)
        sc_row = QHBoxLayout()
        sc_row.setSpacing(4)
        sc_row.addWidget(QLabel("Presets:"))
        for label, val in [
            ("0.85", 0.85),
            ("0.95", 0.95),
            ("1.00", 1.00),
            ("1.05", 1.05),
            ("1.15", 1.15),
        ]:
            b = QPushButton(label)
            b.setObjectName("preset")
            b.clicked.connect(lambda _checked, x=val: self.s_scale.set_value(x))
            sc_row.addWidget(b)
        v.addLayout(sc_row)

        # Pivot info
        pi = QLabel(
            "Pivot @ first-pose hub: ({:.3f}, {:.3f}, {:.3f}) m".format(
                *self.transform.pivot
            )
        )
        pi.setObjectName("status")
        v.addWidget(pi)

        # Wire all the changes
        for s in (
            self.s_tx,
            self.s_ty,
            self.s_tz,
            self.s_rx,
            self.s_ry,
            self.s_rz,
            self.s_scale,
        ):
            s.spin.valueChanged.connect(self._on_transform_changed)

        # Reset row
        reset_row = QHBoxLayout()
        self.btn_reset_t = QPushButton("Reset translations")
        self.btn_reset_t.setToolTip("Set Tx, Ty, Tz back to zero")
        self.btn_reset_t.clicked.connect(self._reset_translations)
        reset_row.addWidget(self.btn_reset_t)
        self.btn_reset_r = QPushButton("Reset rotations")
        self.btn_reset_r.setToolTip("Set Rx, Ry, Rz back to zero")
        self.btn_reset_r.clicked.connect(self._reset_rotations)
        reset_row.addWidget(self.btn_reset_r)
        self.btn_reset_all = QPushButton("Reset all")
        self.btn_reset_all.setToolTip("Reset all translations, rotations, and scale")
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
        self.btn_save.setToolTip("Write the current rigid transform offsets to JSON")
        self.btn_save.setStatusTip("Saves offsets to JSON")
        self.btn_save.clicked.connect(self._on_save_clicked)
        v.addWidget(self.btn_save)

        ses_row = QHBoxLayout()
        self.btn_save_session = QPushButton("Save session…")
        self.btn_save_session.setToolTip(
            "Save the full session (transform, events, view state) to JSON"
        )
        self.btn_save_session.clicked.connect(self._on_save_session_clicked)
        ses_row.addWidget(self.btn_save_session)
        self.btn_load_session = QPushButton("Load session…")
        self.btn_load_session.setToolTip("Load a previously-saved session JSON")
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

        # Default static bounds; if we have data, we'll re-fit via the shared
        # ``equalize_3d_axes`` helper (see #4376) so the view always tracks
        # the loaded mocap + skeleton extents.
        ax.set_xlim(-2.0, 2.0)
        ax.set_ylim(-1.5, 2.0)
        ax.set_zlim(-1.5, 2.5)

        with suppress(AttributeError):
            ax.set_box_aspect((4, 3.5, 4))
        # Dark-theme tick & pane colours
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.set_pane_color((0.16, 0.18, 0.22, 0.85))
            axis.label.set_color("#cbd5e1")
            for t in axis.get_ticklabels():
                t.set_color("#a3a8b3")
            axis._axinfo["grid"]["color"] = (0.35, 0.40, 0.48, 0.45)

    def _autoscale_axes_to_data(self) -> None:
        """Use the shared ``equalize_3d_axes`` helper to fit the view to
        whatever joints / mocap targets are currently visible.  Called from
        ``_redraw`` after the static defaults are set, so the user gets a
        cube-aspect view tightly cropped around the body.
        """
        pts: list[np.ndarray] = []
        # Visible pose skeletons (incl. trajectory frame if active)
        for slot in self.poses.values():
            if not slot.visible:
                continue
            skel = self._effective_skeleton(slot)
            for v in skel.joints.values():
                pts.append(self.transform.apply(v[None, :])[0])
            mp = self._mocap_pos_for(slot, "mid")
            ch = self._mocap_pos_for(slot, "club")
            if mp is not None:
                pts.append(mp)
            if ch is not None:
                pts.append(ch)
        if not pts:
            return
        with suppress(ValueError, AttributeError):
            _shared_equalize_3d_axes(self.ax, np.asarray(pts))

    # ===================================================================== #
    # Event handlers                                                        #
    # ===================================================================== #

    def _apply_camera_preset(self, name: str) -> None:
        elev, azim = _CAMERA_PRESETS.get(name, _CAMERA_PRESETS[_DEFAULT_CAMERA])
        self.ax.view_init(elev=elev, azim=azim)
        self.canvas.draw()

    def _on_load_clicked(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Wiffle xlsx",
            str(Path(__file__).parent),
            "Excel files (*.xlsx *.xls)",
        )
        if path:
            self._load_xlsx(path)

    def _on_load_c3d_body_clicked(self) -> None:
        """Browse for a ``.c3d`` file and route it to the live view.

        Issue #4512: this is the user-facing entry point for the live
        body marker rendering — picking a file here causes 27 markers to
        appear on the existing 3D axes and start scrubbing with the
        timeline slider.
        """
        default_dir = str(Path(__file__).resolve().parents[3] / "data")
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open C3D body file",
            default_dir,
            "C3D files (*.c3d)",
        )
        if not path:
            return
        try:
            from src.shared.python.motion_matching.load_body_target import (
                load_body_target,
            )

            body = load_body_target(path)
        except Exception as exc:  # noqa: BLE001 - surface loader errors
            logger.exception("failed to load C3D body file %s", path)
            QMessageBox.warning(self, "Load failed", f"Could not load C3D body:\n{exc}")
            return

        self._live_body_target = body
        n = int(body.marker_xyz.shape[0])
        m = int(body.marker_xyz.shape[1])
        self.lbl_c3d_body.setText(
            f"Live body: {Path(path).name}  ({m} markers, {n} samples)"
        )
        # Drive the existing slider/spin to the impact frame so the user
        # sees a recognisable pose immediately.
        self._live_view.set_target(body=body, club=None, ball=None)
        with QSignalBlocker(self.frame_slider), QSignalBlocker(self.spin_frame):
            self.frame_slider.setRange(0, max(0, n - 1))
            self.spin_frame.setRange(0, max(0, n - 1))
            self.frame_slider.setValue(0)
            self.spin_frame.setValue(0)
        self._live_view.set_frame(0)
        self.canvas.draw_idle()

    def _on_sheet_changed(self, _: str) -> None:
        if self._xlsx_path:
            self._load_xlsx(self._xlsx_path)

    def _on_pose_toggled(self, _state: int) -> None:
        for key, cb in self._pose_visible_checks.items():
            self.poses[key].visible = cb.isChecked()
        self._redraw()

    def _on_pose_event_changed(self, _: str) -> None:
        for key, combo in self._pose_event_combos.items():
            txt = combo.currentText().strip()
            # Combo items are now "K - Label"; extract the key.
            self.poses[key].target_event = txt.split(" ", 1)[0] if txt else "T"
        self._redraw()

    def _on_traces_toggled(self, _: int) -> None:
        self.show_clubhead_trace = self.cb_clubhead_trace.isChecked()
        self.show_midhands_trace = self.cb_midhands_trace.isChecked()
        self._redraw()

    def _on_playback_target_changed(self, target: str) -> None:
        if target not in ("Mocap", "Skeleton", "Both"):
            target = "Mocap"
        self.playback_target = target
        self._redraw()

    def _load_trajectory(self, slot_key: str) -> None:
        """Load a Simscape CSV trajectory for the given pose slot."""
        slot = self.poses.get(slot_key)
        if slot is None:
            return
        path, _ = QFileDialog.getOpenFileName(
            self,
            f"Load Simscape trajectory CSV for {slot_key}",
            str(Path(__file__).parent),
            "CSV files (*.csv);;All files (*.*)",
        )
        if not path:
            return
        try:
            traj = load_simscape_trajectory_csv(path)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(
                self,
                "Trajectory load failed",
                f"Could not load {Path(path).name}:\n\n{exc}",
            )
            return
        if len(traj) == 0:
            QMessageBox.warning(
                self,
                "Empty trajectory",
                f"{Path(path).name} loaded but has no usable frames.",
            )
            return
        slot.trajectory = traj
        slot.trajectory_frame_index = 0
        # Update the button label so the user can see a trajectory is loaded.
        btn = self._pose_trajectory_buttons.get(slot_key)
        if btn is not None:
            btn.setText(f"✓ {len(traj)}f")
            btn.setToolTip(
                f"Loaded {len(traj)} frames from {Path(path).name}.\n"
                f"Time range: {traj.times[0]:.3f}s … {traj.times[-1]:.3f}s.\n"
                "Click to load a different file."
            )
        # First trajectory load auto-switches to 'Both' mode so the user
        # can immediately see the skeleton animate without having to find
        # the Playback target combo.
        if self.playback_target == "Mocap":
            with QSignalBlocker(self.combo_playback_target):
                self.combo_playback_target.setCurrentText("Both")
            self.playback_target = "Both"
        self._notify(
            f"Loaded {len(traj)}-frame trajectory for {slot_key} "
            f"from {Path(path).name}.  Playback target → Both."
        )
        self._redraw()

    def _toggle_play(self) -> None:
        """Override the parent toggle to surface a helpful message when the
        user presses Play in a target mode that won't visibly do anything.
        """
        if self.is_playing:
            self._timer.stop()
            self.is_playing = False
            self.btn_play.setText("▶ Play")
            return
        # About to start — sanity-check the chosen target.
        if self.playback_target == "Skeleton":
            visible_with_traj = [
                s for s in self.poses.values() if s.visible and s.trajectory is not None
            ]
            if not visible_with_traj:
                QMessageBox.information(
                    self,
                    "No skeleton trajectory loaded",
                    "Playback target is 'Skeleton' but no visible pose has\n"
                    "a trajectory CSV loaded yet.\n\n"
                    "Either:\n"
                    "  • Pose Slots → Trajectory Load… for one of the visible poses, or\n"
                    "  • Switch the Playback target back to 'Mocap'.",
                )
                return
        if self.df is None and self.playback_target in ("Mocap", "Both"):
            QMessageBox.information(
                self,
                "No mocap loaded",
                "Playback target is 'Mocap' or 'Both' but no xlsx file has\n"
                "been loaded yet.  Use Mocap Source → Load xlsx… first.",
            )
            return
        fps = max(1, int(self.spin_speed.value()))
        self._timer.start(int(round(1000.0 / fps)))
        self.is_playing = True
        self.btn_play.setText("⏸ Pause")

    def _phase_window_key(self) -> str:
        return _phase_key_from_label(str(self.phase_window)) or _DEFAULT_PHASE

    def _on_phase_changed(self, phase: int | str) -> None:
        if isinstance(phase, str):
            label = phase
            key = _phase_key_from_label(label) or _DEFAULT_PHASE
        else:
            key = self.phase_combo.currentData()
            label = self.phase_combo.currentText()
            if not key:
                key = _phase_key_from_label(label) or _DEFAULT_PHASE
        self.phase_window = label
        self.manual_range_widget.setVisible(key == "manual")
        if key == "manual" and isinstance(phase, str) and not self.isVisible():
            self.show()
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
        self._update_frame_counter()
        self._redraw()
        if getattr(self, "_live_view", None) is not None:
            self._live_view.set_frame(int(frame))

    def _on_frame_changed_spin(self, frame: int) -> None:
        with QSignalBlocker(self.frame_slider):
            self.frame_slider.setValue(int(frame))
        self.current_frame = int(frame)
        self._update_time_label()
        self._update_frame_counter()
        self._redraw()
        if getattr(self, "_live_view", None) is not None:
            self._live_view.set_frame(int(frame))

    def _update_frame_counter(self) -> None:
        """Refresh the ``12 / 301`` frame-counter label."""
        n = len(self.df) if self.df is not None else 0
        if hasattr(self, "lbl_frame_counter"):
            self.lbl_frame_counter.setText(
                f"{int(self.current_frame)} / {max(0, n - 1)}"
            )

    def _step_frame(self, delta: int) -> None:
        if self.df is None:
            return
        n = len(self.df)
        if delta <= -(10**8):
            self.spin_frame.setValue(0)
        elif delta >= 10**8:
            self.spin_frame.setValue(n - 1)
        else:
            new = max(0, min(n - 1, self.current_frame + delta))
            self.spin_frame.setValue(new)

    def _advance_frame(self) -> None:
        """Advance one playback step.

        Behaviour depends on `self.playback_target`:
            Mocap     - advance current_frame only
            Skeleton  - advance each visible pose's trajectory_frame_index
            Both      - advance both, time-aligned at impact
        """
        target = self.playback_target

        # 1. Mocap frame ----------------------------------------------------
        n = len(self.df) if self.df is not None else 0
        if target in ("Mocap", "Both") and n > 0:
            nxt = self.current_frame + 1
            if nxt >= n:
                if self.loop_playback:
                    nxt = 0
                else:
                    self._toggle_play()
                    return
            self.spin_frame.setValue(nxt)
        elif target == "Skeleton":
            # Without mocap advance, still consider stop condition based on
            # the longest visible trajectory.
            longest = max(
                (
                    len(s.trajectory)
                    for s in self.poses.values()
                    if s.visible and s.trajectory is not None
                ),
                default=0,
            )
            if longest == 0:
                self._toggle_play()
                return

        # 2. Skeleton trajectory frame -------------------------------------
        if target in ("Skeleton", "Both"):
            # In "Both" mode, time-align by mapping mocap_time -> sim_time
            # via the impact-frame offset.
            if target == "Both" and self.df is not None:
                self._sync_trajectory_indices_from_mocap()
            else:
                # Pure Skeleton: advance each visible trajectory by one frame.
                for slot in self.poses.values():
                    if not slot.visible or slot.trajectory is None:
                        continue
                    nxt = slot.trajectory_frame_index + 1
                    if nxt >= len(slot.trajectory):
                        nxt = 0 if self.loop_playback else (len(slot.trajectory) - 1)
                    slot.trajectory_frame_index = nxt
                self._redraw()  # redraw needed when only the skeleton moved

    def _sync_trajectory_indices_from_mocap(self) -> None:
        """In 'Both' mode, set each visible trajectory's frame index from the
        current mocap frame's time, aligned so the trajectory's first frame
        corresponds to the mocap address (A) frame and shafts hit at impact.

        Falls back to a linear stretch when impact times can't be resolved.
        """
        if self.df is None:
            return
        mocap_t = float(self.df.iloc[self.current_frame]["time"])
        a_idx = self._frame_for("A")
        i_idx = self._frame_for("I")
        if a_idx is None or i_idx is None or i_idx <= a_idx:
            # No valid window — pure linear stretch over [0, n_mocap-1].
            n_mocap = len(self.df)
            for slot in self.poses.values():
                if not slot.visible or slot.trajectory is None:
                    continue
                frac = self.current_frame / max(1, n_mocap - 1)
                slot.trajectory_frame_index = int(
                    np.clip(
                        frac * (len(slot.trajectory) - 1), 0, len(slot.trajectory) - 1
                    )
                )
            return
        mocap_t_a = float(self.df.iloc[a_idx]["time"])
        mocap_t_i = float(self.df.iloc[i_idx]["time"])
        # Map mocap_t into [0, 1] across A..I, then onto trajectory's time axis.
        for slot in self.poses.values():
            if not slot.visible or slot.trajectory is None:
                continue
            traj = slot.trajectory
            if len(traj.times) < 2:
                continue
            sim_t_a = float(traj.times[0])
            # Best impact estimate in trajectory: largest clubhead speed.
            sim_t_i = self._estimate_trajectory_impact_time(traj)
            if sim_t_i <= sim_t_a:
                # Fallback: align endpoints linearly.
                frac = (mocap_t - mocap_t_a) / max(1e-9, mocap_t_i - mocap_t_a)
                sim_t = sim_t_a + frac * (float(traj.times[-1]) - sim_t_a)
            else:
                # Linear map mocap_t -> sim_t through (A, I) anchor pair.
                slope = (sim_t_i - sim_t_a) / (mocap_t_i - mocap_t_a)
                sim_t = sim_t_a + slope * (mocap_t - mocap_t_a)
            slot.trajectory_frame_index = traj.frame_at_time(sim_t)

    def _estimate_trajectory_impact_time(self, traj: SkeletonTrajectory) -> float:
        """Return the time of peak |dCH/dt|^2 in the trajectory, or t[0] if
        clubhead positions aren't available.
        """
        if not traj.frames or "ch" not in traj.frames[0].joints:
            return float(traj.times[0]) if len(traj.times) else 0.0
        ch = np.array([f.joints["ch"] for f in traj.frames if "ch" in f.joints])
        if len(ch) < 3:
            return float(traj.times[0])
        # Forward-difference speed
        dt = np.diff(traj.times[: len(ch)])
        dt = np.where(dt == 0, 1e-6, dt)
        v = np.diff(ch, axis=0) / dt[:, None]
        speed = np.linalg.norm(v, axis=1)
        i = int(np.argmax(speed))
        return float(traj.times[i])

    def _on_frame_override_toggled(self, _state: int) -> None:
        self.frame_override_active = self.cb_use_current_frame.isChecked()
        self._redraw()

    def _set_event_to_current_frame(self) -> None:
        if self.df is None:
            return
        # Combo text is "K - Label"; key is first token.
        ev = self.combo_set_event.currentText().split(" ", 1)[0] or "T"
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
            self.events = read_event_header(
                self._xlsx_path, self.sheet_combo.currentText()
            )
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
            pivot=self.transform.pivot,
        )
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
                new_scale = float(
                    np.clip(
                        len_t / len_m, _S_RANGE[0] * _S_SCALE, _S_RANGE[1] * _S_SCALE
                    )
                )
                self.s_scale.set_value(new_scale)

        scale = max(1e-3, self.s_scale.value())

        # Solve Rz from XY-plane shaft directions (delegated to core).
        nt = float(np.linalg.norm((ch_target - mp_target)[:2]))
        nm = float(np.linalg.norm((ch_skel - mp_skel)[:2]))
        if nt < 1e-6 or nm < 1e-6:
            self._notify(
                "Shaft projection onto XY plane is degenerate (vertical "
                "shaft) — Rz cannot be solved.  Adjust manually."
            )
            return
        rz_deg = solve_shaft_rz_deg(mp_target, ch_target, mp_skel, ch_skel)

        # Lock Rx/Ry to 0 for this snap (Z-up).
        if not self.lock_xy_rotation:
            self.cb_lock_xy.setChecked(False)  # leave as-is for user; we just zero
        self.s_rx.set_value(0.0)
        self.s_ry.set_value(0.0)
        self.s_rz.set_value(rz_deg)

        # Translation: rotate+scale mp_skel about pivot, then offset to land on mp_target.
        rotated = RigidTransform(
            rx=0.0, ry=0.0, rz=rz_deg, scale=scale, pivot=self.transform.pivot
        )
        rotated_mp = rotated.apply(mp_skel[None, :])[0]
        delta = mp_target - rotated_mp
        self.s_tx.set_value(float(delta[0]))
        self.s_ty.set_value(float(delta[1]))
        self.s_tz.set_value(float(delta[2]))
        self._notify(
            f"Snapped {slot_key}: Rz={rz_deg:+.1f}°, "
            f"|shaft_target|={nt:.3f}m, |shaft_model|={nm:.3f}m"
        )

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
        # Default initial frame to T (top of backswing) if available, but keep
        # enough room for ordinary step-forward playback on short canonical
        # windows.
        t_frame = self._frame_for("T")
        if t_frame is not None:
            t_frame = min(t_frame, max(0, n - 6))
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
        for k in _EVENT_KEYS:
            v = getattr(e, f"{k}_sample")
            label = self.event_labels.get(k, k)
            sval = "?" if v != v else int(v)
            parts.append(f"{label} ({k})={sval}")
        if e.CHS_mph == e.CHS_mph:
            parts.append(f"CHS={e.CHS_mph:.1f}mph")
        return "Events:  " + "  ".join(parts)

    # ---------- save ------------------------------------------------------ #

    def _on_save_clicked(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save offsets",
            str(Path(__file__).parent / "starting_pose_offsets.json"),
            "JSON (*.json)",
        )
        if not path:
            return
        out = {
            "transform": {
                "tx": self.transform.tx,
                "ty": self.transform.ty,
                "tz": self.transform.tz,
                "rx": self.transform.rx,
                "ry": self.transform.ry,
                "rz": self.transform.rz,
                "scale": self.transform.scale,
                "pivot": list(self.transform.pivot),
                "lock_xy_rotation": self.lock_xy_rotation,
                "units": {
                    "translation": "metres",
                    "rotation": "degrees",
                    "rotation_order": "Rz @ Ry @ Rx (intrinsic XYZ)",
                },
            },
            "poses": {
                key: {
                    "visible": slot.visible,
                    "event": slot.target_event,
                    "skeleton_source": str(
                        Path(__file__).parent / f"simscape_skeleton_{key}.json"
                    ),
                }
                for key, slot in self.poses.items()
            },
            "events": {
                "A_sample": self.events.A_sample,
                "T_sample": self.events.T_sample,
                "I_sample": self.events.I_sample,
                "F_sample": self.events.F_sample,
                "CHS_mph": self.events.CHS_mph,
            },
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
                "tx": self.transform.tx,
                "ty": self.transform.ty,
                "tz": self.transform.tz,
                "rx": self.transform.rx,
                "ry": self.transform.ry,
                "rz": self.transform.rz,
                "scale": self.transform.scale,
                "pivot": list(self.transform.pivot),
            },
            "lock_xy_rotation": self.lock_xy_rotation,
            "poses": {
                key: {
                    "visible": slot.visible,
                    "event": slot.target_event,
                    "skeleton_path": str(
                        Path(__file__).parent / f"simscape_skeleton_{key}.json"
                    ),
                    "trajectory_path": (
                        slot.trajectory.source_path
                        if slot.trajectory is not None
                        else None
                    ),
                    "trajectory_frame_index": slot.trajectory_frame_index,
                }
                for key, slot in self.poses.items()
            },
            "view": {"elev": float(self.ax.elev), "azim": float(self.ax.azim)},
            "traces": {
                "clubhead": self.show_clubhead_trace,
                "midhands": self.show_midhands_trace,
                "phase": self.phase_window,
                "manual_start": self.manual_window_start,
                "manual_end": self.manual_window_end,
                "frame_marker": self.cb_frame_marker.isChecked(),
            },
            "scene": {
                "ball": self.show_ball,
                "ground": self.show_ground,
                "torso_disk": self.show_torso_disk,
            },
            "playback": {
                "current_frame": self.current_frame,
                "frame_override_active": self.frame_override_active,
                "loop": self.loop_playback,
                "fps": int(self.spin_speed.value()),
                "speed": float(self.playback_speed),
                "trail_frames": int(self.trail_frames),
                "show_trail": bool(self.show_trail),
                "target": self.playback_target,
            },
            "event_overrides": dict(self.event_overrides),
            "event_labels": {
                "preset": self.event_label_preset,
                "labels": dict(self.event_labels),
            },
            # Issue #4480: multi-source toggle state.  Older sessions will
            # not have this block; ``_apply_session`` treats absence as the
            # empty default.
            "data_sources": self._serialize_data_sources(),
        }

    def _on_save_session_clicked(self) -> None:
        ses_dir = Path(__file__).parent / "sessions"
        ses_dir.mkdir(exist_ok=True)
        sheet = self.sheet_combo.currentText() or "session"
        ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save session",
            str(ses_dir / f"{sheet}_{ts}.session.json"),
            "JSON (*.json)",
        )
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
            self, "Load session", start, "JSON (*.json)"
        )
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
            logger.warning(
                "Session schema_version=%s newer than supported %s "
                "— ignoring unknown keys.",
                ver,
                _SESSION_SCHEMA_VERSION,
            )

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

        # 3. Pose visibility + events + trajectory.
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
                    for i in range(ec.count()):
                        if ec.itemText(i).startswith(slot_d["event"] + " "):
                            ec.setCurrentIndex(i)
                            break
                self.poses[key].target_event = slot_d["event"]
            # Trajectory CSV (optional)
            traj_path = slot_d.get("trajectory_path")
            if traj_path:
                p = Path(traj_path)
                if p.exists():
                    try:
                        self.poses[key].trajectory = load_simscape_trajectory_csv(p)
                        self.poses[key].trajectory_frame_index = int(
                            slot_d.get("trajectory_frame_index", 0)
                        )
                        btn = self._pose_trajectory_buttons.get(key)
                        if btn is not None:
                            btn.setText(f"✓ {len(self.poses[key].trajectory)}f")
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("Could not reload trajectory %s: %s", p, exc)

        # 4. Transform sliders.
        tf = d.get("transform") or {}
        for attr, widget in [
            ("tx", self.s_tx),
            ("ty", self.s_ty),
            ("tz", self.s_tz),
            ("rx", self.s_rx),
            ("ry", self.s_ry),
            ("rz", self.s_rz),
            ("scale", self.s_scale),
        ]:
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
        phase_in = tr.get("phase")
        if phase_in is not None:
            # Support both v1 (legacy display strings like "Backswing (A → T)")
            # and v2 (logical keys like "backswing").
            key = _phase_key_from_label(str(phase_in)) if phase_in else None
            if key is None and phase_in in _PHASE_KEYS:
                key = phase_in
            if key in _PHASE_KEYS:
                with QSignalBlocker(self.phase_combo):
                    for i in range(self.phase_combo.count()):
                        if self.phase_combo.itemData(i) == key:
                            self.phase_combo.setCurrentIndex(i)
                            break
                self.phase_window = str(phase_in)
                self.manual_range_widget.setVisible(key == "manual")
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

        # Scene toggles
        scene = d.get("scene") or {}
        for attr, cb_name in (
            ("ball", "cb_show_ball"),
            ("ground", "cb_show_ground"),
            ("torso_disk", "cb_show_torso_disk"),
        ):
            if attr in scene:
                val = bool(scene[attr])
                setattr(self, f"show_{attr}", val)
                cb = getattr(self, cb_name, None)
                if cb is not None:
                    with QSignalBlocker(cb):
                        cb.setChecked(val)

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
        if "speed" in pb:
            with suppress(TypeError, ValueError):
                self.playback_speed = float(pb["speed"])
            if hasattr(self, "combo_speed"):
                # Snap to closest allowed speed.
                from .session_schema import ALLOWED_SPEEDS as _ALLOWED_SPEEDS

                snap = min(_ALLOWED_SPEEDS, key=lambda s: abs(s - self.playback_speed))
                idx = self.combo_speed.findText(f"{snap}×")
                if idx >= 0:
                    with QSignalBlocker(self.combo_speed):
                        self.combo_speed.setCurrentIndex(idx)
                self.playback_speed = float(snap)
        if "trail_frames" in pb:
            with suppress(TypeError, ValueError):
                self.trail_frames = int(pb["trail_frames"])
            if hasattr(self, "spin_trail"):
                with QSignalBlocker(self.spin_trail):
                    self.spin_trail.setValue(int(self.trail_frames))
        if "show_trail" in pb:
            self.show_trail = bool(pb["show_trail"])
            if hasattr(self, "cb_show_trail"):
                with QSignalBlocker(self.cb_show_trail):
                    self.cb_show_trail.setChecked(self.show_trail)
        if "target" in pb and pb["target"] in ("Mocap", "Skeleton", "Both"):
            with QSignalBlocker(self.combo_playback_target):
                self.combo_playback_target.setCurrentText(pb["target"])
            self.playback_target = pb["target"]

        # 9. Event labels.
        el = d.get("event_labels") or {}
        if "labels" in el and isinstance(el["labels"], dict):
            for k in _EVENT_KEYS:
                if k in el["labels"]:
                    self.event_labels[k] = str(el["labels"][k])
                    if hasattr(self, "_event_label_edits"):
                        with QSignalBlocker(self._event_label_edits[k]):
                            self._event_label_edits[k].setText(self.event_labels[k])
        if "preset" in el:
            self.event_label_preset = str(el["preset"])
            if hasattr(self, "event_preset_combo"):
                idx = self.event_preset_combo.findText(self.event_label_preset)
                if idx < 0:
                    idx = self.event_preset_combo.findText("Custom…")
                if idx >= 0:
                    with QSignalBlocker(self.event_preset_combo):
                        self.event_preset_combo.setCurrentIndex(idx)
        self._refresh_event_label_dependents()

        # Issue #4480: data-sources panel.  Missing block → empty default.
        self._apply_data_sources(d.get("data_sources"))

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
            entry = {
                "dx_mm": float(d_mid[0]),
                "dy_mm": float(d_mid[1]),
                "dz_mm": float(d_mid[2]),
                "norm_mm": float(np.linalg.norm(d_mid)),
            }
            ch_target = self._mocap_pos_for(slot, "club")
            if ch_target is not None and "ch" in slot.skeleton.joints:
                moved_ch = self.transform.apply(slot.skeleton.joints["ch"][None, :])[0]
                entry["clubhead_norm_mm"] = float(
                    np.linalg.norm((moved_ch - ch_target) * 1000.0)
                )
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
            line = (
                f"{key}:  |Δmid|={r['norm_mm']:5.0f} mm  "
                f"(Δ=[{r['dx_mm']:+5.0f}, {r['dy_mm']:+5.0f}, "
                f"{r['dz_mm']:+5.0f}])"
            )
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

        # Re-fit axes to the actually drawn data via the shared helper.
        # Keeps the view cropped tightly around the body and the mocap
        # target as the user scrubs through frames or loads a trajectory.
        if getattr(self, "auto_fit_axes", True):
            self._autoscale_axes_to_data()

        # Re-attach the live-view layer artists after the axes were
        # cleared by ``ax.clear()`` in this method. The controller keeps
        # its target data, so re-binding is just rebuilding artists.
        if (
            getattr(self, "_live_view", None) is not None
            and getattr(self, "_live_body_target", None) is not None
        ):
            self._live_view.set_target(body=self._live_body_target)
            self._live_view.set_frame(int(self.current_frame))

        self.lbl_residual.setText(self._residual_text())

        leg = self.ax.legend(loc="upper right", fontsize=8, ncol=1, framealpha=0.85)
        if leg is not None:
            for text in leg.get_texts():
                text.set_color("#e6e6e6")
            leg.get_frame().set_facecolor("#1f242b")
            leg.get_frame().set_edgecolor("#404652")
        self.canvas.draw()

    def _draw_floor_and_ball(self) -> None:
        if self.show_ground:
            x = np.linspace(-1.5, 1.5, 5)
            y = np.linspace(-1.5, 1.5, 5)
            X, Y = np.meshgrid(x, y)
            Z = np.zeros_like(X)
            self.ax.plot_surface(X, Y, Z, alpha=0.10, color="#22c55e")
        if self.show_ball:
            self.ax.scatter(
                [0], [0], [0.021], c="white", edgecolor="black", s=40, label="ball"
            )

    def _trace_window(self) -> tuple[int, int]:
        """Return [start, end) frame indices for trace drawing per phase setting."""
        if self.df is None:
            return (0, 0)
        n = len(self.df)
        bounds = _PHASE_BOUNDS.get(self._phase_window_key(), (None, None))
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
            self.ax.plot(
                -sub["mid_X"].values,
                sub["mid_Y"].values,
                sub["mid_Z"].values,
                color="#7dd3fc",
                linestyle="--",
                linewidth=1.2,
                alpha=0.85,
                label="mocap mid-hands trace",
            )
        if self.show_clubhead_trace and len(sub) > 1:
            self.ax.plot(
                -sub["club_X"].values,
                sub["club_Y"].values,
                sub["club_Z"].values,
                color="#fb7185",
                linestyle="--",
                linewidth=1.2,
                alpha=0.85,
                label="mocap clubhead trace",
            )
        # Phase boundary markers (start / end of selected window)
        if (self.show_midhands_trace or self.show_clubhead_trace) and len(sub) > 0:
            for idx, _marker_label, color in [
                (i0, "start", "#22c55e"),
                (min(i1 - 1, len(self.df) - 1), "end", "#a855f7"),
            ]:
                row = self.df.iloc[idx]
                if self.show_midhands_trace:
                    self.ax.scatter(
                        -row["mid_X"],
                        row["mid_Y"],
                        row["mid_Z"],
                        color=color,
                        s=40,
                        marker="^",
                        edgecolor="black",
                        linewidth=0.5,
                    )
                if self.show_clubhead_trace:
                    self.ax.scatter(
                        -row["club_X"],
                        row["club_Y"],
                        row["club_Z"],
                        color=color,
                        s=40,
                        marker="^",
                        edgecolor="black",
                        linewidth=0.5,
                    )
        # Current-frame marker (cross)
        if (
            getattr(self, "cb_frame_marker", None) is not None
            and self.cb_frame_marker.isChecked()
            and self.df is not None
            and 0 <= self.current_frame < len(self.df)
        ):
            row = self.df.iloc[self.current_frame]
            if self.show_midhands_trace:
                self.ax.scatter(
                    -row["mid_X"],
                    row["mid_Y"],
                    row["mid_Z"],
                    color="#fde047",
                    s=120,
                    marker="x",
                    linewidth=2,
                    label="current frame (mid)",
                )
            if self.show_clubhead_trace:
                self.ax.scatter(
                    -row["club_X"],
                    row["club_Y"],
                    row["club_Z"],
                    color="#fde047",
                    s=140,
                    marker="x",
                    linewidth=2,
                    label="current frame (clubhead)",
                )

    def _draw_visible_poses(self) -> None:
        for slot in self.poses.values():
            if not slot.visible:
                continue
            self._draw_one_pose(slot)
        # Live "current-frame mocap club" — always drawn so playback is
        # visible without needing to toggle traces on or set the override.
        # When the override is active OR the slider differs from every
        # visible pose's event frame, draw it as a yellow accent line.
        self._draw_current_frame_club()

    def _draw_current_frame_club(self) -> None:
        """Draw a yellow "live" mocap club at the current frame.

        This is what makes playback visible — the skeleton is static per pose
        slot, so without this draw the only thing that would change as the
        playback timer fires is the spinbox value.  We always show it; when
        the slider matches a visible pose's event frame it lands on top of
        the bold red/orange mocap club anyway.
        """
        if self.df is None or len(self.df) == 0:
            return
        f = max(0, min(self.current_frame, len(self.df) - 1))
        row = self.df.iloc[f]
        mp = np.array([-row["mid_X"], row["mid_Y"], row["mid_Z"]])
        ch = np.array([-row["club_X"], row["club_Y"], row["club_Z"]])
        # Draw thin yellow club so it doesn't obscure the bold pose-targets.
        self.ax.plot(
            [mp[0], ch[0]],
            [mp[1], ch[1]],
            [mp[2], ch[2]],
            color="#fde047",
            linewidth=2.0,
            alpha=0.95,
            label=f"current frame ({self._event_label_for_frame(f)})",
        )
        self.ax.scatter(
            *mp, color="#fde047", s=60, marker="o", edgecolor="black", linewidth=0.6
        )
        self.ax.scatter(
            *ch, color="#fde047", s=110, marker="s", edgecolor="black", linewidth=0.6
        )

    def _event_label_for_frame(self, f: int) -> str:
        """Return 'A', 'T', 'I', 'F' if frame matches an event, else 'frame N'."""
        for label in ("A", "T", "I", "F"):
            ef = self._frame_for(label)
            if ef is not None and ef == f:
                return self.event_labels.get(label, label)
        return f"frame {f}"

    def _draw_one_pose(self, slot: PoseSlot) -> None:
        # Pick which skeleton to draw: trajectory frame when active, else
        # the slot's static pose.
        skel = self._effective_skeleton(slot)
        if not skel.joints:
            return

        mp = self._mocap_pos_for(slot, "mid")
        ch = self._mocap_pos_for(slot, "club")
        if mp is not None and ch is not None:
            pts = np.array([mp, ch])
            self.ax.plot(
                pts[:, 0],
                pts[:, 1],
                pts[:, 2],
                color=slot.mocap_color,
                linewidth=4.5,
                label=f"mocap {slot.name}",
            )
            self.ax.scatter(
                *mp,
                color=slot.mocap_color,
                s=70,
                marker="o",
                edgecolor="black",
                linewidth=0.6,
            )
            self.ax.scatter(
                *ch,
                color=slot.mocap_color,
                s=130,
                marker="s",
                edgecolor="black",
                linewidth=0.6,
            )

        names = list(skel.joints.keys())
        pts = np.array([skel.joints[n] for n in names])
        moved = self.transform.apply(pts)
        pos = {n: moved[i] for i, n in enumerate(names)}

        for parent, child in skel.segments:
            if parent in pos and child in pos:
                a, b = pos[parent], pos[child]
                width = 4.5 if (parent, child) == ("mp", "ch") else 2.6
                self.ax.plot(
                    [a[0], b[0]],
                    [a[1], b[1]],
                    [a[2], b[2]],
                    color=slot.color,
                    linewidth=width,
                )

        # Torso-twist indicator: draw a small disk at the torso joint
        # whose plane normal matches the spine-to-hub direction and whose
        # in-plane "+X" axis is aligned with the LS-RS line.  Makes the
        # body coil visible at a glance.
        if self.show_torso_disk and "torso" in pos and "ls" in pos and "rs" in pos:
            self._draw_torso_disk(
                pos["torso"],
                pos["ls"],
                pos["rs"],
                pos.get("hub"),
                pos.get("spine"),
                slot.color,
            )
        # Indicate that this is a trajectory frame (not the static pose)
        # by appending the frame index to the legend label.
        legend = f"sim {slot.name}"
        if slot.trajectory is not None and self.playback_target in ("Skeleton", "Both"):
            legend = (
                f"sim {slot.name} (trajectory frame "
                f"{slot.trajectory_frame_index}/{len(slot.trajectory) - 1})"
            )
        self.ax.scatter(
            moved[:, 0], moved[:, 1], moved[:, 2], color=slot.color, s=24, label=legend
        )
        if "mp" in pos:
            self.ax.scatter(
                *pos["mp"],
                color=slot.color,
                s=70,
                marker="o",
                edgecolor="black",
                linewidth=0.6,
            )
        if "ch" in pos:
            self.ax.scatter(
                *pos["ch"],
                color=slot.color,
                s=130,
                marker="s",
                edgecolor="black",
                linewidth=0.6,
            )

    def _draw_torso_disk(
        self,
        torso: np.ndarray,
        ls: np.ndarray,
        rs: np.ndarray,
        hub: np.ndarray | None,
        spine: np.ndarray | None,
        color: str,
        radius: float = 0.18,
    ) -> None:
        """Draw a small disc at the torso joint to visualise the twist.

        The disc's normal is the spine→hub direction (or world +Z if those
        are missing); the disc is oriented so a marker arrow points in the
        LS direction along the disc plane.  This makes it instantly obvious
        which way the body has coiled.
        """
        # Build an orthonormal frame at the torso joint.
        if hub is not None and spine is not None:
            n = hub - spine
        elif hub is not None:
            n = hub - torso
        else:
            n = np.array([0.0, 0.0, 1.0])
        nn = float(np.linalg.norm(n))
        n = np.array([0.0, 0.0, 1.0]) if nn < 1e-6 else n / nn

        # In-plane axis: project (rs - ls) onto the plane orthogonal to n.
        rs_dir = rs - ls
        rs_dir = rs_dir - np.dot(rs_dir, n) * n
        rd = float(np.linalg.norm(rs_dir))
        if rd < 1e-6:
            # Pick any perpendicular if shoulders are degenerate.
            rs_dir = np.array([1.0, 0.0, 0.0])
            rs_dir = rs_dir - np.dot(rs_dir, n) * n
            rd = float(np.linalg.norm(rs_dir))
            if rd < 1e-6:
                rs_dir = np.array([0.0, 1.0, 0.0])
                rs_dir = rs_dir - np.dot(rs_dir, n) * n
                rd = float(np.linalg.norm(rs_dir))
                if rd < 1e-6:
                    return
        rs_dir = rs_dir / rd
        n_perp = np.cross(n, rs_dir)
        # Disc points
        thetas = np.linspace(0.0, 2.0 * np.pi, 24)
        disc = torso + radius * (
            np.cos(thetas)[:, None] * rs_dir + np.sin(thetas)[:, None] * n_perp
        )
        self.ax.plot(
            disc[:, 0], disc[:, 1], disc[:, 2], color=color, linewidth=1.5, alpha=0.9
        )
        # Twist-indicator arrow from torso center toward right shoulder.
        tip = torso + (radius * 1.05) * rs_dir
        self.ax.plot(
            [torso[0], tip[0]],
            [torso[1], tip[1]],
            [torso[2], tip[2]],
            color=color,
            linewidth=2.6,
            alpha=0.95,
        )
        self.ax.scatter(
            *tip, color=color, s=24, marker=">", edgecolor="black", linewidth=0.4
        )

    def _effective_skeleton(self, slot: PoseSlot) -> Skeleton:
        """Return the skeleton to draw for this slot.

        When playback target is Skeleton or Both AND the slot has a
        trajectory loaded, returns the trajectory's current frame.
        Otherwise returns the slot's static skeleton.
        """
        if (
            slot.trajectory is not None
            and len(slot.trajectory) > 0
            and self.playback_target in ("Skeleton", "Both")
        ):
            i = max(0, min(slot.trajectory_frame_index, len(slot.trajectory) - 1))
            return slot.trajectory.frames[i]
        return slot.skeleton

    # --------------------------------------------------------------------- #
    # Multi-source target panel hook (issue #4480)                          #
    # --------------------------------------------------------------------- #

    def _on_multi_source_changed(self, target: object) -> None:
        """Cache the latest ``MultiSourceTarget`` from the data-sources panel.

        Downstream consumers (cost/animation, landed in later issues) can
        read ``self._latest_multi_source`` to dispatch on whichever subset
        of targets the user toggled on.
        """
        self._latest_multi_source = target
        if target is None:
            logger.info("Data-sources panel cleared.")
        else:
            logger.info(
                "Data-sources panel: club=%s body=%s",
                getattr(target, "has_club", lambda: False)(),
                getattr(target, "has_body", lambda: False)(),
            )

    def _serialize_data_sources(self) -> dict[str, Any]:
        """Snapshot the data-sources panel for the session JSON."""
        return serialize_data_sources(self.source_panel.snapshot())

    def _apply_data_sources(self, block: dict[str, Any] | None) -> None:
        """Restore the data-sources panel from a (possibly missing) block."""
        parsed: DataSourcesBlock = parse_data_sources(block)
        self.source_panel.restore(parsed)


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
