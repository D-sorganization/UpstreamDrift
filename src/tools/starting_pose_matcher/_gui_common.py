"""Shared GUI primitives for the Starting-Pose Matcher.

This module collects the constants, stylesheet, help-text registry and
the small widget helpers that the GUI splits out of :mod:`gui` so that
:class:`MainWidget` (defined in ``gui_main_widget``) and the supporting
mixins (``gui_render_mixin``, ``gui_builders_mixin``,
``gui_session_mixin``) can all import them without circular imports.

Part of Subtask 5 / #4998 of EPIC #4993 — splitting the original
``gui.py`` (~3.1k lines) below the 1200-line file budget.
"""

from __future__ import annotations

import logging

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QSlider,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from src.tools.starting_pose_matcher.session_schema import (
    BodySkeletonStyleLiteral,
)

logger = logging.getLogger(__name__)

# Display labels shown in the body-skeleton style combo. Mapping is
# bidirectional: combo currentText() -> schema literal -> combo text.
_BODY_SKELETON_STYLE_LABELS: dict[str, BodySkeletonStyleLiteral] = {
    "Lines (default)": "lines",
    "Library shapes": "library_shapes",
}
_BODY_SKELETON_STYLE_LABEL_BY_KEY: dict[BodySkeletonStyleLiteral, str] = {
    v: k for k, v in _BODY_SKELETON_STYLE_LABELS.items()
}

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


__all__ = [
    "_BODY_SKELETON_STYLE_LABELS",
    "_BODY_SKELETON_STYLE_LABEL_BY_KEY",
    "_CAMERA_PRESETS",
    "_DEFAULT_CAMERA",
    "_HELP_TEXT",
    "_QSS",
    "_R_RANGE",
    "_R_SCALE",
    "_S_RANGE",
    "_S_SCALE",
    "_T_RANGE",
    "_T_SCALE",
    "_group_with_help",
    "_help_button",
    "_hsep",
    "LabelledControl",
    "logger",
]
