"""
Abstract base class for all pendulum model control panels.

DRY: Extracts the shared infrastructure duplicated across
ControlsWidget, ControlsWidgetTriple, and ControlsWidgetGolfer:

- Common signal definitions
- Hidden compatibility widgets (playback, slider, speed)
- Playback control methods (set_slider_range, set_slider_value, stop_playback)
- Preset combo builder
- Export section builder
- Run/Reset button builder
- Gravity checkbox builder
- Function generator integration pattern

Subclasses implement:
- PRESETS class attribute
- _build_model_sections(layout) â€” add model-specific parameter sections
- _apply_preset(name) â€” fill inputs from preset data
- get_params() â€” parse inputs into simulation parameter dict
- _get_joint_names() â€” return list of joint names for signal toolkit
- _get_torque_inputs() â€” return mapping of joint name â†’ LabeledInput
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING

import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from .controls_utils import STYLE_CHECK, STYLE_GROUP

if TYPE_CHECKING:
    from .controls_widget import LabeledInput


# ---------------------------------------------------------------------------
# Button style constants (DRY â€” shared across all models)
# ---------------------------------------------------------------------------

STYLE_BTN_RUN = (
    "QPushButton{background:#2d6b3f;color:white;border:none;"
    "border-radius:5px;padding:10px;font-size:13px;font-weight:bold;}"
    "QPushButton:hover{background:#3a8a52;}"
    "QPushButton:pressed{background:#1f5030;}"
)

STYLE_BTN_RESET = (
    "QPushButton{background:#5a3030;color:white;border:none;"
    "border-radius:5px;padding:10px;font-size:13px;}"
    "QPushButton:hover{background:#7a4040;}"
)

STYLE_BTN_EXPORT = (
    "QPushButton{background:#303050;color:#c0c0e0;"
    "border:1px solid #505068;border-radius:4px;padding:6px 10px;}"
    "QPushButton:hover{background:#3a3a60;}"
)

STYLE_BTN_FUNCGEN = (
    "QPushButton{background:#282848;color:#b0b0e0;border:1px solid #404068;"
    "border-radius:4px;padding:4px 8px;font-size:10px;}"
    "QPushButton:hover{background:#32326a;}"
)

STYLE_COMBO = "background:#2a2a38;color:#e0e0f0;border:1px solid #505068;border-radius:3px;padding:4px;"


class ControlsWidgetBase(QWidget):
    """Abstract base for pendulum model control panels.

    DRY resolution (issue #2353): Common signals, preset wiring, playback
    controls, export section, and run/reset builders are defined here and
    shared by ``ControlsWidget`` (double pendulum) and
    ``ControlsWidgetTriple`` (triple pendulum).

    The ``_build_sim_section`` and ``_build_dissipation_section`` methods
    in the subclasses appear similar but differ by joint count, label
    text, and field arrangement, making them intentionally model-specific.
    If a third pendulum model is added, consider introducing a
    ``_build_n_joint_sim_section(n)`` factory method here.

    Subclass contract
    -----------------
    - Define ``PRESETS`` as a class-level dict.
    - Implement ``_build_model_sections(layout)`` to add model-specific UI.
    - Implement ``_apply_preset(name)`` to populate inputs from preset data.
    - Implement ``get_params()`` to parse and return simulation parameters.
    - Implement ``_get_joint_names()`` â†’ list[str] for signal toolkit dialog.
    - Implement ``_get_torque_inputs()`` â†’ dict[str, LabeledInput].
    """

    # â”€â”€ Common signals (identical across all three widgets) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    run_requested = pyqtSignal()
    reset_requested = pyqtSignal()
    play_toggled = pyqtSignal(bool)
    speed_changed = pyqtSignal(float)
    frame_changed = pyqtSignal(int)
    export_data_requested = pyqtSignal()
    export_video_requested = pyqtSignal()
    export_image_requested = pyqtSignal()
    gravity_changed = pyqtSignal(bool)
    forces_changed = pyqtSignal(bool)

    # Subclass must define PRESETS dict
    PRESETS: dict = {}

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._is_playing = False

    # ------------------------------------------------------------------
    # Shared section builders
    # ------------------------------------------------------------------

    def _build_preset_section(self) -> QGroupBox:
        """Build preset combo box.  Shared across all models."""
        box = QGroupBox("Preset")
        box.setStyleSheet(STYLE_GROUP)
        layout = QVBoxLayout(box)
        layout.setContentsMargins(4, 10, 4, 4)
        self.preset_combo = QComboBox()
        self.preset_combo.setStyleSheet(STYLE_COMBO)
        for name in self.PRESETS:
            self.preset_combo.addItem(name)
        self.preset_combo.currentTextChanged.connect(self._apply_preset)
        layout.addWidget(self.preset_combo)
        return box

    def _build_run_reset_buttons(self) -> QHBoxLayout:
        """Build Run / Reset button row.  Shared across all models."""
        btn_layout = QHBoxLayout()
        self.btn_run = QPushButton("Run Simulation")
        self.btn_run.setStyleSheet(STYLE_BTN_RUN)
        self.btn_run.clicked.connect(self.run_requested.emit)

        self.btn_reset = QPushButton("Reset")
        self.btn_reset.setStyleSheet(STYLE_BTN_RESET)
        self.btn_reset.clicked.connect(self.reset_requested.emit)

        btn_layout.addWidget(self.btn_run, stretch=2)
        btn_layout.addWidget(self.btn_reset, stretch=1)
        return btn_layout

    def _build_hidden_compat_widgets(self) -> None:
        """Create hidden playback widgets for signal compatibility.

        These are never shown but must exist so that SimulationPanel's
        existing signal connections (play_toggled, frame_changed, etc.)
        continue to work.  Controls are driven from the ToolStrip.
        """
        # Run/Reset buttons â€” may already exist if subclass called
        # _build_run_reset_buttons().  Create as hidden fallbacks so
        # SimulationPanel._on_run can always reference btn_run/btn_reset.
        if not hasattr(self, "btn_run"):
            from PyQt6.QtWidgets import QPushButton as _QPB

            self.btn_run = _QPB()
            self.btn_run.hide()
        if not hasattr(self, "btn_reset"):
            from PyQt6.QtWidgets import QPushButton as _QPB

            self.btn_reset = _QPB()
            self.btn_reset.hide()
        self.btn_play = QPushButton()
        self.btn_play.setCheckable(True)
        self.btn_play.toggled.connect(self._on_play_toggled)
        self.speed_spin = QDoubleSpinBox()
        self.speed_spin.setRange(0.05, 20.0)
        self.speed_spin.setSingleStep(0.1)
        self.speed_spin.setValue(1.0)
        self.speed_spin.valueChanged.connect(lambda v: self.speed_changed.emit(v))
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 100)
        self.slider.valueChanged.connect(self.frame_changed.emit)

    def _build_torque_clamp_section_ndof(
        self, joint_labels: list[str], defaults: list[float]
    ) -> QGroupBox:
        """Build N-DOF torque saturation section.

        Parameters
        ----------
        joint_labels : list of display labels, one per joint
        defaults : list of default max torque values (NÂ·m)

        Creates:
            self.chk_clamp : QCheckBox
            self.clamp_inputs : list[LabeledInput] (one per joint)
        """
        from .controls_widget import LabeledInput as _LI

        assert len(joint_labels) == len(defaults)
        box = QGroupBox("Torque Saturation")
        box.setStyleSheet(STYLE_GROUP)
        layout = QVBoxLayout(box)
        layout.setContentsMargins(4, 12, 4, 4)
        layout.setSpacing(3)
        self.chk_clamp = QCheckBox("Enable torque clamping")
        self.chk_clamp.setStyleSheet(STYLE_CHECK)
        layout.addWidget(self.chk_clamp)
        self.clamp_inputs: list[_LI] = []
        for label, default in zip(joint_labels, defaults, strict=False):
            inp = _LI(f"Max |Ï„ {label}|", str(default), f"Max {label} torque Â±(NÂ·m)")
            layout.addWidget(inp)
            self.clamp_inputs.append(inp)
        return box

    def _build_joint_limits_section_ndof(
        self,
        joint_labels: list[str],
        min_defaults: list[float],
        max_defaults: list[float],
    ) -> QGroupBox:
        """Build N-DOF joint limits section.

        Parameters
        ----------
        joint_labels : list of display labels, one per joint
        min_defaults / max_defaults : default angle limits in degrees

        Creates:
            self.chk_limits : QCheckBox
            self.limit_min_inputs : list[LabeledInput]
            self.limit_max_inputs : list[LabeledInput]
            self.inp_limit_k : LabeledInput (penalty stiffness)
        """
        from .controls_widget import LabeledInput as _LI

        assert len(joint_labels) == len(min_defaults) == len(max_defaults)
        box = QGroupBox("Joint Limits")
        box.setStyleSheet(STYLE_GROUP)
        layout = QVBoxLayout(box)
        layout.setContentsMargins(4, 12, 4, 4)
        layout.setSpacing(3)
        self.chk_limits = QCheckBox("Enable joint limits")
        self.chk_limits.setStyleSheet(STYLE_CHECK)
        layout.addWidget(self.chk_limits)
        self.limit_min_inputs: list[_LI] = []
        self.limit_max_inputs: list[_LI] = []
        for label, lo, hi in zip(
            joint_labels, min_defaults, max_defaults, strict=False
        ):
            row_layout = QHBoxLayout()
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(4)
            inp_min = _LI(f"{label} minÂ°", str(lo), f"Min {label} angle (deg)", 70)
            inp_max = _LI(f"{label} maxÂ°", str(hi), f"Max {label} angle (deg)", 70)
            row_layout.addWidget(inp_min, stretch=1)
            row_layout.addWidget(inp_max, stretch=1)
            layout.addLayout(row_layout)
            self.limit_min_inputs.append(inp_min)
            self.limit_max_inputs.append(inp_max)
        self.inp_limit_k = _LI("K (NÂ·m/rad)", "500", "Penalty stiffness", 70)
        layout.addWidget(self.inp_limit_k)
        return box

    def _parse_torque_limits(self) -> list[float] | None:
        """Parse torque clamp values from clamp_inputs.

        Returns None if clamping is disabled, else list of max torques.
        """
        from .controls_utils import parse_float

        if not hasattr(self, "chk_clamp") or not self.chk_clamp.isChecked():
            return None
        return [
            parse_float(inp, f"Max torque {i}")
            for i, inp in enumerate(self.clamp_inputs)
        ]

    def _parse_joint_limits(self) -> tuple[list[float], list[float], float] | None:
        """Parse joint limit values.

        Returns None if limits disabled, else (min_rads, max_rads, stiffness).
        """
        from .controls_utils import parse_float

        if not hasattr(self, "chk_limits") or not self.chk_limits.isChecked():
            return None
        mins = [
            np.radians(parse_float(inp, f"Min {i}"))
            for i, inp in enumerate(self.limit_min_inputs)
        ]
        maxs = [
            np.radians(parse_float(inp, f"Max {i}"))
            for i, inp in enumerate(self.limit_max_inputs)
        ]
        k = parse_float(self.inp_limit_k, "Limit K")
        return mins, maxs, k

    def _build_export_section(self) -> QGroupBox:
        """Build export buttons (Data + Video + Image).  Shared across all models."""
        box = QGroupBox("Export")
        box.setStyleSheet(STYLE_GROUP)
        layout = QHBoxLayout(box)
        layout.setContentsMargins(4, 10, 4, 4)
        self.btn_export_data = QPushButton("Export Data")
        self.btn_export_data.setStyleSheet(STYLE_BTN_EXPORT)
        self.btn_export_data.clicked.connect(self.export_data_requested.emit)
        self.btn_export_video = QPushButton("Export Video")
        self.btn_export_video.setStyleSheet(STYLE_BTN_EXPORT)
        self.btn_export_video.clicked.connect(self.export_video_requested.emit)
        self.btn_export_image = QPushButton("Export Image")
        self.btn_export_image.setStyleSheet(STYLE_BTN_EXPORT)
        self.btn_export_image.clicked.connect(self.export_image_requested.emit)
        layout.addWidget(self.btn_export_data)
        layout.addWidget(self.btn_export_video)
        layout.addWidget(self.btn_export_image)
        return box

    def _build_gravity_section(self) -> QGroupBox:
        """Build gravity toggle (and optional force toggle, hidden).

        Shared across all models.  Force checkbox is hidden because
        #1143 moved it to the toolstrip.
        """
        box = QGroupBox("Physics & Display")
        box.setStyleSheet(STYLE_GROUP)
        layout = QVBoxLayout(box)
        layout.setContentsMargins(4, 12, 4, 4)
        layout.setSpacing(4)
        # Gravity is always on (#1209) â€” no checkbox needed

        self.chk_forces = QCheckBox("â†—  Show force vectors")
        self.chk_forces.setChecked(False)
        self.chk_forces.setStyleSheet(STYLE_CHECK)
        self.chk_forces.toggled.connect(self.forces_changed.emit)
        self.chk_forces.setVisible(False)  # #1143: force toggle lives in toolstrip

        return box

    def _build_funcgen_button(self) -> QPushButton:
        """Build a "Signal Toolkitâ€¦" button.  Shared across all models."""
        btn = QPushButton("ðŸ“ˆ Signal Toolkitâ€¦")
        btn.setToolTip("Design a waveform and import as torque coefficients")
        btn.setStyleSheet(STYLE_BTN_FUNCGEN)
        btn.clicked.connect(self._open_function_generator)
        return btn

    # ------------------------------------------------------------------
    # Function generator integration
    # ------------------------------------------------------------------

    def _open_function_generator(self) -> None:
        """Open Signal Toolkit as a dialog for torque design."""
        from .function_generator_dialog import FunctionGeneratorDialog

        dlg = FunctionGeneratorDialog(self, joint_names=self._get_joint_names())
        dlg.torque_imported.connect(self._on_torque_imported)
        dlg.exec()

    def _on_torque_imported(self, joint: str, coeffs: list[float]) -> None:
        """Receive torque profile imported from Function Generator.

        Pre: joint must match one of _get_torque_inputs() keys (case-insensitive)
        Pre: len(coeffs) >= 1
        """
        inputs = self._get_torque_inputs()
        key = joint.lower()
        valid_keys = {k.lower() for k in inputs}
        assert key in valid_keys, (
            f"Unknown joint '{joint}', expected one of {valid_keys}"
        )
        assert len(coeffs) >= 1, "Coefficients list must not be empty"

        coeffs_str = ", ".join(f"{c:.4g}" for c in coeffs)
        # Find the matching input (case-insensitive)
        for name, widget in inputs.items():
            if name.lower() == key:
                widget.set_value(coeffs_str)
                break
        self._update_torque_preview()

    # ------------------------------------------------------------------
    # Playback (shared across all models)
    # ------------------------------------------------------------------

    def _on_play_toggled(self, checked: bool) -> None:
        assert checked is not None, "checked must be provided"
        self._is_playing = checked
        self.btn_play.setText("Pause" if checked else "Play")
        self.play_toggled.emit(checked)

    def set_slider_range(self, max_val: int) -> None:
        """Pre: max_val >= 0"""
        assert max_val >= 0, f"Slider max must be non-negative, got {max_val}"
        self.slider.setRange(0, max_val)

    def set_slider_value(self, val: int) -> None:
        """Pre: 0 <= val <= slider.maximum()"""
        assert 0 <= val <= self.slider.maximum(), (
            f"Slider value {val} out of range [0, {self.slider.maximum()}]"
        )
        self.slider.blockSignals(True)
        self.slider.setValue(val)
        self.slider.blockSignals(False)

    def stop_playback(self) -> None:
        """Reset the play/pause toggle to the stopped state."""
        self.btn_play.setChecked(False)

    def gravity_on(self) -> bool:
        """Return whether gravity is enabled (always True since #1209)."""
        return True  # Gravity always on (#1209)

    def show_forces(self) -> bool:
        """Return whether force vector display is enabled."""
        return self.chk_forces.isChecked()

    # ------------------------------------------------------------------
    # Abstract interface â€” subclasses must implement
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def _apply_preset(self, name: str) -> None:
        """Fill all inputs from the named preset.

        Parameters
        ----------
        name : str
            Name of the preset to apply
        """

    @abc.abstractmethod
    def get_params(self) -> dict:
        """Parse all inputs and return a simulation parameter dict.

        Returns
        -------
        dict
            Dictionary of simulation parameters
        """

    @abc.abstractmethod
    def _get_joint_names(self) -> list[str]:
        """Return joint names for the function generator dialog.

        Returns
        -------
        list[str]
            List of joint names
        """

    @abc.abstractmethod
    def _get_torque_inputs(self) -> dict[str, LabeledInput]:
        """Return mapping of joint name â†’ torque input widget.

        Returns
        -------
        dict[str, LabeledInput]
            Mapping from joint name to torque input widget
        """

    @staticmethod
    def _uai_or_parse(widget: object, label: str) -> float:
        """Extract SI value from UnitAwareInput or parse from LabeledInput.

        Works regardless of whether upstream_drift_tools is installed.
        Subclasses should use this when parsing inputs that might be
        UnitAwareInput or plain LabeledInput widgets.
        """
        assert widget is not None, "widget must be provided"
        from .controls_utils import parse_float

        try:
            from upstream_drift_tools.ui.widgets.unit_aware_input import (
                UnitAwareInput as _UAI,
            )

            if isinstance(widget, _UAI):
                return widget.value_si()
        except ImportError:
            pass
        return parse_float(widget, label)  # type: ignore[arg-type]

    def _update_torque_preview(self) -> None:
        """Update the torque preview widget.  Override in subclass if needed."""
