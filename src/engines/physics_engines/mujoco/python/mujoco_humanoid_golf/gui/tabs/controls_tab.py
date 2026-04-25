# ARCHITECTURE_DEBT:
# This module historically exceeds standard length metrics and accumulates excessive domain responsibility.  # noqa: E501
# It requires domain-aware structural extraction to isolate its internal classes appropriately.  # noqa: E501

"""Controls tab for the MuJoCo humanoid golf GUI.

Provides joint angle sliders, actuator controls, and simulation
playback controls for the humanoid golf simulation viewer.

Actuator management is in :mod:`actuator_controls_mixin`.
Kinematic controls are in :mod:`kinematic_controls_mixin`.
Playback handlers are in :mod:`simulation_controls_mixin`.
"""

from __future__ import annotations

import typing

from PyQt6 import QtCore, QtWidgets

from src.shared.python.logging_pkg.logging_config import get_logger
from src.shared.python.theme.style_constants import Styles

from ...sim_widget import MuJoCoSimWidget
from .actuator_controls_mixin import _ActuatorControlsMixin
from .kinematic_controls_mixin import _KinematicControlsMixin
from .simulation_controls_mixin import _SimulationControlsMixin

if typing.TYPE_CHECKING:
    from ..advanced_gui import AdvancedGolfAnalysisWindow

logger = get_logger(__name__)


class ControlsTab(
    _SimulationControlsMixin,
    _ActuatorControlsMixin,
    _KinematicControlsMixin,
    QtWidgets.QWidget,
):
    """Tab for simulation playback and actuator control."""

    SIMPLIFIED_ACTUATOR_THRESHOLD = 20

    def __init__(
        self,
        sim_widget: MuJoCoSimWidget,
        main_window: AdvancedGolfAnalysisWindow,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        if not (sim_widget is not None):
            raise ValueError("sim_widget must be provided")
        super().__init__(parent)
        self.sim_widget = sim_widget
        self.main_window = main_window

        # Actuator state (used by _ActuatorControlsMixin)
        self.actuator_groups: list[QtWidgets.QGroupBox] = []
        self.actuator_control_widgets: list[QtWidgets.QWidget] = []
        self.actuator_sliders: list[QtWidgets.QSlider] = []
        self.actuator_labels: list[QtWidgets.QLabel] = []
        self.actuator_control_types: list[QtWidgets.QComboBox] = []
        self.actuator_constant_inputs: list[QtWidgets.QDoubleSpinBox] = []
        self.actuator_polynomial_coeffs: list[list[QtWidgets.QDoubleSpinBox]] = []
        self.actuator_damping_inputs: list[QtWidgets.QDoubleSpinBox] = []
        self.quick_camera_buttons: dict[str, QtWidgets.QPushButton] = {}
        self._simplified_notice: QtWidgets.QLabel | None = None
        self.simplified_actuator_mode = False

        self._setup_ui()

    def _setup_ui(self) -> None:
        """Create the simulation controls interface."""
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)

        self._create_help_panel(main_layout)
        self._create_quick_camera_buttons(main_layout)
        self._create_simulation_buttons(main_layout)
        self._create_recording_info(main_layout)
        self._create_dynamic_controls(main_layout)
        self._create_kinematic_controls(main_layout)

        self.joint_widgets: dict[str, dict[str, QtWidgets.QWidget]] = {}

    def _create_simulation_buttons(self, main_layout: QtWidgets.QVBoxLayout) -> None:
        if not (main_layout is not None):
            raise ValueError("main_layout must be provided")
        buttons_group = QtWidgets.QGroupBox("Simulation Control")
        buttons_layout = QtWidgets.QGridLayout(buttons_group)

        style = self.style()

        self.play_pause_btn = QtWidgets.QPushButton("Pause")
        self.play_pause_btn.setCheckable(True)
        if style:
            self.play_pause_btn.setIcon(
                style.standardIcon(QtWidgets.QStyle.StandardPixmap.SP_MediaPause)
            )
        self.play_pause_btn.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.play_pause_btn.toggled.connect(self.on_play_pause_toggled)
        self.play_pause_btn.setToolTip("Pause/Resume simulation (Shortcut: Space)")

        self.reset_btn = QtWidgets.QPushButton("Reset")
        if style:
            self.reset_btn.setIcon(
                style.standardIcon(QtWidgets.QStyle.StandardPixmap.SP_BrowserReload)
            )
        self.reset_btn.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.reset_btn.clicked.connect(self.on_reset_clicked)
        self.reset_btn.setToolTip("Reset simulation to initial state (Shortcut: R)")

        self.screenshot_btn = QtWidgets.QPushButton("Screenshot")
        if style:
            self.screenshot_btn.setIcon(
                style.standardIcon(QtWidgets.QStyle.StandardPixmap.SP_DialogSaveButton)
            )
        self.screenshot_btn.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.screenshot_btn.clicked.connect(self.on_take_screenshot)
        self.screenshot_btn.setToolTip("Save screenshot to output/screenshots/")

        self.record_btn = QtWidgets.QPushButton("Start Recording")
        if style:
            self.record_btn.setIcon(
                style.standardIcon(QtWidgets.QStyle.StandardPixmap.SP_DialogYesButton)
            )
        self.record_btn.setCheckable(True)
        self.record_btn.toggled.connect(self.on_record_toggled)
        self.record_btn.setToolTip("Record simulation data for analysis and export")
        self.record_btn.setStyleSheet(Styles.BTN_RECORD_CHECKED)

        buttons_layout.addWidget(self.play_pause_btn, 0, 0)
        buttons_layout.addWidget(self.reset_btn, 0, 1)
        buttons_layout.addWidget(self.screenshot_btn, 1, 0)
        buttons_layout.addWidget(self.record_btn, 1, 1)
        main_layout.addWidget(buttons_group)

    def _create_recording_info(self, main_layout: QtWidgets.QVBoxLayout) -> None:
        if not (main_layout is not None):
            raise ValueError("main_layout must be provided")
        self.recording_label = QtWidgets.QLabel("Not recording")
        self.recording_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.recording_label.setStyleSheet(Styles.RECORDING_IDLE)
        main_layout.addWidget(self.recording_label)

        self.chk_live_analysis = QtWidgets.QCheckBox(
            "Enable Live Analysis (CPU Intensive)"
        )  # noqa: E501
        self.chk_live_analysis.setToolTip(
            "Compute Induced Accelerations and Counterfactuals in real-time"
        )
        main_layout.addWidget(self.chk_live_analysis)

    def _create_dynamic_controls(self, main_layout: QtWidgets.QVBoxLayout) -> None:
        if not (main_layout is not None):
            raise ValueError("main_layout must be provided")
        self.dynamic_controls_widget = QtWidgets.QWidget()
        dynamic_layout = QtWidgets.QVBoxLayout(self.dynamic_controls_widget)
        dynamic_layout.setContentsMargins(0, 0, 0, 0)

        filter_layout = QtWidgets.QHBoxLayout()
        filter_label = QtWidgets.QLabel("Filter actuators:")
        self.actuator_filter_input = QtWidgets.QLineEdit()
        self.actuator_filter_input.setPlaceholderText("Type actuator or group name...")
        self.actuator_filter_input.setClearButtonEnabled(True)
        self.actuator_filter_input.textChanged.connect(self.on_actuator_filter_changed)
        filter_layout.addWidget(filter_label)
        filter_layout.addWidget(self.actuator_filter_input)
        dynamic_layout.addLayout(filter_layout)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )

        self.actuator_container = QtWidgets.QWidget()
        self.actuator_layout = QtWidgets.QVBoxLayout(self.actuator_container)
        scroll.setWidget(self.actuator_container)
        dynamic_layout.addWidget(scroll)

        main_layout.addWidget(self.dynamic_controls_widget)

    def _create_kinematic_controls(self, main_layout: QtWidgets.QVBoxLayout) -> None:
        if not (main_layout is not None):
            raise ValueError("main_layout must be provided")
        self.kinematic_controls_widget = QtWidgets.QWidget()
        self.kinematic_controls_widget.setVisible(False)
        kinematic_layout = QtWidgets.QVBoxLayout(self.kinematic_controls_widget)
        kinematic_layout.setContentsMargins(0, 0, 0, 0)

        k_scroll = QtWidgets.QScrollArea()
        k_scroll.setWidgetResizable(True)
        self.joint_container = QtWidgets.QWidget()
        self.joint_layout = QtWidgets.QVBoxLayout(self.joint_container)
        k_scroll.setWidget(self.joint_container)
        kinematic_layout.addWidget(k_scroll)

        main_layout.addWidget(self.kinematic_controls_widget)

    def _create_help_panel(self, parent_layout: QtWidgets.QVBoxLayout) -> None:
        """Create a collapsible help panel."""
        if not (parent_layout is not None):
            raise ValueError("parent_layout must be provided")
        self.help_group = QtWidgets.QGroupBox("Quick Start Guide")
        self.help_group.setCheckable(True)
        self.help_group.setChecked(False)
        help_layout = QtWidgets.QVBoxLayout(self.help_group)

        help_text = (
            "1. <b>Physics Tab:</b> Select Model and Operating Mode.<br>"
            "2. <b>Dynamic Mode:</b> Apply torques/forces to joints/muscles.<br>"
            "3. <b>Kinematic Mode:</b> Directly manipulate pose (drag bodies).<br>"
            "4. <b>Visualization Tab:</b> Change camera, colors, and show forces.<br>"
            "5. <b>Analysis Tab:</b> View real-time energy and biomechanics plots."
        )
        label = QtWidgets.QLabel(help_text)
        label.setWordWrap(True)
        help_layout.addWidget(label)
        parent_layout.addWidget(self.help_group)

    def _create_quick_camera_buttons(
        self, parent_layout: QtWidgets.QVBoxLayout
    ) -> None:  # noqa: E501
        """Create quick access camera buttons."""
        if not (parent_layout is not None):
            raise ValueError("parent_layout must be provided")
        camera_group = QtWidgets.QGroupBox("Quick Camera Views")
        camera_layout = QtWidgets.QHBoxLayout(camera_group)

        presets = [
            ("Front", "front"),
            ("Side", "side"),
            ("Top", "top"),
            ("Follow", "follow"),
        ]
        for label, preset_name in presets:
            btn = QtWidgets.QPushButton(label)
            btn.setToolTip(f"Switch to {label} view")
            btn.clicked.connect(
                lambda checked, n=preset_name: self._on_quick_camera_clicked(n)
            )  # noqa: E501
            camera_layout.addWidget(btn)
            self.quick_camera_buttons[preset_name] = btn

        parent_layout.addWidget(camera_group)

    def _on_quick_camera_clicked(self, preset_name: str) -> None:
        self.sim_widget.set_camera(preset_name)
        if hasattr(self.main_window, "visualization_tab"):
            self.main_window.update_visualization_camera_sliders()
            self.main_window.set_visualization_camera_preset(preset_name)

    # -------- Signal Handlers (Connected by Main Window) --------

    def on_model_loaded(self, model_name: str, config: dict) -> None:
        """Handle new model loaded from PhysicsTab."""
        if not (model_name is not None):
            raise ValueError("model_name must be provided")
        self._clear_actuator_controls()

        actuators = config.get("actuators", [])
        if (
            self.sim_widget.has_model()
            and len(actuators) != self.sim_widget.get_num_actuators()
        ):
            logger.warning("Actuator count mismatch in ControlsTab update")

        self._create_actuator_controls(actuators)

    def on_mode_changed(self, mode: str) -> None:
        """Handle operating mode change (dynamic/kinematic)."""
        if not (mode is not None):
            raise ValueError("mode must be provided")
        self.dynamic_controls_widget.setVisible(mode == "dynamic")
        self.kinematic_controls_widget.setVisible(mode == "kinematic")

        if mode == "kinematic":
            self._refresh_kinematic_controls()
            if self.sim_widget.has_model():
                if self.play_pause_btn.isChecked():
                    self.play_pause_btn.setChecked(False)
                else:
                    self.sim_widget.set_running(True)
