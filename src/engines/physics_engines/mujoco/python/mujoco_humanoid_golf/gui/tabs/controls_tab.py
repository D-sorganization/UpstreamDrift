from __future__ import annotations

from src.shared.python.logging_config import get_logger
import typing
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from PyQt6 import QtCore, QtWidgets

from ...control_system import ControlSystem, ControlType
from ...sim_widget import MuJoCoSimWidget

if typing.TYPE_CHECKING:
    from ..advanced_gui import AdvancedGolfAnalysisWindow

logger = get_logger(__name__)


class ControlsTab(QtWidgets.QWidget):
    """Tab for simulation playback and actuator control."""

    SIMPLIFIED_ACTUATOR_THRESHOLD = 20

    def __init__(
        self,
        sim_widget: MuJoCoSimWidget,
        main_window: AdvancedGolfAnalysisWindow,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.sim_widget = sim_widget
        self.main_window = main_window

        # State storage
        self.actuator_groups: list[QtWidgets.QGroupBox] = []
        self.actuator_control_widgets: list[QtWidgets.QWidget] = []
        self.actuator_sliders: list[QtWidgets.QSlider] = []
        self.actuator_labels: list[QtWidgets.QLabel] = []
        self.actuator_control_types: list[QtWidgets.QComboBox] = []
        self.actuator_constant_inputs: list[QtWidgets.QDoubleSpinBox] = []
        # List of lists for coeffs? The new code uses list of lists of double spin boxes
        self.actuator_polynomial_coeffs: list[list[QtWidgets.QDoubleSpinBox]] = []
        self.actuator_damping_inputs: list[QtWidgets.QDoubleSpinBox] = []
        self.quick_camera_buttons: dict[str, QtWidgets.QPushButton] = {}
        self._simplified_notice: QtWidgets.QLabel | None = None
        self.simplified_actuator_mode = False

        self._setup_ui()

    def _setup_ui(self) -> None:
        """Create the simulation controls interface."""
        # Use simple vertical layout for the whole tab
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)

        # 1. Quick Start Help Panel (collapsible)
        self._create_help_panel(main_layout)

        # 2. Quick Camera Access Buttons
        self._create_quick_camera_buttons(main_layout)

        # 3. Simulation control buttons
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
        self.record_btn.setStyleSheet(
            "QPushButton:checked { background-color: #d62728; color: white; "
            "font-weight: bold; }"
        )

        buttons_layout.addWidget(self.play_pause_btn, 0, 0)
        buttons_layout.addWidget(self.reset_btn, 0, 1)
        buttons_layout.addWidget(self.screenshot_btn, 1, 0)
        buttons_layout.addWidget(self.record_btn, 1, 1)
        main_layout.addWidget(buttons_group)

        # Recording info
        self.recording_label = QtWidgets.QLabel("Not recording")
        self.recording_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.recording_label.setStyleSheet("font-weight: bold; padding: 5px;")
        main_layout.addWidget(self.recording_label)

        # Real-time analysis toggle
        self.chk_live_analysis = QtWidgets.QCheckBox(
            "Enable Live Analysis (CPU Intensive)"
        )
        self.chk_live_analysis.setToolTip(
            "Compute Induced Accelerations and Counterfactuals in real-time"
        )
        main_layout.addWidget(self.chk_live_analysis)

        # 4. Container for Dynamic Mode Controls (Actuators)
        self.dynamic_controls_widget = QtWidgets.QWidget()
        dynamic_layout = QtWidgets.QVBoxLayout(self.dynamic_controls_widget)
        dynamic_layout.setContentsMargins(0, 0, 0, 0)

        # Actuator filter
        filter_layout = QtWidgets.QHBoxLayout()
        filter_label = QtWidgets.QLabel("Filter actuators:")
        self.actuator_filter_input = QtWidgets.QLineEdit()
        self.actuator_filter_input.setPlaceholderText("Type actuator or group name...")
        self.actuator_filter_input.setClearButtonEnabled(True)
        self.actuator_filter_input.textChanged.connect(self.on_actuator_filter_changed)
        filter_layout.addWidget(filter_label)
        filter_layout.addWidget(self.actuator_filter_input)
        dynamic_layout.addLayout(filter_layout)

        # Scroll area for actuators
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

        # 5. Container for Kinematic Mode Controls (Joints)
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

        # Storage for joint widgets
        self.joint_widgets: dict[str, dict[str, QtWidgets.QWidget]] = {}

    def _create_help_panel(self, parent_layout: QtWidgets.QVBoxLayout) -> None:
        """Create a collapsible help panel."""
        self.help_group = QtWidgets.QGroupBox("Quick Start Guide")
        self.help_group.setCheckable(True)
        self.help_group.setChecked(False)  # Collapsed by default
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
    ) -> None:
        """Create quick access camera buttons."""
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
            )
            camera_layout.addWidget(btn)
            self.quick_camera_buttons[preset_name] = btn

        parent_layout.addWidget(camera_group)

    def _on_quick_camera_clicked(self, preset_name: str) -> None:
        self.sim_widget.set_camera(preset_name)
        if hasattr(self.main_window, "visualization_tab"):
            self.main_window.visualization_tab._update_camera_sliders()
            # Update combo box in vis tab loop back
            idx = self.main_window.visualization_tab.camera_combo.findText(preset_name)
            if idx >= 0:
                self.main_window.visualization_tab.camera_combo.setCurrentIndex(idx)

    # -------- Signal Handlers (Connected by Main Window) --------

    def on_model_loaded(self, model_name: str, config: dict) -> None:
        """Handle new model loaded from PhysicsTab."""
        self._clear_actuator_controls()

        actuators = config.get("actuators", [])
        if self.sim_widget.model and len(actuators) != self.sim_widget.model.nu:
            # Re-verify if fixup happened in PhysicsTab, but just in case
            logger.warning("Actuator count mismatch in ControlsTab update")

        self._create_actuator_controls(actuators)

    def on_mode_changed(self, mode: str) -> None:
        """Handle operating mode change (dynamic/kinematic)."""
        self.dynamic_controls_widget.setVisible(mode == "dynamic")
        self.kinematic_controls_widget.setVisible(mode == "kinematic")

        if mode == "kinematic":
            self._refresh_kinematic_controls()
            # Ensure simulation is "running" so interactive events work
            if self.sim_widget.model is not None:
                if self.play_pause_btn.isChecked():  # If paused
                    self.play_pause_btn.setChecked(False)  # Resume
                else:
                    self.sim_widget.set_running(True)

    # -------- Actuator Management --------

    def _clear_actuator_controls(self) -> None:
        """Remove all existing actuator control widgets."""
        self.actuator_sliders.clear()
        self.actuator_labels.clear()
        self.actuator_control_types.clear()
        self.actuator_constant_inputs.clear()
        self.actuator_polynomial_coeffs.clear()
        self.actuator_damping_inputs.clear()

        for widget in self.actuator_control_widgets:
            self.actuator_layout.removeWidget(widget)
            widget.deleteLater()
        self.actuator_control_widgets.clear()
        self.actuator_groups.clear()

        if self._simplified_notice:
            self.actuator_layout.removeWidget(self._simplified_notice)
            self._simplified_notice.deleteLater()
            self._simplified_notice = None

    def _create_actuator_controls(self, actuator_names: list[str]) -> None:
        groups = self._group_actuators(actuator_names)
        actuator_index = 0
        total = len(actuator_names)

        self.simplified_actuator_mode = total >= self.SIMPLIFIED_ACTUATOR_THRESHOLD

        if self.simplified_actuator_mode:
            self._simplified_notice = QtWidgets.QLabel(
                "Large musculoskeletal model detected. Showing simplified "
                "actuator controls."
            )
            self._simplified_notice.setStyleSheet(
                "background-color: #fff3cd; padding: 6px;"
            )
            self.actuator_layout.addWidget(self._simplified_notice)

        for group_name, actuators in groups.items():
            group_box = QtWidgets.QGroupBox(f"{group_name} ({len(actuators)})")
            group_box.setCheckable(True)
            group_box.setChecked(True)
            group_box.setProperty("actuator_names", actuators)

            content = QtWidgets.QWidget()
            layout = QtWidgets.QVBoxLayout(content)
            layout.setContentsMargins(0, 0, 0, 0)

            for act_name in actuators:
                if self.simplified_actuator_mode:
                    w = self._create_simplified_actuator_row(actuator_index, act_name)
                else:
                    w = self._create_advanced_actuator_control(actuator_index, act_name)
                self.actuator_control_widgets.append(w)
                layout.addWidget(w)
                actuator_index += 1

            group_box.toggled.connect(content.setVisible)

            # Wrap content in group
            gl = QtWidgets.QVBoxLayout(group_box)
            gl.addWidget(content)

            self.actuator_groups.append(group_box)
            self.actuator_layout.addWidget(group_box)

        self.actuator_layout.addStretch(1)

    def _group_actuators(self, names: list[str]) -> dict[str, list[str]]:
        groups: dict[str, list[str]] = {}
        for name in names:
            if "Shoulder" in name:
                key = "Shoulder"
            elif "Elbow" in name or "Forearm" in name:
                key = "Arm/Elbow"
            elif "Wrist" in name:
                key = "Wrist"
            elif "Spine" in name:
                key = "Spine/Torso"
            elif "Leg" in name or "Knee" in name or "Ankle" in name:
                key = "Legs"
            elif "Scap" in name:
                key = "Scapula"
            elif "Muscle" in name:
                key = "Muscles"
            else:
                key = "Other"

            if key not in groups:
                groups[key] = []
            groups[key].append(name)
        return groups

    def _create_simplified_actuator_row(
        self, index: int, name: str
    ) -> QtWidgets.QWidget:
        container = QtWidgets.QFrame()
        layout = QtWidgets.QHBoxLayout(container)
        layout.setContentsMargins(4, 2, 4, 2)

        layout.addWidget(QtWidgets.QLabel(f"<b>{name}</b>"), stretch=2)

        slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        slider.setRange(-100, 100)
        slider.valueChanged.connect(
            lambda v, i=index: self.on_actuator_slider_changed(i, v)
        )
        self.actuator_sliders.append(slider)
        layout.addWidget(slider, stretch=4)

        label = QtWidgets.QLabel("0 Nm")
        label.setMinimumWidth(60)
        self.actuator_labels.append(label)
        layout.addWidget(label)

        # Detail button
        detail_btn = QtWidgets.QPushButton("Edit...")
        detail_btn.setFixedWidth(50)
        detail_btn.clicked.connect(
            lambda _, i=index, n=name, s=slider: self.open_actuator_detail_dialog(
                i, n, s
            )
        )
        layout.addWidget(detail_btn)

        return container

    def _create_advanced_actuator_control(
        self, index: int, name: str
    ) -> QtWidgets.QWidget:
        container = QtWidgets.QFrame()
        container.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        layout = QtWidgets.QVBoxLayout(container)

        # Header
        hl = QtWidgets.QHBoxLayout()
        hl.addWidget(QtWidgets.QLabel(f"<b>{name}</b>"))

        combo = QtWidgets.QComboBox()
        combo.addItems(["Constant", "Polynomial", "Sine Wave", "Step"])
        combo.currentIndexChanged.connect(
            lambda idx, i=index: self.on_control_type_changed(i, idx)
        )
        self.actuator_control_types.append(combo)
        hl.addWidget(QtWidgets.QLabel("Type:"))
        hl.addWidget(combo)
        layout.addLayout(hl)

        # Constant Control
        ql = QtWidgets.QHBoxLayout()
        slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        slider.setRange(-100, 100)
        slider.valueChanged.connect(
            lambda v, i=index: self.on_actuator_slider_changed(i, v)
        )
        self.actuator_sliders.append(slider)

        spin = QtWidgets.QDoubleSpinBox()
        spin.setRange(-1000, 1000)
        spin.valueChanged.connect(
            lambda v, i=index: self.on_constant_value_changed(i, v)
        )
        self.actuator_constant_inputs.append(spin)

        label = QtWidgets.QLabel("0 Nm")
        self.actuator_labels.append(label)

        ql.addWidget(QtWidgets.QLabel("Value:"))
        ql.addWidget(slider)
        ql.addWidget(spin)
        ql.addWidget(label)
        layout.addLayout(ql)

        # Damping
        dl = QtWidgets.QHBoxLayout()
        d_spin = QtWidgets.QDoubleSpinBox()
        d_spin.setRange(0, 100)
        d_spin.valueChanged.connect(lambda v, i=index: self.on_damping_changed(i, v))
        self.actuator_damping_inputs.append(d_spin)
        dl.addWidget(QtWidgets.QLabel("Damping:"))
        dl.addWidget(d_spin)

        # Details button for advanced layout too (for Poly/Sine/Step params)
        detail_btn = QtWidgets.QPushButton("Params...")
        detail_btn.clicked.connect(
            lambda _, i=index, n=name, s=slider: self.open_actuator_detail_dialog(
                i, n, s
            )
        )
        dl.addWidget(detail_btn)

        layout.addLayout(dl)

        return container

    def open_actuator_detail_dialog(
        self,
        actuator_index: int,
        actuator_name: str,
        slider: QtWidgets.QSlider | None = None,
    ) -> None:
        """Open a dialog with comprehensive controls for an actuator."""
        control_system = self.sim_widget.get_control_system()
        if control_system is None:
            QtWidgets.QMessageBox.warning(
                self, "Error", "Control system not initialized."
            )
            return

        slider_sync: Callable[[float], None] | None = None
        if slider is not None:

            def slider_sync_func(value: float) -> None:
                slider.blockSignals(True)
                slider.setValue(int(value))
                slider.blockSignals(False)
                if actuator_index < len(self.actuator_labels):
                    self.actuator_labels[actuator_index].setText(f"{value:.0f} Nm")

            slider_sync = slider_sync_func

        dialog = ActuatorDetailDialog(
            control_system=control_system,
            actuator_index=actuator_index,
            actuator_name=actuator_name,
            slider_sync=slider_sync,
            parent=self,
        )
        dialog.exec()

    # Callbacks
    def on_actuator_filter_changed(self, text: str) -> None:
        text = text.lower()
        for group in self.actuator_groups:
            group_name = group.title().lower()
            actuators = group.property("actuator_names") or []
            match = (text in group_name) or any(text in a.lower() for a in actuators)
            group.setVisible(match)

    def on_actuator_slider_changed(self, index: int, value: int) -> None:
        # Update label
        if index < len(self.actuator_labels):
            self.actuator_labels[index].setText(f"{value} Nm")

        # Sync spinbox if exists
        if index < len(self.actuator_constant_inputs):
            s = self.actuator_constant_inputs[index]
            s.blockSignals(True)
            s.setValue(float(value))
            s.blockSignals(False)

        # Apply to sim
        cs = self.sim_widget.get_control_system()
        if cs:
            cs.set_constant_value(index, float(value))
            cs.set_control_type(index, ControlType.CONSTANT)

    def on_constant_value_changed(self, index: int, value: float) -> None:
        if index < len(self.actuator_sliders):
            s = self.actuator_sliders[index]
            s.blockSignals(True)
            s.setValue(int(value))
            s.blockSignals(False)

        cs = self.sim_widget.get_control_system()
        if cs:
            cs.set_constant_value(index, value)
            cs.set_control_type(index, ControlType.CONSTANT)

    def on_damping_changed(self, index: int, value: float) -> None:
        cs = self.sim_widget.get_control_system()
        if cs:
            cs.set_damping(index, value)

    def on_control_type_changed(self, index: int, type_idx: int) -> None:
        cs = self.sim_widget.get_control_system()
        if cs:
            types = [
                ControlType.CONSTANT,
                ControlType.POLYNOMIAL,
                ControlType.SINE_WAVE,
                ControlType.STEP,
            ]
            if type_idx < len(types):
                cs.set_control_type(index, types[type_idx])

    def on_play_pause_toggled(self, checked: bool) -> None:
        # Toggle simulation running state
        self.sim_widget.running = not checked
        self.play_pause_btn.setText("Resume" if checked else "Pause")

        style = self.style()
        if style:
            icon = (
                QtWidgets.QStyle.StandardPixmap.SP_MediaPlay
                if checked
                else QtWidgets.QStyle.StandardPixmap.SP_MediaPause
            )
            self.play_pause_btn.setIcon(style.standardIcon(icon))

    def on_reset_clicked(self) -> None:
        self.sim_widget.reset_state()
        self.play_pause_btn.setChecked(False)  # Resume if paused
        self.sim_widget.running = True

    def on_record_toggled(self, checked: bool) -> None:
        recorder = self.sim_widget.get_recorder()
        if checked:
            self.record_btn.setText("Stop Recording")
            if style := self.style():
                self.record_btn.setIcon(
                    style.standardIcon(QtWidgets.QStyle.StandardPixmap.SP_MediaStop)
                )
            recorder.start_recording()
        else:
            self.record_btn.setText("Start Recording")
            if style := self.style():
                self.record_btn.setIcon(
                    style.standardIcon(
                        QtWidgets.QStyle.StandardPixmap.SP_DialogYesButton
                    )
                )
            recorder.stop_recording()

    def on_take_screenshot(self) -> None:
        pixmap = self.sim_widget.label.pixmap()
        if not pixmap or pixmap.isNull():
            return

        output_dir = Path("output/screenshots")
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = (
            output_dir / f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        )
        pixmap.save(str(filename))
        logger.info("Screenshot saved: %s", filename)

        if self.main_window.statusBar():
            self.main_window.statusBar().showMessage(
                f"Screenshot saved: {filename}", 3000
            )

    def on_export_data(self) -> None:
        if hasattr(self.main_window, "on_export_data"):
            self.main_window.on_export_data()

    def _refresh_kinematic_controls(self) -> None:
        """Rebuild the kinematic joint controls."""
        # Clear existing
        while self.joint_layout.count():
            item = self.joint_layout.takeAt(0)
            if item is None:
                continue
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        # Initialize storage for cross-referencing
        self.joint_widgets = {}

        dof_info = self.sim_widget.get_dof_info()

        if not dof_info:
            self.joint_layout.addWidget(
                QtWidgets.QLabel("No controllable joints found.")
            )
            return

        for name, (min_val, max_val), current_val in dof_info:
            # Container
            container = QtWidgets.QFrame()
            container.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
            layout = QtWidgets.QVBoxLayout(container)

            # Label
            header = QtWidgets.QHBoxLayout()
            header.addWidget(QtWidgets.QLabel(f"<b>{name}</b>"))
            val_label = QtWidgets.QLabel(f"{current_val:.3f}")
            header.addWidget(val_label, alignment=QtCore.Qt.AlignmentFlag.AlignRight)
            layout.addLayout(header)

            # Slider
            slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
            steps = 1000
            slider.setRange(0, steps)

            # Set init value
            range_span = max_val - min_val
            if range_span <= 0:
                range_span = 1.0  # Protect div zero

            norm_val = (current_val - min_val) / range_span
            slider_val = int(norm_val * steps)
            slider_val = max(0, min(steps, slider_val))
            slider.setValue(slider_val)

            def _on_slider_change(
                v: int,
                n: str = name,
                mn: float = min_val,
                mx: float = max_val,
                lbl: Any = val_label,
            ) -> None:
                self._on_joint_slider_changed(n, v, mn, mx, lbl)

            slider.valueChanged.connect(_on_slider_change)

            layout.addWidget(slider)

            # Text Input for precise control
            spin = QtWidgets.QDoubleSpinBox()
            spin.setRange(min_val, max_val)
            spin.setSingleStep(0.01)
            spin.setValue(current_val)

            def _on_spin_change(
                v: float,
                n: str = name,
                mn: float = min_val,
                mx: float = max_val,
                sl: Any = slider,
                lbl: Any = val_label,
            ) -> None:
                self._on_joint_spin_changed(n, v, mn, mx, sl, lbl)

            spin.valueChanged.connect(_on_spin_change)

            layout.addWidget(spin)

            # Store references
            self.joint_widgets[name] = {"slider": slider, "spin": spin}

            self.joint_layout.addWidget(container)

    def _on_joint_slider_changed(
        self,
        name: str,
        value_int: int,
        min_val: float,
        max_val: float,
        label: QtWidgets.QLabel,
    ) -> None:
        """Handle joint slider change."""
        steps = 1000
        val = min_val + (value_int / steps) * (max_val - min_val)

        # Update label
        label.setText(f"{val:.3f}")

        # Update simulation
        self.sim_widget.set_joint_qpos(name, val)

        # Update spinbox if available
        if hasattr(self, "joint_widgets") and name in self.joint_widgets:
            spin = self.joint_widgets[name]["spin"]
            if isinstance(spin, QtWidgets.QDoubleSpinBox):
                spin.blockSignals(True)
                spin.setValue(val)
                spin.blockSignals(False)

    def _on_joint_spin_changed(
        self,
        name: str,
        value: float,
        min_val: float,
        max_val: float,
        slider: QtWidgets.QSlider,
        label: QtWidgets.QLabel,
    ) -> None:
        """Handle joint spinbox change."""
        # Update simulation
        self.sim_widget.set_joint_qpos(name, value)

        # Update label
        label.setText(f"{value:.3f}")

        # Update slider
        steps = 1000
        range_span = max_val - min_val
        if range_span <= 0:
            range_span = 1.0

        norm_val = (value - min_val) / range_span
        slider_val = int(norm_val * steps)
        slider_val = max(0, min(steps, slider_val))

        if isinstance(slider, QtWidgets.QSlider):
            slider.blockSignals(True)
            slider.setValue(slider_val)
            slider.blockSignals(False)


class ActuatorDetailDialog(QtWidgets.QDialog):
    """On-demand editor for actuator control parameters."""

    CONTROL_TYPE_LABELS = [
        "Constant",
        "Polynomial (6th order)",
        "Sine Wave",
        "Step Function",
    ]

    def __init__(
        self,
        *,
        control_system: ControlSystem,
        actuator_index: int,
        actuator_name: str,
        slider_sync: Callable[[float], None] | None,
        parent: QtWidgets.QWidget | None = None,
    ):
        """Build the detail dialog for a single actuator."""
        super().__init__(parent)
        self.control_system = control_system
        self.actuator_index = actuator_index
        self.slider_sync = slider_sync
        self.setWindowTitle(f"Actuator Detail — {actuator_name}")
        self.setModal(True)
        self.resize(500, 540)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)

        self.control = self.control_system.get_actuator_control(actuator_index)

        self.control_type_combo = QtWidgets.QComboBox()
        self.control_type_combo.addItems(self.CONTROL_TYPE_LABELS)
        self.control_type_combo.setCurrentIndex(
            self._type_to_index(self.control.control_type),
        )
        self.control_type_combo.currentIndexChanged.connect(self._on_type_changed)
        layout.addWidget(QtWidgets.QLabel("<b>Control Type</b>"))
        layout.addWidget(self.control_type_combo)

        self.constant_input = QtWidgets.QDoubleSpinBox()
        self.constant_input.setRange(-1000.0, 1000.0)
        self.constant_input.setDecimals(3)
        self.constant_input.setValue(float(self.control.constant_value))
        self.constant_input.setSuffix(" Nm")
        self.constant_input.valueChanged.connect(self._on_constant_changed)

        self.damping_input = QtWidgets.QDoubleSpinBox()
        self.damping_input.setRange(0.0, 200.0)
        self.damping_input.setDecimals(3)
        self.damping_input.setValue(float(self.control.damping))
        self.damping_input.setSuffix(" N·s/m")
        self.damping_input.valueChanged.connect(self._on_damping_changed)

        const_form = QtWidgets.QFormLayout()
        const_form.addRow("Constant Torque:", self.constant_input)
        const_form.addRow("Damping:", self.damping_input)
        layout.addLayout(const_form)

        self.poly_widget = QtWidgets.QGroupBox("Polynomial Coefficients")
        poly_layout = QtWidgets.QGridLayout(self.poly_widget)
        coeffs = self.control.get_polynomial_coeffs()
        self.poly_spinboxes: list[QtWidgets.QDoubleSpinBox] = []
        for idx in range(7):
            spin = QtWidgets.QDoubleSpinBox()
            spin.setRange(-1000.0, 1000.0)
            spin.setDecimals(4)
            spin.setValue(float(coeffs[idx]))
            spin.valueChanged.connect(
                lambda val, c_idx=idx: self._on_polynomial_changed(c_idx, val),
            )
            self.poly_spinboxes.append(spin)
            row = idx // 2
            col = (idx % 2) * 2
            poly_layout.addWidget(QtWidgets.QLabel(f"c{idx}:"), row, col)
            poly_layout.addWidget(spin, row, col + 1)
        layout.addWidget(self.poly_widget)

        self.sine_widget = QtWidgets.QGroupBox("Sine Wave Parameters")
        sine_form = QtWidgets.QFormLayout(self.sine_widget)

        self.sine_amp_spin = QtWidgets.QDoubleSpinBox()
        self.sine_amp_spin.setRange(0.0, 1000.0)
        self.sine_amp_spin.setValue(float(self.control.sine_amplitude))
        self.sine_amp_spin.setSuffix(" Nm")
        self.sine_amp_spin.valueChanged.connect(
            lambda val: self._on_sine_changed("amplitude", val),
        )
        sine_form.addRow("Amplitude:", self.sine_amp_spin)

        self.sine_freq_spin = QtWidgets.QDoubleSpinBox()
        self.sine_freq_spin.setRange(0.01, 100.0)
        self.sine_freq_spin.setDecimals(3)
        self.sine_freq_spin.setValue(float(self.control.sine_frequency))
        self.sine_freq_spin.setSuffix(" Hz")
        self.sine_freq_spin.valueChanged.connect(
            lambda val: self._on_sine_changed("frequency", val),
        )
        sine_form.addRow("Frequency:", self.sine_freq_spin)

        self.sine_phase_spin = QtWidgets.QDoubleSpinBox()
        self.sine_phase_spin.setRange(-6.28319, 6.28319)
        self.sine_phase_spin.setDecimals(3)
        self.sine_phase_spin.setValue(float(self.control.sine_phase))
        self.sine_phase_spin.setSuffix(" rad")
        self.sine_phase_spin.valueChanged.connect(
            lambda val: self._on_sine_changed("phase", val),
        )
        sine_form.addRow("Phase:", self.sine_phase_spin)
        layout.addWidget(self.sine_widget)

        self.step_widget = QtWidgets.QGroupBox("Step Function Parameters")
        step_form = QtWidgets.QFormLayout(self.step_widget)

        self.step_time_spin = QtWidgets.QDoubleSpinBox()
        self.step_time_spin.setRange(0.0, 120.0)
        self.step_time_spin.setDecimals(3)
        self.step_time_spin.setValue(float(self.control.step_time))
        self.step_time_spin.setSuffix(" s")
        self.step_time_spin.valueChanged.connect(
            lambda val: self._on_step_changed("time", val),
        )
        step_form.addRow("Step Time:", self.step_time_spin)

        self.step_value_spin = QtWidgets.QDoubleSpinBox()
        self.step_value_spin.setRange(-1000.0, 1000.0)
        self.step_value_spin.setDecimals(3)
        self.step_value_spin.setValue(float(self.control.step_value))
        self.step_value_spin.setSuffix(" Nm")
        self.step_value_spin.valueChanged.connect(
            lambda val: self._on_step_changed("value", val),
        )
        step_form.addRow("Step Value:", self.step_value_spin)
        layout.addWidget(self.step_widget)

        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Close,
        )
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        self._update_visibility()

    def _type_to_index(self, ctype: ControlType) -> int:
        mapping = {
            ControlType.CONSTANT: 0,
            ControlType.POLYNOMIAL: 1,
            ControlType.SINE_WAVE: 2,
            ControlType.STEP: 3,
        }
        return mapping.get(ctype, 0)

    def _on_type_changed(self, idx: int) -> None:
        self.control_system.set_control_type(
            self.actuator_index,
            [
                ControlType.CONSTANT,
                ControlType.POLYNOMIAL,
                ControlType.SINE_WAVE,
                ControlType.STEP,
            ][idx],
        )
        self._update_visibility()

    def _on_constant_changed(self, val: float) -> None:
        self.control_system.set_constant_value(self.actuator_index, val)
        if self.slider_sync:
            self.slider_sync(val)

    def _on_damping_changed(self, val: float) -> None:
        self.control_system.set_damping(self.actuator_index, val)

    def _on_polynomial_changed(self, coeff_idx: int, val: float) -> None:
        coeffs = self.control.get_polynomial_coeffs()
        coeffs[coeff_idx] = val
        self.control_system.set_polynomial_coeffs(self.actuator_index, coeffs)

    def _on_sine_changed(self, param: str, val: float) -> None:
        if param == "amplitude":
            self.control.sine_amplitude = val
        elif param == "frequency":
            self.control.sine_frequency = val
        elif param == "phase":
            self.control.sine_phase = val

    def _on_step_changed(self, param: str, val: float) -> None:
        if param == "time":
            self.control.step_time = val
        elif param == "value":
            self.control.step_value = val

    def _update_visibility(self) -> None:
        idx = self.control_type_combo.currentIndex()
        self.constant_input.setEnabled(idx == 0)
        self.poly_widget.setVisible(idx == 1)
        self.sine_widget.setVisible(idx == 2)
        self.step_widget.setVisible(idx == 3)
