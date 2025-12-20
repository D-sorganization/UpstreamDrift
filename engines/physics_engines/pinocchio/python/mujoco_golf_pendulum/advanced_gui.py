"""Advanced professional GUI for golf swing analysis.

This module provides a comprehensive interface with:
- Simulation controls and visualization
- Real-time biomechanical analysis
- Advanced plotting and data export
- Force/torque vector visualization
- Camera controls
"""

import csv
import json
import logging
import typing
from collections.abc import Callable
from pathlib import Path

import mujoco
from PyQt6 import QtCore, QtGui, QtWidgets

from .control_system import ControlSystem, ControlType
from .interactive_manipulation import ConstraintType
from .linkage_mechanisms import LINKAGE_CATALOG
from .models import (
    ADVANCED_BIOMECHANICAL_GOLF_SWING_XML,
    CHAOTIC_PENDULUM_XML,
    DOUBLE_PENDULUM_XML,
    FULL_BODY_GOLF_SWING_XML,
    MYOARM_SIMPLE_PATH,
    MYOBODY_PATH,
    MYOUPPERBODY_PATH,
    TRIPLE_PENDULUM_XML,
    UPPER_BODY_GOLF_SWING_XML,
)
from .plotting import GolfSwingPlotter, MplCanvas
from .sim_widget import MuJoCoSimWidget

logger = logging.getLogger(__name__)


class AdvancedGolfAnalysisWindow(QtWidgets.QMainWindow):
    """Professional golf swing analysis application with comprehensive features."""

    SIMPLIFIED_ACTUATOR_THRESHOLD: typing.Final[int] = 60

    def __init__(self) -> None:
        """Docstring for __init__."""
        super().__init__()

        self.setWindowTitle("Golf Swing Biomechanical Analysis Suite")
        self.resize(1600, 900)

        # Model configurations
        self.model_configs = [
            {
                "name": "chaotic_pendulum",
                "xml": CHAOTIC_PENDULUM_XML,
                "actuators": ["Base Drive (Forcing)", "Pendulum Control"],
            },
            {
                "name": "double",
                "xml": DOUBLE_PENDULUM_XML,
                "actuators": ["Shoulder", "Wrist"],
            },
            {
                "name": "triple",
                "xml": TRIPLE_PENDULUM_XML,
                "actuators": ["Shoulder", "Elbow", "Wrist"],
            },
            {
                "name": "upper_body",
                "xml": UPPER_BODY_GOLF_SWING_XML,
                "actuators": [
                    "Spine Rotation",
                    "L Shoulder Swing",
                    "L Shoulder Lift",
                    "L Elbow",
                    "L Wrist",
                    "R Shoulder Swing",
                    "R Shoulder Lift",
                    "R Elbow",
                    "R Wrist",
                    "Club Wrist",
                ],
            },
            {
                "name": "full_body",
                "xml": FULL_BODY_GOLF_SWING_XML,
                "actuators": [
                    "L Ankle",
                    "L Knee",
                    "R Ankle",
                    "R Knee",
                    "Spine Bend",
                    "Spine Rotation",
                    "L Shoulder Swing",
                    "L Shoulder Lift",
                    "L Elbow",
                    "L Wrist",
                    "R Shoulder Swing",
                    "R Shoulder Lift",
                    "R Elbow",
                    "R Wrist",
                    "Club Wrist",
                ],
            },
            {
                "name": "advanced_biomech",
                "xml": ADVANCED_BIOMECHANICAL_GOLF_SWING_XML,
                "actuators": [
                    "L Ankle Plantar",
                    "L Ankle Invert",
                    "L Knee",
                    "R Ankle Plantar",
                    "R Ankle Invert",
                    "R Knee",
                    "Spine Lateral",
                    "Spine Sagittal",
                    "Spine Rotation",
                    "L Scap Elev",
                    "L Scap Prot",
                    "L Shldr Flex",
                    "L Shldr Abd",
                    "L Shldr Rot",
                    "L Elbow",
                    "L Wrist Flex",
                    "L Wrist Dev",
                    "R Scap Elev",
                    "R Scap Prot",
                    "R Shldr Flex",
                    "R Shldr Abd",
                    "R Shldr Rot",
                    "R Elbow",
                    "R Wrist Flex",
                    "R Wrist Dev",
                    "Shaft Upper",
                    "Shaft Middle",
                    "Shaft Tip",
                ],
            },
            {
                "name": "myoupperbody",
                "xml_path": MYOUPPERBODY_PATH,
                "actuators": [
                    "R Shoulder Flex",
                    "R Shoulder Add",
                    "R Shoulder Rot",
                    "R Elbow",
                    "R Forearm",
                    "R Wrist Flex",
                    "R Wrist Dev",
                    "L Shoulder Flex",
                    "L Shoulder Add",
                    "L Shoulder Rot",
                    "L Elbow",
                    "L Forearm",
                    "L Wrist Flex",
                    "L Wrist Dev",
                    "R Erector Spinae",
                    "L Erector Spinae",
                    "R Int Oblique",
                    "L Int Oblique",
                    "R Ext Oblique",
                    "L Ext Oblique",
                ],
            },
            {
                "name": "myobody",
                "xml_path": MYOBODY_PATH,
                "actuators": [
                    f"Muscle {i + 1}" for i in range(290)
                ],  # 290 muscles - simplified names for UI
            },
            {
                "name": "myoarm_simple",
                "xml_path": MYOARM_SIMPLE_PATH,
                "actuators": [
                    "R Shoulder Flex",
                    "R Shoulder Add",
                    "R Shoulder Rot",
                    "R Elbow",
                    "R Forearm",
                    "R Wrist Flex",
                    "R Wrist Dev",
                    "L Shoulder Flex",
                    "L Shoulder Add",
                    "L Shoulder Rot",
                    "L Elbow",
                    "L Forearm",
                    "L Wrist Flex",
                    "L Wrist Dev",
                ],
            },
        ]

        # Add linkage mechanisms to model catalog
        for mech_name, mech_config in LINKAGE_CATALOG.items():
            self.model_configs.append(
                {
                    "name": mech_name.lower()
                    .replace(" ", "_")
                    .replace(":", "")
                    .replace("(", "")
                    .replace(")", ""),
                    "xml": mech_config["xml"],
                    "actuators": mech_config["actuators"],
                    "category": mech_config.get("category", "Mechanisms"),
                    "description": mech_config.get("description", ""),
                },
            )

        # Create central widget with splitter
        central = QtWidgets.QWidget(self)
        self.setCentralWidget(central)
        main_layout = QtWidgets.QHBoxLayout(central)

        # Main horizontal splitter: simulation | controls/analysis
        main_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)

        # Left: Simulation view
        self.sim_widget = MuJoCoSimWidget(width=900, height=700, fps=60)
        main_splitter.addWidget(self.sim_widget)

        # Right: Tabbed interface for controls and analysis
        self.tab_widget = QtWidgets.QTabWidget()
        self.tab_widget.setMinimumWidth(400)
        main_splitter.addWidget(self.tab_widget)

        # Set splitter proportions (70% simulation, 30% controls)
        main_splitter.setSizes([1100, 500])

        main_layout.addWidget(main_splitter)

        # Create tabs
        self._create_control_tab()
        self._create_visualization_tab()
        self._create_analysis_tab()
        self._create_plotting_tab()
        self._create_manipulation_tab()

        # Storage for actuator controls
        self.actuator_sliders: list[QtWidgets.QSlider] = []
        self.actuator_labels: list[QtWidgets.QLabel] = []
        self.actuator_groups: list[QtWidgets.QGroupBox] = []

        # Advanced control widgets
        self.actuator_control_types: list[
            QtWidgets.QComboBox
        ] = []  # ComboBoxes for control type
        self.actuator_constant_inputs: list[
            QtWidgets.QDoubleSpinBox
        ] = []  # SpinBoxes for constant values
        self.actuator_polynomial_coeffs: list[
            list[QtWidgets.QDoubleSpinBox]
        ] = []  # Lists of SpinBoxes for polynomial coefficients
        self.actuator_damping_inputs: list[
            QtWidgets.QDoubleSpinBox
        ] = []  # SpinBoxes for damping
        self.actuator_control_widgets: list[
            QtWidgets.QWidget
        ] = []  # Store all control widgets per actuator
        self.simplified_actuator_mode = False
        self.actuator_filter_input: QtWidgets.QLineEdit | None = None
        self._simplified_notice: QtWidgets.QLabel | None = None

        # Current plot canvas
        self.current_plot_canvas: MplCanvas | None = None

        # Load default model
        self.load_current_model()
        self.sim_widget.reset_state()

        # Apply professional styling
        self._apply_styling()

        # Create status bar
        self._create_status_bar()

        # Start status bar update timer
        self.status_timer = QtCore.QTimer(self)
        self.status_timer.timeout.connect(self._update_status_bar)
        self.status_timer.start(200)  # Update every 200ms

    def _create_control_tab(self) -> None:
        """Create the simulation controls tab."""
        control_widget = QtWidgets.QWidget()
        control_layout = QtWidgets.QVBoxLayout(control_widget)
        control_layout.setContentsMargins(8, 8, 8, 8)

        # Quick Start Help Panel (collapsible)
        self._create_help_panel(control_layout)

        # Quick Camera Access Buttons
        self._create_quick_camera_buttons(control_layout)

        # Model selector with enhanced descriptions
        model_group = QtWidgets.QGroupBox("Physics Models & Mechanisms")
        model_layout = QtWidgets.QVBoxLayout(model_group)
        self.model_combo = QtWidgets.QComboBox()
        tooltip_text = (
            "Select a physics model to simulate.\n"
            "DOF = Degrees of Freedom (number of independent joints)\n"
            "Higher DOF = more complex/realistic model"
        )
        self.model_combo.setToolTip(tooltip_text)

        # Add golf swing models with descriptive tooltips stored for later
        self.model_descriptions = {}

        # Golf models
        self.model_combo.addItem(
            "Golf: Chaotic Pendulum (2 DOF) - Forced oscillation demo",
        )
        self.model_descriptions[0] = (
            "Simple driven pendulum showing chaotic behavior. Great for learning "
            "control basics."
        )

        self.model_combo.addItem("Golf: Double Pendulum (2 DOF) - Shoulder + Wrist")
        self.model_descriptions[1] = (
            "Basic golf swing with shoulder and wrist joints. Simplest realistic "
            "swing model."
        )

        self.model_combo.addItem(
            "Golf: Triple Pendulum (3 DOF) - Shoulder + Elbow + Wrist",
        )
        self.model_descriptions[2] = (
            "Adds elbow joint for more realistic arm mechanics."
        )

        self.model_combo.addItem("Golf: Upper Body (10 DOF) - Spine + Both Arms + Club")
        self.model_descriptions[3] = (
            "Upper body model with spine rotation and both arms. Good balance of "
            "complexity."
        )

        self.model_combo.addItem("Golf: Full Body (15 DOF) - Legs + Torso + Arms")
        self.model_descriptions[4] = (
            "Full body model including leg drive and weight transfer."
        )

        self.model_combo.addItem(
            "Golf: Advanced Biomech (28 DOF) - Full Biomechanical Model",
        )
        self.model_descriptions[5] = (
            "Most detailed model with scapulae, 3-DOF shoulders, and flexible shaft."
        )

        # Musculoskeletal models
        self.model_combo.addItem("Musculoskeletal: Upper Body (19 DOF, 20 muscles)")
        self.model_descriptions[6] = (
            "Muscle-actuated upper body. Each muscle can be controlled independently."
        )

        self.model_combo.addItem("Musculoskeletal: Full Body (52 DOF, 290 muscles)")
        self.model_descriptions[7] = (
            "Complete musculoskeletal model. Very complex - for advanced users."
        )

        self.model_combo.addItem("Musculoskeletal: Bilateral Arms (14 DOF)")
        self.model_descriptions[8] = (
            "Both arms with muscle actuation. Good for arm mechanics study."
        )

        # Add linkage mechanisms from catalog
        idx = 9
        for mech_name in LINKAGE_CATALOG:
            self.model_combo.addItem(f"Mechanism: {mech_name}")
            mech_config = LINKAGE_CATALOG[mech_name]
            self.model_descriptions[idx] = mech_config.get(
                "description",
                "Mechanical linkage system",
            )
            idx += 1

        self.model_combo.currentIndexChanged.connect(self.on_model_changed)
        self.model_combo.currentIndexChanged.connect(self._update_model_description)
        model_layout.addWidget(self.model_combo)

        # Model description label
        self.model_description_label = QtWidgets.QLabel()
        self.model_description_label.setWordWrap(True)
        self.model_description_label.setStyleSheet(
            """
            QLabel {
                color: #666;
                font-style: italic;
                padding: 5px;
                background-color: #f5f5f5;
                border-radius: 3px;
            }
        """,
        )
        model_layout.addWidget(self.model_description_label)
        self._update_model_description(0)  # Set initial description

        control_layout.addWidget(model_group)

        # Actuator filter
        filter_layout = QtWidgets.QHBoxLayout()
        filter_layout.addWidget(QtWidgets.QLabel("Filter actuators:"))
        self.actuator_filter_input = QtWidgets.QLineEdit()
        self.actuator_filter_input.setPlaceholderText("Type actuator or group name…")
        self.actuator_filter_input.textChanged.connect(self.on_actuator_filter_changed)
        filter_layout.addWidget(self.actuator_filter_input)
        control_layout.addLayout(filter_layout)

        # Simulation control
        buttons_group = QtWidgets.QGroupBox("Simulation Control")
        buttons_layout = QtWidgets.QGridLayout(buttons_group)

        self.play_pause_btn = QtWidgets.QPushButton("Pause")
        self.play_pause_btn.setCheckable(True)
        self.play_pause_btn.toggled.connect(self.on_play_pause_toggled)
        self.play_pause_btn.setToolTip("Pause/Resume simulation (Shortcut: Space)")

        self.reset_btn = QtWidgets.QPushButton("Reset")
        self.reset_btn.clicked.connect(self.on_reset_clicked)
        self.reset_btn.setToolTip("Reset simulation to initial state (Shortcut: R)")

        self.record_btn = QtWidgets.QPushButton("Start Recording")
        self.record_btn.setCheckable(True)
        self.record_btn.toggled.connect(self.on_record_toggled)
        self.record_btn.setToolTip("Record simulation data for analysis and export")
        self.record_btn.setStyleSheet(
            """
            QPushButton:checked {
                background-color: #d62728;
                color: white;
                font-weight: bold;
            }
        """,
        )

        buttons_layout.addWidget(self.play_pause_btn, 0, 0)
        buttons_layout.addWidget(self.reset_btn, 0, 1)
        buttons_layout.addWidget(self.record_btn, 1, 0, 1, 2)
        control_layout.addWidget(buttons_group)

        # Recording info
        self.recording_label = QtWidgets.QLabel("Not recording")
        self.recording_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.recording_label.setStyleSheet("font-weight: bold; padding: 5px;")
        control_layout.addWidget(self.recording_label)

        # Actuator Controls Header with collapse/expand buttons
        actuator_header = QtWidgets.QGroupBox("Actuator Controls")
        actuator_header_layout = QtWidgets.QVBoxLayout(actuator_header)

        # Collapse/Expand all buttons
        actuator_btn_layout = QtWidgets.QHBoxLayout()
        self.expand_all_btn = QtWidgets.QPushButton("Expand All")
        self.expand_all_btn.setToolTip("Expand all actuator control groups")
        self.expand_all_btn.clicked.connect(self._expand_all_actuator_groups)
        self.collapse_all_btn = QtWidgets.QPushButton("Collapse All")
        self.collapse_all_btn.setToolTip("Collapse all actuator control groups")
        self.collapse_all_btn.clicked.connect(self._collapse_all_actuator_groups)

        # Actuator summary label
        self.actuator_summary_label = QtWidgets.QLabel("0 actuators")
        self.actuator_summary_label.setStyleSheet("color: #666; font-weight: bold;")

        actuator_btn_layout.addWidget(self.actuator_summary_label)
        actuator_btn_layout.addStretch()
        actuator_btn_layout.addWidget(self.expand_all_btn)
        actuator_btn_layout.addWidget(self.collapse_all_btn)
        actuator_header_layout.addLayout(actuator_btn_layout)

        # Scrollable area for actuator controls
        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff,
        )

        self.actuator_container = QtWidgets.QWidget()
        self.actuator_layout = QtWidgets.QVBoxLayout(self.actuator_container)
        scroll_area.setWidget(self.actuator_container)

        actuator_header_layout.addWidget(scroll_area, stretch=1)
        control_layout.addWidget(actuator_header, stretch=1)

        self.tab_widget.addTab(control_widget, "Controls")

    def _create_help_panel(self, parent_layout: QtWidgets.QVBoxLayout) -> None:
        """Create a collapsible Quick Start help panel."""
        help_group = QtWidgets.QGroupBox("Quick Start Guide")
        help_group.setCheckable(True)
        help_group.setChecked(False)  # Collapsed by default
        help_layout = QtWidgets.QVBoxLayout(help_group)
        help_layout.setContentsMargins(8, 8, 8, 8)

        # Help content
        help_text = QtWidgets.QLabel(
            "<b>Mouse Controls:</b><br>"
            "- <b>Left Drag:</b> Rotate camera / Select & drag body<br>"
            "- <b>Right Drag / Ctrl+Left:</b> Rotate camera<br>"
            "- <b>Middle Drag / Shift+Left:</b> Pan camera<br>"
            "- <b>Scroll Wheel:</b> Zoom in/out<br><br>"
            "<b>Keyboard Shortcuts:</b><br>"
            "- <b>1-5:</b> Camera presets (Side, Front, Top, Follow, Down-the-line)<br>"
            "- <b>Space:</b> Play/Pause simulation<br>"
            "- <b>R:</b> Reset simulation<br>"
            "- <b>H:</b> Toggle this help panel<br><br>"
            "<b>Control Types:</b><br>"
            "- <b>Constant:</b> Fixed torque value<br>"
            "- <b>Polynomial:</b> Time-varying torque (c0 + c1*t + ... + c6*t^6)<br>"
            "- <b>Sine Wave:</b> Oscillating torque<br>"
            "- <b>Step:</b> Jump to value at specified time",
        )
        help_text.setWordWrap(True)
        help_text.setStyleSheet(
            """
            QLabel {
                background-color: #f0f8ff;
                padding: 10px;
                border-radius: 5px;
                font-size: 10pt;
            }
        """,
        )
        help_layout.addWidget(help_text)

        # Connect toggle to show/hide content
        help_group.toggled.connect(lambda checked: help_text.setVisible(checked))
        help_text.setVisible(False)  # Initially hidden

        parent_layout.addWidget(help_group)

    def _create_quick_camera_buttons(
        self,
        parent_layout: QtWidgets.QVBoxLayout,
    ) -> None:
        """Create quick access camera preset buttons."""
        camera_group = QtWidgets.QGroupBox("Quick Camera Views")
        camera_layout = QtWidgets.QHBoxLayout(camera_group)
        camera_layout.setContentsMargins(5, 5, 5, 5)

        # Camera preset buttons with tooltips
        camera_presets = [
            ("Side", "side", "Side view (Key: 1) - Standard golf swing view"),
            ("Front", "front", "Front view (Key: 2) - Face-on view"),
            ("Top", "top", "Top view (Key: 3) - Overhead bird's eye view"),
            ("Follow", "follow", "Follow view (Key: 4) - Dynamic tracking view"),
            (
                "DTL",
                "down-the-line",
                "Down-the-line view (Key: 5) - Behind golfer view",
            ),
        ]

        self.quick_camera_buttons = {}
        for label, preset_name, tooltip in camera_presets:
            btn = QtWidgets.QPushButton(label)
            btn.setToolTip(tooltip)
            btn.setMaximumWidth(60)
            btn.clicked.connect(
                lambda *, name=preset_name: self._on_quick_camera_clicked(name),
            )
            camera_layout.addWidget(btn)
            self.quick_camera_buttons[preset_name] = btn

        # Reset camera button
        reset_btn = QtWidgets.QPushButton("Reset")
        reset_btn.setToolTip("Reset camera to default position")
        reset_btn.setMaximumWidth(50)
        reset_btn.clicked.connect(self.on_reset_camera)
        reset_btn.setStyleSheet("background-color: #ff7f0e;")
        camera_layout.addWidget(reset_btn)

        parent_layout.addWidget(camera_group)

    def _on_quick_camera_clicked(self, preset_name: str) -> None:
        """Handle quick camera button click."""
        self.sim_widget.set_camera(preset_name)
        self._update_camera_sliders()
        # Update preset combo in visualization tab
        index = self.camera_combo.findText(preset_name)
        if index >= 0:
            self.camera_combo.blockSignals(block=True)
            self.camera_combo.setCurrentIndex(index)
            self.camera_combo.blockSignals(block=False)

    def _update_model_description(self, index: int) -> None:
        """Update the model description label based on selected model."""
        if hasattr(self, "model_description_label") and hasattr(
            self,
            "model_descriptions",
        ):
            description = self.model_descriptions.get(
                index,
                "Select a model to see its description.",
            )
            self.model_description_label.setText(description)

    def _expand_all_actuator_groups(self) -> None:
        """Expand all actuator control groups."""
        for group in self.actuator_groups:
            if isinstance(group, QtWidgets.QGroupBox) and group.isCheckable():
                group.setChecked(True)

    def _collapse_all_actuator_groups(self) -> None:
        """Collapse all actuator control groups."""
        for group in self.actuator_groups:
            if isinstance(group, QtWidgets.QGroupBox) and group.isCheckable():
                group.setChecked(False)

    def _update_actuator_summary(self) -> None:
        """Update the actuator count summary label."""
        if hasattr(self, "actuator_summary_label"):
            count = len(self.actuator_sliders)
            self.actuator_summary_label.setText(f"{count} actuators")

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:  # noqa: N802
        """Handle keyboard shortcuts."""
        key = event.key()

        # Camera preset shortcuts (1-5)
        camera_shortcuts = {
            QtCore.Qt.Key.Key_1: "side",
            QtCore.Qt.Key.Key_2: "front",
            QtCore.Qt.Key.Key_3: "top",
            QtCore.Qt.Key.Key_4: "follow",
            QtCore.Qt.Key.Key_5: "down-the-line",
        }

        if key in camera_shortcuts:
            preset = camera_shortcuts[key]
            self._on_quick_camera_clicked(preset)
            return

        # Space key: Play/Pause
        if key == QtCore.Qt.Key.Key_Space:
            self.play_pause_btn.toggle()
            return

        # R key: Reset
        if key == QtCore.Qt.Key.Key_R:
            self.on_reset_clicked()
            return

        # H key: Toggle help panel
        if key == QtCore.Qt.Key.Key_H:
            # Find the help group and toggle it
            for i in range(self.tab_widget.widget(0).layout().count()):
                widget = self.tab_widget.widget(0).layout().itemAt(i).widget()
                if (
                    isinstance(widget, QtWidgets.QGroupBox)
                    and widget.title() == "Quick Start Guide"
                ):
                    widget.setChecked(not widget.isChecked())
                    break
            return

        super().keyPressEvent(event)

    def _create_status_bar(self) -> None:
        """Create a status bar showing simulation information."""
        status_bar = self.statusBar()
        status_bar.setStyleSheet(
            """
            QStatusBar {
                background-color: #2c3e50;
                color: white;
                font-weight: bold;
            }
            QStatusBar::item {
                border: none;
            }
        """,
        )

        # Model info label
        self.status_model_label = QtWidgets.QLabel("Model: --")
        self.status_model_label.setStyleSheet("color: #3498db; padding: 0 10px;")
        status_bar.addWidget(self.status_model_label)

        # Separator
        sep1 = QtWidgets.QLabel("|")
        sep1.setStyleSheet("color: #7f8c8d;")
        status_bar.addWidget(sep1)

        # Time label
        self.status_time_label = QtWidgets.QLabel("Time: 0.00s")
        self.status_time_label.setStyleSheet("color: #2ecc71; padding: 0 10px;")
        status_bar.addWidget(self.status_time_label)

        # Separator
        sep2 = QtWidgets.QLabel("|")
        sep2.setStyleSheet("color: #7f8c8d;")
        status_bar.addWidget(sep2)

        # Camera info label
        self.status_camera_label = QtWidgets.QLabel("Camera: side")
        self.status_camera_label.setStyleSheet("color: #9b59b6; padding: 0 10px;")
        status_bar.addWidget(self.status_camera_label)

        # Separator
        sep3 = QtWidgets.QLabel("|")
        sep3.setStyleSheet("color: #7f8c8d;")
        status_bar.addWidget(sep3)

        # Simulation state label
        self.status_state_label = QtWidgets.QLabel("Running")
        self.status_state_label.setStyleSheet("color: #2ecc71; padding: 0 10px;")
        status_bar.addWidget(self.status_state_label)

        # Permanent widget for recording status (right side)
        self.status_recording_label = QtWidgets.QLabel("")
        self.status_recording_label.setStyleSheet("color: #e74c3c; padding: 0 10px;")
        status_bar.addPermanentWidget(self.status_recording_label)

    def _update_status_bar(self) -> None:
        """Update status bar with current simulation info."""
        if self.sim_widget.model is None:
            return

        # Update model info
        config_idx = self.model_combo.currentIndex()
        if config_idx < len(self.model_configs):
            model_name = self.model_configs[config_idx]["name"]
            num_actuators = self.sim_widget.model.nu
            self.status_model_label.setText(
                f"Model: {model_name} ({num_actuators} actuators)",
            )

        # Update time
        if self.sim_widget.data is not None:
            time = self.sim_widget.data.time
            self.status_time_label.setText(f"Time: {time:.2f}s")

        # Update camera info
        if self.sim_widget.camera is not None:
            az = self.sim_widget.camera.azimuth
            el = self.sim_widget.camera.elevation
            dist = self.sim_widget.camera.distance
            self.status_camera_label.setText(
                f"Camera: Az={az:.0f}° El={el:.0f}° D={dist:.1f}",
            )

        # Update simulation state
        if self.sim_widget.running:
            self.status_state_label.setText("Running")
            self.status_state_label.setStyleSheet("color: #2ecc71; padding: 0 10px;")
        else:
            self.status_state_label.setText("Paused")
            self.status_state_label.setStyleSheet("color: #f39c12; padding: 0 10px;")

        # Update recording status
        recorder = self.sim_widget.get_recorder()
        if recorder.is_recording:
            frames = recorder.get_num_frames()
            duration = recorder.get_duration()
            self.status_recording_label.setText(
                f"RECORDING: {frames} frames ({duration:.1f}s)",
            )
            self.status_recording_label.setStyleSheet(
                "color: #e74c3c; font-weight: bold; padding: 0 10px;",
            )
        else:
            frames = recorder.get_num_frames()
            if frames > 0:
                self.status_recording_label.setText(f"Recorded: {frames} frames")
                self.status_recording_label.setStyleSheet(
                    "color: #f39c12; padding: 0 10px;",
                )
            else:
                self.status_recording_label.setText("")

    def _create_visualization_tab(self) -> None:
        """Create the visualization settings tab."""
        viz_widget = QtWidgets.QWidget()
        viz_layout = QtWidgets.QVBoxLayout(viz_widget)
        viz_layout.setContentsMargins(8, 8, 8, 8)

        # Camera controls
        camera_group = QtWidgets.QGroupBox("Camera View")
        camera_layout = QtWidgets.QVBoxLayout(camera_group)

        # Preset camera views
        preset_layout = QtWidgets.QHBoxLayout()
        preset_layout.addWidget(QtWidgets.QLabel("Preset:"))
        self.camera_combo = QtWidgets.QComboBox()
        self.camera_combo.addItems(["side", "front", "top", "follow", "down-the-line"])
        self.camera_combo.currentTextChanged.connect(self.on_camera_changed)
        preset_layout.addWidget(self.camera_combo)
        camera_layout.addLayout(preset_layout)

        # Reset camera button
        reset_cam_btn = QtWidgets.QPushButton("Reset Camera")
        reset_cam_btn.clicked.connect(self.on_reset_camera)
        camera_layout.addWidget(reset_cam_btn)

        # Advanced camera controls
        advanced_cam_group = QtWidgets.QGroupBox("Advanced Camera Controls")
        advanced_cam_layout = QtWidgets.QFormLayout(advanced_cam_group)

        # Azimuth (rotation around vertical axis)
        self.azimuth_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.azimuth_slider.setMinimum(0)
        self.azimuth_slider.setMaximum(360)
        self.azimuth_slider.setValue(90)
        self.azimuth_slider.valueChanged.connect(self.on_azimuth_changed)
        self.azimuth_label = QtWidgets.QLabel("90°")
        advanced_cam_layout.addRow("Azimuth:", self.azimuth_slider)
        advanced_cam_layout.addRow("", self.azimuth_label)

        # Elevation (up/down angle)
        self.elevation_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.elevation_slider.setMinimum(-90)
        self.elevation_slider.setMaximum(90)
        self.elevation_slider.setValue(-20)
        self.elevation_slider.valueChanged.connect(self.on_elevation_changed)
        self.elevation_label = QtWidgets.QLabel("-20°")
        advanced_cam_layout.addRow("Elevation:", self.elevation_slider)
        advanced_cam_layout.addRow("", self.elevation_label)

        # Distance slider for zoom control
        self.distance_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.distance_slider.setMinimum(1)
        self.distance_slider.setMaximum(500)
        self.distance_slider.setValue(30)
        self.distance_slider.valueChanged.connect(self.on_distance_changed)
        self.distance_label = QtWidgets.QLabel("3.0")
        advanced_cam_layout.addRow("Distance:", self.distance_slider)
        advanced_cam_layout.addRow("", self.distance_label)

        # Lookat position (X, Y, Z)
        lookat_layout = QtWidgets.QHBoxLayout()
        self.lookat_x_spin = QtWidgets.QDoubleSpinBox()
        self.lookat_x_spin.setRange(-10.0, 10.0)
        self.lookat_x_spin.setSingleStep(0.1)
        self.lookat_x_spin.setValue(0.0)
        self.lookat_x_spin.valueChanged.connect(self.on_lookat_changed)
        lookat_layout.addWidget(QtWidgets.QLabel("X:"))
        lookat_layout.addWidget(self.lookat_x_spin)

        self.lookat_y_spin = QtWidgets.QDoubleSpinBox()
        self.lookat_y_spin.setRange(-10.0, 10.0)
        self.lookat_y_spin.setSingleStep(0.1)
        self.lookat_y_spin.setValue(0.0)
        self.lookat_y_spin.valueChanged.connect(self.on_lookat_changed)
        lookat_layout.addWidget(QtWidgets.QLabel("Y:"))
        lookat_layout.addWidget(self.lookat_y_spin)

        self.lookat_z_spin = QtWidgets.QDoubleSpinBox()
        self.lookat_z_spin.setRange(-10.0, 10.0)
        self.lookat_z_spin.setSingleStep(0.1)
        self.lookat_z_spin.setValue(1.0)
        self.lookat_z_spin.valueChanged.connect(self.on_lookat_changed)
        lookat_layout.addWidget(QtWidgets.QLabel("Z:"))
        lookat_layout.addWidget(self.lookat_z_spin)

        advanced_cam_layout.addRow("Lookat:", lookat_layout)

        # Mouse controls info
        mouse_info = QtWidgets.QLabel(
            "Mouse Controls:\n"
            "• Left Drag: Rotate camera\n"
            "• Right/Ctrl+Left: Rotate camera\n"
            "• Middle/Shift+Left: Pan camera\n"
            "• Wheel: Zoom",
        )
        mouse_info.setWordWrap(True)
        mouse_info.setStyleSheet("color: #666; font-size: 9pt;")
        advanced_cam_layout.addRow("", mouse_info)

        camera_layout.addWidget(advanced_cam_group)
        viz_layout.addWidget(camera_group)

        # Background color controls
        bg_group = QtWidgets.QGroupBox("Background Color")
        bg_layout = QtWidgets.QVBoxLayout(bg_group)

        # Sky color
        sky_layout = QtWidgets.QHBoxLayout()
        sky_layout.addWidget(QtWidgets.QLabel("Sky Color:"))
        self.sky_color_btn = QtWidgets.QPushButton()
        self.sky_color_btn.setFixedSize(60, 30)
        self.sky_color_btn.setStyleSheet("background-color: rgb(51, 77, 102);")
        self.sky_color_btn.clicked.connect(self.on_sky_color_clicked)
        sky_layout.addWidget(self.sky_color_btn)
        sky_layout.addStretch()
        bg_layout.addLayout(sky_layout)

        # Ground color
        ground_layout = QtWidgets.QHBoxLayout()
        ground_layout.addWidget(QtWidgets.QLabel("Ground Color:"))
        self.ground_color_btn = QtWidgets.QPushButton()
        self.ground_color_btn.setFixedSize(60, 30)
        self.ground_color_btn.setStyleSheet("background-color: rgb(51, 51, 51);")
        self.ground_color_btn.clicked.connect(self.on_ground_color_clicked)
        ground_layout.addWidget(self.ground_color_btn)
        ground_layout.addStretch()
        bg_layout.addLayout(ground_layout)

        # Reset to defaults button
        reset_bg_btn = QtWidgets.QPushButton("Reset to Defaults")
        reset_bg_btn.clicked.connect(self.on_reset_background)
        bg_layout.addWidget(reset_bg_btn)

        viz_layout.addWidget(bg_group)

        # Force/Torque visualization
        force_group = QtWidgets.QGroupBox("Force & Torque Visualization")
        force_layout = QtWidgets.QVBoxLayout(force_group)

        # Torque vectors
        self.show_torques_cb = QtWidgets.QCheckBox("Show Joint Torque Vectors")
        self.show_torques_cb.stateChanged.connect(self.on_show_torques_changed)
        force_layout.addWidget(self.show_torques_cb)

        torque_scale_layout = QtWidgets.QFormLayout()
        self.torque_scale_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.torque_scale_slider.setMinimum(1)
        self.torque_scale_slider.setMaximum(100)
        self.torque_scale_slider.setValue(10)
        self.torque_scale_slider.valueChanged.connect(self.on_torque_scale_changed)
        self.torque_scale_label = QtWidgets.QLabel("1.0%")
        torque_scale_layout.addRow("Torque Scale:", self.torque_scale_slider)
        torque_scale_layout.addRow("", self.torque_scale_label)
        force_layout.addLayout(torque_scale_layout)

        # Force vectors
        self.show_forces_cb = QtWidgets.QCheckBox("Show Constraint Forces")
        self.show_forces_cb.stateChanged.connect(self.on_show_forces_changed)
        force_layout.addWidget(self.show_forces_cb)

        force_scale_layout = QtWidgets.QFormLayout()
        self.force_scale_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.force_scale_slider.setMinimum(1)
        self.force_scale_slider.setMaximum(100)
        self.force_scale_slider.setValue(10)
        self.force_scale_slider.valueChanged.connect(self.on_force_scale_changed)
        self.force_scale_label = QtWidgets.QLabel("10%")
        force_scale_layout.addRow("Force Scale:", self.force_scale_slider)
        force_scale_layout.addRow("", self.force_scale_label)
        force_layout.addLayout(force_scale_layout)

        # Contact forces
        self.show_contacts_cb = QtWidgets.QCheckBox("Show Contact Forces")
        self.show_contacts_cb.stateChanged.connect(self.on_show_contacts_changed)
        force_layout.addWidget(self.show_contacts_cb)

        viz_layout.addWidget(force_group)

        viz_layout.addStretch(1)

        self.tab_widget.addTab(viz_widget, "Visualization")

    def _create_analysis_tab(self) -> None:
        """Create the biomechanical analysis tab."""
        analysis_widget = QtWidgets.QWidget()
        analysis_layout = QtWidgets.QVBoxLayout(analysis_widget)
        analysis_layout.setContentsMargins(8, 8, 8, 8)

        # Real-time metrics
        metrics_group = QtWidgets.QGroupBox("Real-Time Metrics")
        metrics_layout = QtWidgets.QFormLayout(metrics_group)

        self.club_speed_label = QtWidgets.QLabel("--")
        self.total_energy_label = QtWidgets.QLabel("--")
        self.recording_time_label = QtWidgets.QLabel("--")
        self.num_frames_label = QtWidgets.QLabel("--")

        metrics_layout.addRow("Club Head Speed:", self.club_speed_label)
        metrics_layout.addRow("Total Energy:", self.total_energy_label)
        metrics_layout.addRow("Recording Time:", self.recording_time_label)
        metrics_layout.addRow("Frames Recorded:", self.num_frames_label)

        analysis_layout.addWidget(metrics_group)

        # Data export
        export_group = QtWidgets.QGroupBox("Data Export")
        export_layout = QtWidgets.QVBoxLayout(export_group)

        self.export_csv_btn = QtWidgets.QPushButton("Export to CSV")
        self.export_csv_btn.clicked.connect(self.on_export_csv)
        export_layout.addWidget(self.export_csv_btn)

        self.export_json_btn = QtWidgets.QPushButton("Export to JSON")
        self.export_json_btn.clicked.connect(self.on_export_json)
        export_layout.addWidget(self.export_json_btn)

        analysis_layout.addWidget(export_group)

        # Update metrics timer
        self.metrics_timer = QtCore.QTimer(self)
        self.metrics_timer.timeout.connect(self.update_metrics)
        self.metrics_timer.start(100)  # Update every 100ms

        analysis_layout.addStretch(1)

        self.tab_widget.addTab(analysis_widget, "Analysis")

    def _create_plotting_tab(self) -> None:
        """Create the advanced plotting tab."""
        plotting_widget = QtWidgets.QWidget()
        plotting_layout = QtWidgets.QVBoxLayout(plotting_widget)
        plotting_layout.setContentsMargins(8, 8, 8, 8)

        # Plot selection
        plot_group = QtWidgets.QGroupBox("Plot Type")
        plot_layout = QtWidgets.QVBoxLayout(plot_group)

        self.plot_combo = QtWidgets.QComboBox()
        self.plot_combo.addItems(
            [
                "Summary Dashboard",
                "Joint Angles",
                "Joint Velocities",
                "Joint Torques",
                "Actuator Powers",
                "Energy Analysis",
                "Club Head Speed",
                "Club Head Trajectory (3D)",
                "Phase Diagram",
                "Torque Comparison",
            ],
        )
        plot_layout.addWidget(self.plot_combo)

        # Joint selection for phase diagrams
        self.joint_select_layout = QtWidgets.QFormLayout()
        self.joint_select_combo = QtWidgets.QComboBox()
        self.joint_select_layout.addRow("Joint:", self.joint_select_combo)
        self.joint_select_widget = QtWidgets.QWidget()
        self.joint_select_widget.setLayout(self.joint_select_layout)
        self.joint_select_widget.setVisible(False)
        plot_layout.addWidget(self.joint_select_widget)

        self.plot_combo.currentTextChanged.connect(self.on_plot_type_changed)

        self.generate_plot_btn = QtWidgets.QPushButton("Generate Plot")
        self.generate_plot_btn.clicked.connect(self.on_generate_plot)
        self.generate_plot_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #2ca02c;
                color: white;
                font-weight: bold;
                padding: 8px;
            }
            QPushButton:hover {
                background-color: #238c23;
            }
        """,
        )
        plot_layout.addWidget(self.generate_plot_btn)

        plotting_layout.addWidget(plot_group)

        # Plot canvas container
        self.plot_container = QtWidgets.QWidget()
        self.plot_container_layout = QtWidgets.QVBoxLayout(self.plot_container)
        self.plot_container_layout.setContentsMargins(0, 0, 0, 0)

        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(self.plot_container)
        plotting_layout.addWidget(scroll_area, stretch=1)

        self.tab_widget.addTab(plotting_widget, "Plotting")

    def _create_manipulation_tab(self) -> None:
        """Create the interactive manipulation tab."""

        manip_widget = QtWidgets.QWidget()
        manip_layout = QtWidgets.QVBoxLayout(manip_widget)
        manip_layout.setContentsMargins(8, 8, 8, 8)

        # Drag mode controls
        drag_group = QtWidgets.QGroupBox("Drag Mode")
        drag_layout = QtWidgets.QVBoxLayout(drag_group)

        self.enable_drag_cb = QtWidgets.QCheckBox("Enable Drag Manipulation")
        self.enable_drag_cb.setChecked(True)
        self.enable_drag_cb.stateChanged.connect(self.on_drag_enabled_changed)
        drag_layout.addWidget(self.enable_drag_cb)

        self.maintain_orientation_cb = QtWidgets.QCheckBox(
            "Maintain Orientation While Dragging",
        )
        self.maintain_orientation_cb.stateChanged.connect(
            self.on_maintain_orientation_changed,
        )
        drag_layout.addWidget(self.maintain_orientation_cb)

        self.nullspace_posture_cb = QtWidgets.QCheckBox(
            "Use Nullspace Posture Optimization",
        )
        self.nullspace_posture_cb.setChecked(True)
        self.nullspace_posture_cb.stateChanged.connect(
            self.on_nullspace_posture_changed,
        )
        drag_layout.addWidget(self.nullspace_posture_cb)

        manip_layout.addWidget(drag_group)

        # Constraint controls
        constraint_group = QtWidgets.QGroupBox("Body Constraints")
        constraint_layout = QtWidgets.QVBoxLayout(constraint_group)

        # Body selection
        body_select_layout = QtWidgets.QHBoxLayout()
        body_select_layout.addWidget(QtWidgets.QLabel("Body:"))
        self.constraint_body_combo = QtWidgets.QComboBox()
        self.constraint_body_combo.setMinimumWidth(150)
        body_select_layout.addWidget(self.constraint_body_combo, stretch=1)
        constraint_layout.addLayout(body_select_layout)

        # Constraint type
        type_layout = QtWidgets.QHBoxLayout()
        type_layout.addWidget(QtWidgets.QLabel("Type:"))
        self.constraint_type_combo = QtWidgets.QComboBox()
        self.constraint_type_combo.addItems(["Fixed in Space", "Relative to Body"])
        type_layout.addWidget(self.constraint_type_combo, stretch=1)
        constraint_layout.addLayout(type_layout)

        # Reference body (for relative constraints)
        self.ref_body_layout = QtWidgets.QHBoxLayout()
        self.ref_body_layout.addWidget(QtWidgets.QLabel("Reference:"))
        self.ref_body_combo = QtWidgets.QComboBox()
        self.ref_body_combo.setMinimumWidth(150)
        self.ref_body_layout.addWidget(self.ref_body_combo, stretch=1)
        self.ref_body_widget = QtWidgets.QWidget()
        self.ref_body_widget.setLayout(self.ref_body_layout)
        self.ref_body_widget.setVisible(False)
        constraint_layout.addWidget(self.ref_body_widget)

        self.constraint_type_combo.currentIndexChanged.connect(
            lambda idx: self.ref_body_widget.setVisible(idx == 1),
        )

        # Constraint buttons
        constraint_btn_layout = QtWidgets.QHBoxLayout()
        self.add_constraint_btn = QtWidgets.QPushButton("Add Constraint")
        self.add_constraint_btn.clicked.connect(self.on_add_constraint)
        self.remove_constraint_btn = QtWidgets.QPushButton("Remove Constraint")
        self.remove_constraint_btn.clicked.connect(self.on_remove_constraint)
        constraint_btn_layout.addWidget(self.add_constraint_btn)
        constraint_btn_layout.addWidget(self.remove_constraint_btn)
        constraint_layout.addLayout(constraint_btn_layout)

        # Clear all constraints button
        self.clear_constraints_btn = QtWidgets.QPushButton("Clear All Constraints")
        self.clear_constraints_btn.clicked.connect(self.on_clear_constraints)
        self.clear_constraints_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #d62728;
            }
            QPushButton:hover {
                background-color: #a81f20;
            }
        """,
        )
        constraint_layout.addWidget(self.clear_constraints_btn)

        # Constrained bodies list
        constraint_layout.addWidget(QtWidgets.QLabel("Active Constraints:"))
        self.constraints_list = QtWidgets.QListWidget()
        self.constraints_list.setMaximumHeight(100)
        constraint_layout.addWidget(self.constraints_list)

        manip_layout.addWidget(constraint_group)

        # Pose library controls
        pose_group = QtWidgets.QGroupBox("Pose Library")
        pose_layout = QtWidgets.QVBoxLayout(pose_group)

        # Save pose
        save_layout = QtWidgets.QHBoxLayout()
        self.pose_name_input = QtWidgets.QLineEdit()
        self.pose_name_input.setPlaceholderText("Pose name...")
        save_layout.addWidget(self.pose_name_input)
        self.save_pose_btn = QtWidgets.QPushButton("Save Pose")
        self.save_pose_btn.clicked.connect(self.on_save_pose)
        save_layout.addWidget(self.save_pose_btn)
        pose_layout.addLayout(save_layout)

        # Pose list
        self.pose_list = QtWidgets.QListWidget()
        self.pose_list.setMaximumHeight(120)
        pose_layout.addWidget(self.pose_list)

        # Pose actions
        pose_btn_layout = QtWidgets.QGridLayout()
        self.load_pose_btn = QtWidgets.QPushButton("Load")
        self.load_pose_btn.clicked.connect(self.on_load_pose)
        self.delete_pose_btn = QtWidgets.QPushButton("Delete")
        self.delete_pose_btn.clicked.connect(self.on_delete_pose)
        self.export_poses_btn = QtWidgets.QPushButton("Export Library")
        self.export_poses_btn.clicked.connect(self.on_export_poses)
        self.import_poses_btn = QtWidgets.QPushButton("Import Library")
        self.import_poses_btn.clicked.connect(self.on_import_poses)

        pose_btn_layout.addWidget(self.load_pose_btn, 0, 0)
        pose_btn_layout.addWidget(self.delete_pose_btn, 0, 1)
        pose_btn_layout.addWidget(self.export_poses_btn, 1, 0)
        pose_btn_layout.addWidget(self.import_poses_btn, 1, 1)
        pose_layout.addLayout(pose_btn_layout)

        # Pose interpolation
        interp_group = QtWidgets.QGroupBox("Pose Interpolation")
        interp_layout = QtWidgets.QVBoxLayout(interp_group)

        interp_slider_layout = QtWidgets.QFormLayout()
        self.interp_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.interp_slider.setMinimum(0)
        self.interp_slider.setMaximum(100)
        self.interp_slider.setValue(0)
        self.interp_slider.valueChanged.connect(self.on_interpolate_poses)
        self.interp_label = QtWidgets.QLabel("0%")
        interp_slider_layout.addRow("Blend:", self.interp_slider)
        interp_slider_layout.addRow("", self.interp_label)
        interp_layout.addLayout(interp_slider_layout)

        interp_note = QtWidgets.QLabel("Select two poses in library to interpolate")
        interp_note.setStyleSheet("font-style: italic; font-size: 9pt;")
        interp_layout.addWidget(interp_note)

        pose_layout.addWidget(interp_group)

        manip_layout.addWidget(pose_group)

        # IK settings
        ik_group = QtWidgets.QGroupBox("IK Solver Settings (Advanced)")
        ik_layout = QtWidgets.QFormLayout(ik_group)

        self.ik_damping_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.ik_damping_slider.setMinimum(1)
        self.ik_damping_slider.setMaximum(100)
        self.ik_damping_slider.setValue(5)
        self.ik_damping_slider.valueChanged.connect(self.on_ik_damping_changed)
        self.ik_damping_label = QtWidgets.QLabel("0.05")
        ik_layout.addRow("Damping:", self.ik_damping_slider)
        ik_layout.addRow("", self.ik_damping_label)

        self.ik_step_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.ik_step_slider.setMinimum(1)
        self.ik_step_slider.setMaximum(100)
        self.ik_step_slider.setValue(30)
        self.ik_step_slider.valueChanged.connect(self.on_ik_step_changed)
        self.ik_step_label = QtWidgets.QLabel("0.30")
        ik_layout.addRow("Step Size:", self.ik_step_slider)
        ik_layout.addRow("", self.ik_step_label)

        manip_layout.addWidget(ik_group)

        # Instructions
        instructions = QtWidgets.QLabel(
            "<b>Quick Start:</b><br>"
            "• Click and drag any body part to move it<br>"
            "• Scroll wheel to zoom camera<br>"
            "• Add constraints to fix bodies in space<br>"
            "• Save poses for later use",
        )
        instructions.setWordWrap(True)
        instructions.setStyleSheet(
            "padding: 10px; background-color: #e8f4f8; border-radius: 5px;",
        )
        manip_layout.addWidget(instructions)

        manip_layout.addStretch(1)

        self.tab_widget.addTab(manip_widget, "Interactive Pose")

        # Update body lists when model changes
        self.update_body_lists()

    def _apply_styling(self) -> None:
        """Apply professional styling to the application."""
        self.setStyleSheet(
            """
            QMainWindow {
                background-color: #f0f0f0;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #cccccc;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QPushButton {
                padding: 5px 10px;
                border-radius: 3px;
                background-color: #1f77b4;
                color: white;
            }
            QPushButton:hover {
                background-color: #1a5f8f;
            }
            QPushButton:pressed {
                background-color: #144a6e;
            }
            QTabWidget::pane {
                border: 1px solid #cccccc;
                background-color: white;
            }
            QTabBar::tab {
                padding: 8px 16px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background-color: #1f77b4;
                color: white;
                font-weight: bold;
            }
        """,
        )

    # -------- Model management --------

    def load_current_model(self) -> None:
        """Load selected model and recreate actuator controls."""
        index = self.model_combo.currentIndex()
        config = self.model_configs[index]

        # Clear existing controls
        self._clear_actuator_controls()

        # Load the model - handle both XML strings and file paths
        if "xml_path" in config:
            # Load from external file (e.g., MyoSuite models)
            self.sim_widget.load_model_from_file(config["xml_path"])
        elif "xml" in config:
            # Load from XML string (traditional models)
            self.sim_widget.load_model_from_xml(config["xml"])
        else:
            error_msg = (
                f"Model config must have either 'xml' or 'xml_path': {config['name']}"
            )
            raise ValueError(error_msg)

        # Verify actuator count matches model
        if self.sim_widget.model is not None:
            model_actuator_count = self.sim_widget.model.nu
            config_actuator_count = len(config["actuators"])

            if model_actuator_count != config_actuator_count:
                logger.warning(
                    "Model has %d actuators but config specifies %d. "
                    "Using model's actual actuator count. Config names may not match.",
                    model_actuator_count,
                    config_actuator_count,
                )

                # If config has fewer names, pad with generic names
                if config_actuator_count < model_actuator_count:
                    for i in range(config_actuator_count, model_actuator_count):
                        config["actuators"].append(f"Actuator {i}")
                # If config has more names, truncate
                elif config_actuator_count > model_actuator_count:
                    config["actuators"] = config["actuators"][:model_actuator_count]

        # Create new actuator controls (now guaranteed to match model.nu)
        self._create_actuator_controls(config["actuators"])

        # Verify control system matches model
        if not self.sim_widget.verify_control_system():
            logger.error("Control system does not match model actuator count!")
            logger.error("Model has %d actuators", self.sim_widget.model.nu)
            if self.sim_widget.control_system:
                logger.error(
                    "Control system has %d actuators",
                    self.sim_widget.control_system.num_actuators,
                )
        else:
            logger.debug(
                "Control system verified: %d actuators",
                self.sim_widget.model.nu,
            )

        # Update joint selector for phase diagrams
        self.joint_select_combo.clear()
        for i, name in enumerate(config["actuators"]):
            self.joint_select_combo.addItem(f"{i}: {name}")

        # Update body lists for interactive manipulation
        self.update_body_lists()

        # Update camera controls to match new model
        self._update_camera_sliders()

    def _clear_actuator_controls(self) -> None:
        """Remove all existing actuator control widgets."""
        # Clear all widget references
        self.actuator_sliders.clear()
        self.actuator_labels.clear()
        self.actuator_control_types.clear()
        self.actuator_constant_inputs.clear()
        self.actuator_polynomial_coeffs.clear()
        self.actuator_damping_inputs.clear()

        # Delete all control widgets
        for widget in self.actuator_control_widgets:
            self.actuator_layout.removeWidget(widget)
            widget.deleteLater()
        self.actuator_control_widgets.clear()

        if self._simplified_notice is not None:
            self.actuator_layout.removeWidget(self._simplified_notice)
            self._simplified_notice.deleteLater()
            self._simplified_notice = None

        self.simplified_actuator_mode = False

        # Remove and delete all groups
        for group in self.actuator_groups:
            self.actuator_layout.removeWidget(group)
            group.deleteLater()
        self.actuator_groups.clear()

        # Force layout update
        self.actuator_container.update()

    def _create_actuator_controls(self, actuator_names: list[str]) -> None:
        """Create advanced controls for all actuators with logical grouping.

        This method creates control widgets for each actuator in the current model.
        The actuator_index corresponds directly to the model's actuator indices\
        (0 to model.nu-1).
        """
        groups = self._group_actuators(actuator_names)
        actuator_index = 0

        total_actuators = len(actuator_names)
        self.simplified_actuator_mode = (
            total_actuators >= self.SIMPLIFIED_ACTUATOR_THRESHOLD
        )

        if self.simplified_actuator_mode:
            self._simplified_notice = QtWidgets.QLabel(
                "Large musculoskeletal model detected. Showing simplified actuator "
                "controls to keep the interface responsive. Use the Edit buttons "
                "for detailed polynomial, sine, or damping parameters.",
            )
            self._simplified_notice.setWordWrap(True)
            self._simplified_notice.setStyleSheet(
                "background-color: #fff3cd; border: 1px solid #ffeeba; padding: 6px;",
            )
            self.actuator_layout.addWidget(self._simplified_notice)

        for group_name, actuators in groups.items():
            # Make group box collapsible
            group_box = QtWidgets.QGroupBox(
                f"{group_name} ({len(actuators)} actuators)",
            )
            group_box.setCheckable(True)
            group_box.setChecked(True)  # Expanded by default
            group_box.setProperty("actuator_names", actuators)

            # Content widget that will be shown/hidden
            content_widget = QtWidgets.QWidget()
            content_layout = QtWidgets.QVBoxLayout(content_widget)
            content_layout.setContentsMargins(0, 0, 0, 0)

            for actuator_name in actuators:
                if self.simplified_actuator_mode:
                    control_widget = self._create_simplified_actuator_row(
                        actuator_index,
                        actuator_name,
                    )
                else:
                    # Create advanced control widget for this actuator
                    # actuator_index matches the model's actuator index (0-based)
                    control_widget = self._create_advanced_actuator_control(
                        actuator_index,
                        actuator_name,
                    )
                self.actuator_control_widgets.append(control_widget)
                content_layout.addWidget(control_widget)
                actuator_index += 1

            group_layout = QtWidgets.QVBoxLayout(group_box)
            group_layout.addWidget(content_widget)

            # Connect group toggle to show/hide content
            group_box.toggled.connect(
                lambda checked, w=content_widget: w.setVisible(checked),
            )

            self.actuator_groups.append(group_box)
            self.actuator_layout.addWidget(group_box)

        # Verify we created controls for all actuators
        if self.sim_widget.model is not None:
            expected_count = self.sim_widget.model.nu
            actual_count = actuator_index
            if actual_count != expected_count:
                logger.warning(
                    "Created %d control widgets but model has %d actuators",
                    actual_count,
                    expected_count,
                )
            else:
                logger.debug(
                    "Successfully created %d actuator controls matching model",
                    actual_count,
                )

        # Update actuator summary
        self._update_actuator_summary()

        self.actuator_layout.addStretch(1)

    def _create_simplified_actuator_row(
        self,
        actuator_index: int,
        actuator_name: str,
    ) -> QtWidgets.QWidget:
        """Create lightweight controls for very large actuator sets."""
        container = QtWidgets.QFrame()
        layout = QtWidgets.QHBoxLayout(container)
        layout.setContentsMargins(4, 2, 4, 2)

        name_label = QtWidgets.QLabel(f"<b>{actuator_name}</b>")
        layout.addWidget(name_label, stretch=2)

        slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        slider.setMinimum(-100)
        slider.setMaximum(100)
        slider.setValue(0)
        slider.setSingleStep(1)
        slider.valueChanged.connect(
            lambda val, i=actuator_index: self.on_actuator_slider_changed(i, val),
        )
        self.actuator_sliders.append(slider)
        layout.addWidget(slider, stretch=3)

        value_label = QtWidgets.QLabel("0 Nm")
        value_label.setMinimumWidth(60)
        self.actuator_labels.append(value_label)
        layout.addWidget(value_label)

        detail_btn = QtWidgets.QPushButton("Edit…")
        detail_btn.setToolTip("Open detailed control options for this actuator")
        detail_btn.clicked.connect(
            lambda *, i=actuator_index, name=actuator_name, s=slider: (
                self.open_actuator_detail_dialog(i, name, slider=s)
            ),
        )
        layout.addWidget(detail_btn)

        return container

    def _create_advanced_actuator_control(
        self,
        actuator_index: int,
        actuator_name: str,
    ) -> QtWidgets.QWidget:
        """Create an advanced control widget for a single actuator.

        Returns a collapsible widget with:
        - Control type selector
        - Constant value input
        - Polynomial coefficient inputs (7 coefficients)
        - Damping control
        - Quick slider for constant control
        """
        # Main container
        container = QtWidgets.QFrame()
        container.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        container_layout = QtWidgets.QVBoxLayout(container)
        container_layout.setContentsMargins(5, 5, 5, 5)

        # Header with actuator name and control type
        header_layout = QtWidgets.QHBoxLayout()
        name_label = QtWidgets.QLabel(f"<b>{actuator_name}</b>")
        header_layout.addWidget(name_label)

        # Control type selector
        control_type_combo = QtWidgets.QComboBox()
        control_type_combo.addItems(
            ["Constant", "Polynomial (6th order)", "Sine Wave", "Step Function"],
        )
        control_type_combo.setCurrentIndex(0)  # Default to constant
        control_type_combo.currentIndexChanged.connect(
            lambda idx, i=actuator_index: self.on_control_type_changed(i, idx),
        )
        self.actuator_control_types.append(control_type_combo)
        header_layout.addWidget(QtWidgets.QLabel("Type:"))
        header_layout.addWidget(control_type_combo)
        header_layout.addStretch()
        container_layout.addLayout(header_layout)

        # Quick constant control (slider + value input)
        quick_control_layout = QtWidgets.QHBoxLayout()
        slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        slider.setMinimum(-100)
        slider.setMaximum(100)
        slider.setValue(0)
        slider.setSingleStep(1)
        slider.valueChanged.connect(
            lambda val, i=actuator_index: self.on_actuator_slider_changed(i, val),
        )
        self.actuator_sliders.append(slider)

        constant_input = QtWidgets.QDoubleSpinBox()
        constant_input.setRange(-1000.0, 1000.0)
        constant_input.setSingleStep(1.0)
        constant_input.setDecimals(2)
        constant_input.setValue(0.0)
        constant_input.setSuffix(" Nm")
        constant_input.valueChanged.connect(
            lambda val, i=actuator_index: self.on_constant_value_changed(i, val),
        )
        self.actuator_constant_inputs.append(constant_input)

        label = QtWidgets.QLabel("0 Nm")
        label.setMinimumWidth(60)
        self.actuator_labels.append(label)

        quick_control_layout.addWidget(QtWidgets.QLabel("Value:"))
        quick_control_layout.addWidget(slider, stretch=2)
        quick_control_layout.addWidget(constant_input)
        quick_control_layout.addWidget(label)
        container_layout.addLayout(quick_control_layout)

        # Damping control (always visible)
        damping_layout = QtWidgets.QHBoxLayout()
        damping_input = QtWidgets.QDoubleSpinBox()
        damping_input.setRange(0.0, 100.0)
        damping_input.setSingleStep(0.1)
        damping_input.setDecimals(2)
        damping_input.setValue(0.0)
        damping_input.setSuffix(" N·s/m")
        damping_input.valueChanged.connect(
            lambda val, i=actuator_index: self.on_damping_changed(i, val),
        )
        self.actuator_damping_inputs.append(damping_input)
        damping_layout.addWidget(QtWidgets.QLabel("Damping:"))
        damping_layout.addWidget(damping_input)
        damping_layout.addStretch()
        container_layout.addLayout(damping_layout)

        # Polynomial coefficients (hidden by default, shown when polynomial selected)
        poly_widget = QtWidgets.QWidget()
        poly_layout = QtWidgets.QVBoxLayout(poly_widget)
        poly_layout.setContentsMargins(10, 5, 5, 5)

        poly_label = QtWidgets.QLabel(
            "Polynomial Coefficients (c0 + c1*t + c2*t^2 + ... + c6*t^6):",
        )
        poly_label.setStyleSheet("font-weight: bold;")
        poly_layout.addWidget(poly_label)

        coeff_spinboxes = []
        for i in range(7):  # 7 coefficients for 6th order polynomial
            coeff_layout = QtWidgets.QHBoxLayout()
            coeff_label = QtWidgets.QLabel(f"c{i}:")
            coeff_label.setMinimumWidth(30)
            coeff_spinbox = QtWidgets.QDoubleSpinBox()
            coeff_spinbox.setRange(-1000.0, 1000.0)
            coeff_spinbox.setSingleStep(0.1)
            coeff_spinbox.setDecimals(4)
            coeff_spinbox.setValue(0.0)
            coeff_spinbox.valueChanged.connect(
                lambda val, idx=i, act_idx=actuator_index: (
                    self.on_polynomial_coeff_changed(act_idx, idx, val)
                ),
            )
            coeff_spinboxes.append(coeff_spinbox)
            coeff_layout.addWidget(coeff_label)
            coeff_layout.addWidget(coeff_spinbox, stretch=1)
            poly_layout.addLayout(coeff_layout)

        self.actuator_polynomial_coeffs.append(coeff_spinboxes)
        poly_widget.setVisible(False)  # Hidden by default
        container_layout.addWidget(poly_widget)

        # Sine Wave parameters (hidden by default)
        sine_widget = QtWidgets.QWidget()
        sine_layout = QtWidgets.QVBoxLayout(sine_widget)
        sine_layout.setContentsMargins(10, 5, 5, 5)

        sine_title = QtWidgets.QLabel(
            "<b>Sine Wave Parameters:</b> amplitude x sin(2πft + phase)",
        )
        sine_layout.addWidget(sine_title)

        # Amplitude
        amp_layout = QtWidgets.QHBoxLayout()
        amp_layout.addWidget(QtWidgets.QLabel("Amplitude:"))
        sine_amp_spin = QtWidgets.QDoubleSpinBox()
        sine_amp_spin.setRange(0.0, 1000.0)
        sine_amp_spin.setSingleStep(1.0)
        sine_amp_spin.setValue(10.0)
        sine_amp_spin.setSuffix(" Nm")
        sine_amp_spin.setToolTip("Peak torque amplitude of the sine wave")
        sine_amp_spin.valueChanged.connect(
            lambda val, i=actuator_index: self._on_sine_param_changed(
                i,
                "amplitude",
                val,
            ),
        )
        amp_layout.addWidget(sine_amp_spin)
        sine_layout.addLayout(amp_layout)

        # Frequency
        freq_layout = QtWidgets.QHBoxLayout()
        freq_layout.addWidget(QtWidgets.QLabel("Frequency:"))
        sine_freq_spin = QtWidgets.QDoubleSpinBox()
        sine_freq_spin.setRange(0.01, 100.0)
        sine_freq_spin.setSingleStep(0.1)
        sine_freq_spin.setValue(1.0)
        sine_freq_spin.setSuffix(" Hz")
        sine_freq_spin.setToolTip("Oscillation frequency in Hz (cycles per second)")
        sine_freq_spin.valueChanged.connect(
            lambda val, i=actuator_index: self._on_sine_param_changed(
                i,
                "frequency",
                val,
            ),
        )
        freq_layout.addWidget(sine_freq_spin)
        sine_layout.addLayout(freq_layout)

        # Phase
        phase_layout = QtWidgets.QHBoxLayout()
        phase_layout.addWidget(QtWidgets.QLabel("Phase:"))
        sine_phase_spin = QtWidgets.QDoubleSpinBox()
        sine_phase_spin.setRange(-6.28, 6.28)
        sine_phase_spin.setSingleStep(0.1)
        sine_phase_spin.setValue(0.0)
        sine_phase_spin.setSuffix(" rad")
        sine_phase_spin.setToolTip("Phase offset in radians (0 to 2π)")
        sine_phase_spin.valueChanged.connect(
            lambda val, i=actuator_index: self._on_sine_param_changed(i, "phase", val),
        )
        phase_layout.addWidget(sine_phase_spin)
        sine_layout.addLayout(phase_layout)

        sine_widget.setVisible(False)  # Hidden by default
        container_layout.addWidget(sine_widget)

        # Step function parameters (hidden by default)
        step_widget = QtWidgets.QWidget()
        step_layout = QtWidgets.QVBoxLayout(step_widget)
        step_layout.setContentsMargins(10, 5, 5, 5)

        step_title = QtWidgets.QLabel(
            "<b>Step Function:</b> 0 → step_value at step_time",
        )
        step_layout.addWidget(step_title)

        # Step time
        time_layout = QtWidgets.QHBoxLayout()
        time_layout.addWidget(QtWidgets.QLabel("Step Time:"))
        step_time_spin = QtWidgets.QDoubleSpinBox()
        step_time_spin.setRange(0.0, 100.0)
        step_time_spin.setSingleStep(0.1)
        step_time_spin.setValue(0.5)
        step_time_spin.setSuffix(" s")
        step_time_spin.setToolTip("Time at which step occurs (seconds)")
        step_time_spin.valueChanged.connect(
            lambda val, i=actuator_index: self._on_step_param_changed(i, "time", val),
        )
        time_layout.addWidget(step_time_spin)
        step_layout.addLayout(time_layout)

        # Step value
        value_layout = QtWidgets.QHBoxLayout()
        value_layout.addWidget(QtWidgets.QLabel("Step Value:"))
        step_value_spin = QtWidgets.QDoubleSpinBox()
        step_value_spin.setRange(-1000.0, 1000.0)
        step_value_spin.setSingleStep(1.0)
        step_value_spin.setValue(50.0)
        step_value_spin.setSuffix(" Nm")
        step_value_spin.setToolTip("Torque value after step occurs")
        step_value_spin.valueChanged.connect(
            lambda val, i=actuator_index: self._on_step_param_changed(i, "value", val),
        )
        value_layout.addWidget(step_value_spin)
        step_layout.addLayout(value_layout)

        step_widget.setVisible(False)  # Hidden by default
        container_layout.addWidget(step_widget)

        # Store references to all parameter widgets for showing/hiding
        control_type_combo.polynomial_widget = poly_widget
        control_type_combo.sine_widget = sine_widget
        control_type_combo.step_widget = step_widget

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
                self,
                "Control System",
                "Control system is not initialized yet.",
            )
            return

        slider_sync: Callable[[float], None] | None = None
        if slider is not None:

            def slider_sync_func(value: float) -> None:
                """Update slider value from dialog control."""
                slider.setValue(int(value))

            slider_sync = slider_sync_func

        dialog = ActuatorDetailDialog(
            control_system=control_system,
            actuator_index=actuator_index,
            actuator_name=actuator_name,
            slider_sync=slider_sync,
            parent=self,
        )
        dialog.exec()

    def _on_sine_param_changed(
        self,
        actuator_index: int,
        param: str,
        value: float,
    ) -> None:
        """Handle sine wave parameter changes."""
        control_system = self.sim_widget.get_control_system()
        if control_system is None:
            return

        control = control_system.get_actuator_control(actuator_index)
        if param == "amplitude":
            control.sine_amplitude = value
        elif param == "frequency":
            control.sine_frequency = value
        elif param == "phase":
            control.sine_phase = value

        # Ensure control type is set to sine wave
        control_system.set_control_type(actuator_index, ControlType.SINE_WAVE)

    def _on_step_param_changed(
        self,
        actuator_index: int,
        param: str,
        value: float,
    ) -> None:
        """Handle step function parameter changes."""
        control_system = self.sim_widget.get_control_system()
        if control_system is None:
            return

        control = control_system.get_actuator_control(actuator_index)
        if param == "time":
            control.step_time = value
        elif param == "value":
            control.step_value = value

        # Ensure control type is set to step
        control_system.set_control_type(actuator_index, ControlType.STEP)

    def _group_actuators(self, actuator_names: list[str]) -> dict[str, list[str]]:
        """Group actuators by body part for organized display."""
        groups: dict[str, list[str]] = {
            "Control Inputs": [],
            "Legs": [],
            "Torso/Spine": [],
            "Left Scapula": [],
            "Right Scapula": [],
            "Left Arm": [],
            "Right Arm": [],
            "Club/Shaft": [],
            "Simple Joints": [],
        }

        for name in actuator_names:
            name_lower = name.lower()
            if (
                "base drive" in name_lower
                or "forcing" in name_lower
                or "pendulum control" in name_lower
            ):
                groups["Control Inputs"].append(name)
            elif "ankle" in name_lower or "knee" in name_lower:
                groups["Legs"].append(name)
            elif "spine" in name_lower:
                groups["Torso/Spine"].append(name)
            elif name.startswith("L Scap") or (
                "scap" in name_lower and name.startswith("L")
            ):
                groups["Left Scapula"].append(name)
            elif name.startswith("R Scap") or (
                "scap" in name_lower and name.startswith("R")
            ):
                groups["Right Scapula"].append(name)
            elif name.startswith("L "):
                groups["Left Arm"].append(name)
            elif name.startswith("R "):
                groups["Right Arm"].append(name)
            elif "club" in name_lower or "shaft" in name_lower:
                groups["Club/Shaft"].append(name)
            else:
                groups["Simple Joints"].append(name)

        return {k: v for k, v in groups.items() if v}

    def _create_slider_control(
        self,
        actuator_name: str,
    ) -> tuple[QtWidgets.QSlider, QtWidgets.QLabel]:
        """Create a slider and label for a single actuator."""
        slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        slider.setMinimum(-100)
        slider.setMaximum(100)
        slider.setValue(0)
        slider.setSingleStep(1)
        slider.valueChanged.connect(self.on_actuator_changed)

        label = QtWidgets.QLabel("0 Nm")
        label.setMinimumWidth(60)

        return slider, label

    # -------- Callbacks --------

    def on_actuator_filter_changed(self, text: str) -> None:
        """Filter actuator groups by substring match."""
        if not self.actuator_groups:
            return

        pattern = text.strip().lower()
        for group in self.actuator_groups:
            if not isinstance(group, QtWidgets.QGroupBox):
                continue
            if pattern == "":
                group.setVisible(True)
                continue

            names = group.property("actuator_names") or []
            matches = pattern in group.title().lower() or any(
                pattern in name.lower() for name in names
            )
            group.setVisible(matches)

    def on_model_changed(self, index: int) -> None:
        """Handle model selection change."""
        self.load_current_model()
        self.sim_widget.reset_state()

    def on_play_pause_toggled(self, checked: bool) -> None:
        """Handle play/pause button toggle."""
        if checked:
            self.play_pause_btn.setText("Play")
            self.sim_widget.set_running(False)
        else:
            self.play_pause_btn.setText("Pause")
            self.sim_widget.set_running(True)

    def on_reset_clicked(self) -> None:
        """Reset simulation to initial state."""
        self.sim_widget.reset_state()
        self.sim_widget.reset_control_system()

        # Reset all controls to default values
        for slider in self.actuator_sliders:
            slider.setValue(0)
        for constant_input in self.actuator_constant_inputs:
            constant_input.setValue(0.0)
        for damping_input in self.actuator_damping_inputs:
            damping_input.setValue(0.0)
        for coeff_list in self.actuator_polynomial_coeffs:
            for coeff_spinbox in coeff_list:
                coeff_spinbox.setValue(0.0)
        for control_type_combo in self.actuator_control_types:
            control_type_combo.setCurrentIndex(0)  # Reset to constant
            # Hide all parameter widgets
            if hasattr(control_type_combo, "polynomial_widget"):
                control_type_combo.polynomial_widget.setVisible(False)
            if hasattr(control_type_combo, "sine_widget"):
                control_type_combo.sine_widget.setVisible(False)
            if hasattr(control_type_combo, "step_widget"):
                control_type_combo.step_widget.setVisible(False)

    def on_record_toggled(self, checked: bool) -> None:
        """Handle recording toggle."""
        recorder = self.sim_widget.get_recorder()
        if checked:
            self.record_btn.setText("Stop Recording")
            recorder.start_recording()
        else:
            self.record_btn.setText("Start Recording")
            recorder.stop_recording()

    def on_actuator_changed(self) -> None:
        """Update actuator torques when any slider changes."""
        for i, (slider, label) in enumerate(
            zip(self.actuator_sliders, self.actuator_labels),  # noqa: B905
        ):
            value = float(slider.value())
            label.setText(f"{value:.0f} Nm")
            self.sim_widget.set_joint_torque(i, value)

    def on_actuator_slider_changed(self, actuator_index: int, value: int) -> None:
        """Handle slider change - update constant value and label."""
        if actuator_index < len(self.actuator_constant_inputs):
            self.actuator_constant_inputs[actuator_index].setValue(float(value))
        if actuator_index < len(self.actuator_labels):
            self.actuator_labels[actuator_index].setText(f"{value:.0f} Nm")
        # Update control system
        control_system = self.sim_widget.get_control_system()
        if control_system is not None:
            control_system.set_constant_value(actuator_index, float(value))
            control_system.set_control_type(actuator_index, ControlType.CONSTANT)

    def on_constant_value_changed(self, actuator_index: int, value: float) -> None:
        """Handle constant value input change."""
        # Update slider
        if actuator_index < len(self.actuator_sliders):
            self.actuator_sliders[actuator_index].setValue(int(value))
        # Update control system
        control_system = self.sim_widget.get_control_system()
        if control_system is not None:
            control_system.set_constant_value(actuator_index, value)
            control_system.set_control_type(actuator_index, ControlType.CONSTANT)

    def on_control_type_changed(self, actuator_index: int, type_index: int) -> None:
        """Handle control type selection change."""
        control_system = self.sim_widget.get_control_system()
        if control_system is None:
            return

        # Map index to ControlType
        type_map = [
            ControlType.CONSTANT,
            ControlType.POLYNOMIAL,
            ControlType.SINE_WAVE,
            ControlType.STEP,
        ]

        if type_index < len(type_map):
            control_type = type_map[type_index]
            control_system.set_control_type(actuator_index, control_type)

            # Show/hide parameter widgets based on control type
            if actuator_index < len(self.actuator_control_types):
                combo = self.actuator_control_types[actuator_index]

                # Hide all parameter widgets first
                if hasattr(combo, "polynomial_widget"):
                    combo.polynomial_widget.setVisible(False)
                if hasattr(combo, "sine_widget"):
                    combo.sine_widget.setVisible(False)
                if hasattr(combo, "step_widget"):
                    combo.step_widget.setVisible(False)

                # Show the appropriate widget for the selected type
                if control_type == ControlType.POLYNOMIAL and hasattr(
                    combo,
                    "polynomial_widget",
                ):
                    combo.polynomial_widget.setVisible(True)
                elif control_type == ControlType.SINE_WAVE and hasattr(
                    combo,
                    "sine_widget",
                ):
                    combo.sine_widget.setVisible(True)
                elif control_type == ControlType.STEP and hasattr(combo, "step_widget"):
                    combo.step_widget.setVisible(True)

    def on_polynomial_coeff_changed(
        self,
        actuator_index: int,
        coeff_index: int,
        value: float,
    ) -> None:
        """Handle polynomial coefficient change."""
        control_system = self.sim_widget.get_control_system()
        if control_system is None:
            return

        # Get current coefficients
        control = control_system.get_actuator_control(actuator_index)
        coeffs = control.get_polynomial_coeffs()
        coeffs[coeff_index] = value

        # Update control system
        control_system.set_polynomial_coeffs(actuator_index, coeffs)
        control_system.set_control_type(actuator_index, ControlType.POLYNOMIAL)

    def on_damping_changed(self, actuator_index: int, damping: float) -> None:
        """Handle damping value change."""
        control_system = self.sim_widget.get_control_system()
        if control_system is not None:
            control_system.set_damping(actuator_index, damping)

    def on_camera_changed(self, camera_name: str) -> None:
        """Handle camera view change."""
        self.sim_widget.set_camera(camera_name)
        # Update sliders to match camera preset
        self._update_camera_sliders()

    def _update_camera_sliders(self) -> None:
        """Update camera control sliders to match current camera state."""
        if self.sim_widget.camera is not None:
            # Update azimuth (0-360)
            az = self.sim_widget.camera.azimuth % 360
            self.azimuth_slider.setValue(int(az))
            self.azimuth_label.setText(f"{az:.1f}°")

            # Update elevation
            el = self.sim_widget.camera.elevation
            self.elevation_slider.setValue(int(el))
            self.elevation_label.setText(f"{el:.1f}°")

            # Update distance (convert to slider scale: 1-500 represents 0.1-50.0)
            dist = self.sim_widget.camera.distance
            slider_val = int((dist - 0.1) / (50.0 - 0.1) * 499) + 1
            self.distance_slider.setValue(slider_val)
            self.distance_label.setText(f"{dist:.2f}")

            # Update lookat
            lookat = self.sim_widget.camera.lookat
            self.lookat_x_spin.setValue(lookat[0])
            self.lookat_y_spin.setValue(lookat[1])
            self.lookat_z_spin.setValue(lookat[2])

    def on_azimuth_changed(self, value: int) -> None:
        """Handle azimuth slider change."""
        self.sim_widget.set_camera_azimuth(float(value))
        self.azimuth_label.setText(f"{value}°")

    def on_elevation_changed(self, value: int) -> None:
        """Handle elevation slider change."""
        self.sim_widget.set_camera_elevation(float(value))
        self.elevation_label.setText(f"{value}°")

    def on_distance_changed(self, value: int) -> None:
        """Handle distance slider change."""
        # Convert slider value (1-500) to distance (0.1-50.0)
        distance = 0.1 + (value - 1) / 499.0 * (50.0 - 0.1)
        self.sim_widget.set_camera_distance(distance)
        self.distance_label.setText(f"{distance:.2f}")

    def on_lookat_changed(self) -> None:
        """Handle lookat position change."""
        x = self.lookat_x_spin.value()
        y = self.lookat_y_spin.value()
        z = self.lookat_z_spin.value()
        self.sim_widget.set_camera_lookat(x, y, z)

    def on_reset_camera(self) -> None:
        """Reset camera to default position."""
        self.sim_widget.reset_camera()
        self._update_camera_sliders()

    def on_sky_color_clicked(self) -> None:
        """Handle sky color button click - open color picker."""
        current_color = QtGui.QColor(
            int(self.sim_widget.sky_color[0] * 255),
            int(self.sim_widget.sky_color[1] * 255),
            int(self.sim_widget.sky_color[2] * 255),
        )
        color = QtWidgets.QColorDialog.getColor(current_color, self, "Select Sky Color")
        if color.isValid():
            rgba = [
                color.red() / 255.0,
                color.green() / 255.0,
                color.blue() / 255.0,
                1.0,
            ]
            self.sim_widget.set_background_color(sky_color=rgba)
            # Update button color
            self.sky_color_btn.setStyleSheet(
                f"background-color: rgb({color.red()}, {color.green()}, \
                    {color.blue()});",
            )

    def on_ground_color_clicked(self) -> None:
        """Handle ground color button click - open color picker."""
        current_color = QtGui.QColor(
            int(self.sim_widget.ground_color[0] * 255),
            int(self.sim_widget.ground_color[1] * 255),
            int(self.sim_widget.ground_color[2] * 255),
        )
        color = QtWidgets.QColorDialog.getColor(
            current_color,
            self,
            "Select Ground Color",
        )
        if color.isValid():
            rgba = [
                color.red() / 255.0,
                color.green() / 255.0,
                color.blue() / 255.0,
                1.0,
            ]
            self.sim_widget.set_background_color(ground_color=rgba)
            # Update button color
            self.ground_color_btn.setStyleSheet(
                f"background-color: rgb({color.red()}, {color.green()}, \
                    {color.blue()});",
            )

    def on_reset_background(self) -> None:
        """Reset background colors to defaults."""
        default_sky = [0.2, 0.3, 0.4, 1.0]
        default_ground = [0.2, 0.2, 0.2, 1.0]
        self.sim_widget.set_background_color(
            sky_color=default_sky,
            ground_color=default_ground,
        )
        # Update button colors
        self.sky_color_btn.setStyleSheet("background-color: rgb(51, 77, 102);")
        self.ground_color_btn.setStyleSheet("background-color: rgb(51, 51, 51);")

    def on_show_torques_changed(self, state: int) -> None:
        """Handle torque visualization toggle."""
        enabled = state == QtCore.Qt.CheckState.Checked.value
        self.sim_widget.set_torque_visualization(enabled)

    def on_torque_scale_changed(self, value: int) -> None:
        """Handle torque scale slider change."""
        scale = value / 100.0  # Convert to 0.01 - 1.0
        self.torque_scale_label.setText(f"{scale:.2f}%")
        self.sim_widget.set_torque_visualization(
            self.show_torques_cb.isChecked(),
            scale * 0.01,
        )

    def on_show_forces_changed(self, state: int) -> None:
        """Handle force visualization toggle."""
        enabled = state == QtCore.Qt.CheckState.Checked.value
        self.sim_widget.set_force_visualization(enabled)

    def on_force_scale_changed(self, value: int) -> None:
        """Handle force scale slider change."""
        scale = value / 10.0
        self.force_scale_label.setText(f"{scale:.1f}%")
        self.sim_widget.set_force_visualization(
            self.show_forces_cb.isChecked(),
            scale * 0.1,
        )

    def on_show_contacts_changed(self, state: int) -> None:
        """Handle contact force visualization toggle."""
        enabled = state == QtCore.Qt.CheckState.Checked.value
        self.sim_widget.set_contact_force_visualization(enabled)

    def on_plot_type_changed(self, plot_type: str) -> None:
        """Handle plot type selection change."""
        # Show joint selector only for phase diagrams
        self.joint_select_widget.setVisible(plot_type == "Phase Diagram")

    def on_generate_plot(self) -> None:
        """Generate the selected plot."""
        recorder = self.sim_widget.get_recorder()

        if recorder.get_num_frames() == 0:
            QtWidgets.QMessageBox.warning(
                self,
                "No Data",
                "No recorded data available. Please record some data first.",
            )
            return

        # Clear existing plot
        if self.current_plot_canvas is not None:
            self.plot_container_layout.removeWidget(self.current_plot_canvas)
            self.current_plot_canvas.deleteLater()

        # Create new canvas
        canvas = MplCanvas(width=8, height=6, dpi=100)
        plotter = GolfSwingPlotter(recorder, self.sim_widget.model)

        # Generate appropriate plot
        plot_type = self.plot_combo.currentText()

        try:
            if plot_type == "Summary Dashboard":
                plotter.plot_summary_dashboard(canvas.fig)
            elif plot_type == "Joint Angles":
                plotter.plot_joint_angles(canvas.fig)
            elif plot_type == "Joint Velocities":
                plotter.plot_joint_velocities(canvas.fig)
            elif plot_type == "Joint Torques":
                plotter.plot_joint_torques(canvas.fig)
            elif plot_type == "Actuator Powers":
                plotter.plot_actuator_powers(canvas.fig)
            elif plot_type == "Energy Analysis":
                plotter.plot_energy_analysis(canvas.fig)
            elif plot_type == "Club Head Speed":
                plotter.plot_club_head_speed(canvas.fig)
            elif plot_type == "Club Head Trajectory (3D)":
                plotter.plot_club_head_trajectory(canvas.fig)
            elif plot_type == "Phase Diagram":
                joint_idx = self.joint_select_combo.currentIndex()
                plotter.plot_phase_diagram(canvas.fig, joint_idx)
            elif plot_type == "Torque Comparison":
                plotter.plot_torque_comparison(canvas.fig)

            canvas.draw()
            self.current_plot_canvas = canvas
            self.plot_container_layout.addWidget(canvas)

        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self,
                "Plot Error",
                f"Error generating plot: {e!s}",
            )

    def update_metrics(self) -> None:
        """Update real-time metrics display."""
        recorder = self.sim_widget.get_recorder()
        analyzer = self.sim_widget.get_analyzer()

        # Update recording status
        if recorder.is_recording:
            duration = recorder.get_duration()
            num_frames = recorder.get_num_frames()
            self.recording_label.setText(
                f"Recording: {duration:.2f}s ({num_frames} frames)",
            )
            self.recording_label.setStyleSheet(
                "background-color: #d62728; color: white; \
                    font-weight: bold; padding: 5px;",
            )
        else:
            num_frames = recorder.get_num_frames()
            if num_frames > 0:
                duration = recorder.get_duration()
                self.recording_label.setText(
                    f"Stopped: {duration:.2f}s ({num_frames} frames)",
                )
                self.recording_label.setStyleSheet(
                    "background-color: #ff7f0e; color: white; \
                        font-weight: bold; padding: 5px;",
                )
            else:
                self.recording_label.setText("Not recording")
                self.recording_label.setStyleSheet("font-weight: bold; padding: 5px;")

        # Update metrics
        if analyzer is not None:
            _, _, club_speed = analyzer.get_club_head_data()
            _, _, total_energy = analyzer.compute_energies()

            self.club_speed_label.setText(
                f"{club_speed * 2.23694:.1f} mph ({club_speed:.1f} m/s)",
            )
            self.total_energy_label.setText(f"{total_energy:.2f} J")

        self.recording_time_label.setText(f"{recorder.get_duration():.2f} s")
        self.num_frames_label.setText(str(recorder.get_num_frames()))

    def on_export_csv(self) -> None:
        """Export recorded data to CSV."""
        recorder = self.sim_widget.get_recorder()

        if recorder.get_num_frames() == 0:
            QtWidgets.QMessageBox.warning(
                self,
                "No Data",
                "No recorded data available to export.",
            )
            return

        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export CSV",
            "",
            "CSV Files (*.csv)",
        )

        if filename:
            try:
                data_dict = recorder.export_to_dict()

                # Write to CSV
                with Path(filename).open("w", newline="") as csvfile:
                    writer = csv.writer(csvfile)

                    # Write header
                    writer.writerow(data_dict.keys())

                    # Write data rows
                    num_rows = len(next(iter(data_dict.values())))
                    for i in range(num_rows):
                        row = [
                            data_dict[key][i] if i < len(data_dict[key]) else ""
                            for key in data_dict
                        ]
                        writer.writerow(row)

                QtWidgets.QMessageBox.information(
                    self,
                    "Export Successful",
                    f"Data exported to {filename}",
                )

            except Exception as e:
                QtWidgets.QMessageBox.critical(
                    self,
                    "Export Error",
                    f"Error exporting data: {e!s}",
                )

    def on_export_json(self) -> None:
        """Export recorded data to JSON."""
        recorder = self.sim_widget.get_recorder()

        if recorder.get_num_frames() == 0:
            QtWidgets.QMessageBox.warning(
                self,
                "No Data",
                "No recorded data available to export.",
            )
            return

        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export JSON",
            "",
            "JSON Files (*.json)",
        )

        if filename:
            try:
                data_dict = recorder.export_to_dict()

                with Path(filename).open("w") as jsonfile:
                    json.dump(data_dict, jsonfile, indent=2)

                QtWidgets.QMessageBox.information(
                    self,
                    "Export Successful",
                    f"Data exported to {filename}",
                )

            except Exception as e:
                QtWidgets.QMessageBox.critical(
                    self,
                    "Export Error",
                    f"Error exporting data: {e!s}",
                )

    # -------- Interactive manipulation event handlers --------

    def update_body_lists(self) -> None:
        """Update body selection combo boxes."""
        if self.sim_widget.model is None:
            return

        # Clear existing items
        self.constraint_body_combo.clear()
        self.ref_body_combo.clear()

        # Add all bodies

        for body_id in range(1, self.sim_widget.model.nbody):  # Skip world (0)
            body_name = mujoco.mj_id2name(
                self.sim_widget.model,
                mujoco.mjtObj.mjOBJ_BODY,
                body_id,
            )
            if body_name:
                self.constraint_body_combo.addItem(f"{body_id}: {body_name}")
                self.ref_body_combo.addItem(f"{body_id}: {body_name}")

    def on_drag_enabled_changed(self, state: int) -> None:
        """Handle drag mode enable/disable."""
        enabled = state == QtCore.Qt.CheckState.Checked.value
        manipulator = self.sim_widget.get_manipulator()
        if manipulator:
            manipulator.enable_drag(enabled)

    def on_maintain_orientation_changed(self, state: int) -> None:
        """Handle maintain orientation setting."""
        enabled = state == QtCore.Qt.CheckState.Checked.value
        manipulator = self.sim_widget.get_manipulator()
        if manipulator:
            manipulator.maintain_orientation = enabled

    def on_nullspace_posture_changed(self, state: int) -> None:
        """Handle nullspace posture optimization setting."""
        enabled = state == QtCore.Qt.CheckState.Checked.value
        manipulator = self.sim_widget.get_manipulator()
        if manipulator:
            manipulator.use_nullspace_posture = enabled

    def on_add_constraint(self) -> None:
        """Add a constraint to the selected body."""

        manipulator = self.sim_widget.get_manipulator()
        if not manipulator:
            return

        # Get selected body ID from combo box
        body_text = self.constraint_body_combo.currentText()
        if not body_text:
            return

        body_id = int(body_text.split(":")[0])

        # Get constraint type
        constraint_type_idx = self.constraint_type_combo.currentIndex()
        if constraint_type_idx == 0:
            # Fixed in space
            manipulator.add_constraint(body_id, ConstraintType.FIXED_IN_SPACE)
        else:
            # Relative to body
            ref_text = self.ref_body_combo.currentText()
            if not ref_text:
                QtWidgets.QMessageBox.warning(
                    self,
                    "No Reference Body",
                    "Please select a reference body.",
                )
                return

            ref_body_id = int(ref_text.split(":")[0])
            manipulator.add_constraint(
                body_id,
                ConstraintType.RELATIVE_TO_BODY,
                reference_body_id=ref_body_id,
            )

        # Update constraints list
        self.update_constraints_list()

    def on_remove_constraint(self) -> None:
        """Remove constraint from selected body."""
        manipulator = self.sim_widget.get_manipulator()
        if not manipulator:
            return

        body_text = self.constraint_body_combo.currentText()
        if not body_text:
            return

        body_id = int(body_text.split(":")[0])
        manipulator.remove_constraint(body_id)
        self.update_constraints_list()

    def on_clear_constraints(self) -> None:
        """Clear all constraints."""
        manipulator = self.sim_widget.get_manipulator()
        if manipulator:
            manipulator.clear_constraints()
            self.update_constraints_list()

    def update_constraints_list(self) -> None:
        """Update the list of active constraints."""
        self.constraints_list.clear()

        manipulator = self.sim_widget.get_manipulator()
        if not manipulator:
            return

        for body_id in manipulator.get_constrained_bodies():
            body_name = manipulator.get_body_name(body_id)
            self.constraints_list.addItem(f"{body_id}: {body_name}")

    def on_save_pose(self) -> None:
        """Save current pose to library."""
        manipulator = self.sim_widget.get_manipulator()
        if not manipulator:
            return

        pose_name = self.pose_name_input.text().strip()
        if not pose_name:
            QtWidgets.QMessageBox.warning(
                self,
                "No Name",
                "Please enter a name for the pose.",
            )
            return

        # Save pose
        manipulator.save_pose(pose_name)

        # Update pose list
        self.update_pose_list()

        # Clear input
        self.pose_name_input.clear()

        QtWidgets.QMessageBox.information(
            self,
            "Pose Saved",
            f"Pose '{pose_name}' saved successfully.",
        )

    def on_load_pose(self) -> None:
        """Load selected pose from library."""
        manipulator = self.sim_widget.get_manipulator()
        if not manipulator:
            return

        current_item = self.pose_list.currentItem()
        if not current_item:
            QtWidgets.QMessageBox.warning(
                self,
                "No Selection",
                "Please select a pose to load.",
            )
            return

        pose_name = current_item.text()
        if manipulator.load_pose(pose_name):
            QtWidgets.QMessageBox.information(
                self,
                "Pose Loaded",
                f"Pose '{pose_name}' loaded successfully.",
            )
        else:
            QtWidgets.QMessageBox.warning(
                self,
                "Load Failed",
                f"Failed to load pose '{pose_name}'.",
            )

    def on_delete_pose(self) -> None:
        """Delete selected pose from library."""
        manipulator = self.sim_widget.get_manipulator()
        if not manipulator:
            return

        current_item = self.pose_list.currentItem()
        if not current_item:
            QtWidgets.QMessageBox.warning(
                self,
                "No Selection",
                "Please select a pose to delete.",
            )
            return

        pose_name = current_item.text()

        # Confirm deletion
        reply = QtWidgets.QMessageBox.question(
            self,
            "Confirm Deletion",
            f"Are you sure you want to delete pose '{pose_name}'?",
            QtWidgets.QMessageBox.StandardButton.Yes
            | QtWidgets.QMessageBox.StandardButton.No,
        )

        if (
            reply == QtWidgets.QMessageBox.StandardButton.Yes
            and manipulator.delete_pose(pose_name)
        ):
            self.update_pose_list()
            QtWidgets.QMessageBox.information(
                self,
                "Pose Deleted",
                f"Pose '{pose_name}' deleted successfully.",
            )

    def on_export_poses(self) -> None:
        """Export pose library to file."""
        manipulator = self.sim_widget.get_manipulator()
        if not manipulator:
            return

        if len(manipulator.list_poses()) == 0:
            QtWidgets.QMessageBox.warning(self, "No Poses", "No poses to export.")
            return

        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export Pose Library",
            "",
            "JSON Files (*.json)",
        )

        if filename:
            try:
                manipulator.export_pose_library(filename)
                QtWidgets.QMessageBox.information(
                    self,
                    "Export Successful",
                    f"Pose library exported to {filename}",
                )
            except Exception as e:
                QtWidgets.QMessageBox.critical(
                    self,
                    "Export Error",
                    f"Error exporting poses: {e!s}",
                )

    def on_import_poses(self) -> None:
        """Import pose library from file."""
        manipulator = self.sim_widget.get_manipulator()
        if not manipulator:
            return

        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Import Pose Library",
            "",
            "JSON Files (*.json)",
        )

        if filename:
            try:
                count = manipulator.import_pose_library(filename)
                self.update_pose_list()
                QtWidgets.QMessageBox.information(
                    self,
                    "Import Successful",
                    f"Imported {count} poses from {filename}",
                )
            except Exception as e:
                QtWidgets.QMessageBox.critical(
                    self,
                    "Import Error",
                    f"Error importing poses: {e!s}",
                )

    def update_pose_list(self) -> None:
        """Update the pose library list."""
        self.pose_list.clear()

        manipulator = self.sim_widget.get_manipulator()
        if not manipulator:
            return

        for pose_name in manipulator.list_poses():
            self.pose_list.addItem(pose_name)

    def on_interpolate_poses(self, value: int) -> None:
        """Interpolate between two selected poses."""
        manipulator = self.sim_widget.get_manipulator()
        if not manipulator:
            return

        # Update label
        alpha = value / 100.0
        self.interp_label.setText(f"{value}%")

        # Get selected poses
        selected_items = self.pose_list.selectedItems()
        if len(selected_items) != 2:
            return

        pose_a = selected_items[0].text()
        pose_b = selected_items[1].text()

        # Interpolate
        manipulator.interpolate_poses(pose_a, pose_b, alpha)

    def on_ik_damping_changed(self, value: int) -> None:
        """Handle IK damping slider change."""
        manipulator = self.sim_widget.get_manipulator()
        if not manipulator:
            return

        damping = value / 100.0
        manipulator.ik_damping = damping
        self.ik_damping_label.setText(f"{damping:.2f}")

    def on_ik_step_changed(self, value: int) -> None:
        """Handle IK step size slider change."""
        manipulator = self.sim_widget.get_manipulator()
        if not manipulator:
            return

        step_size = value / 100.0
        manipulator.ik_step_size = step_size
        self.ik_step_label.setText(f"{step_size:.2f}")


class ActuatorDetailDialog(QtWidgets.QDialog):
    """On-demand editor for actuator control parameters."""

    CONTROL_TYPE_LABELS: typing.ClassVar[list[str]] = [
        "Constant",
        "Polynomial (6th order)",
        "Sine Wave",
        "Step Function",
    ]

    def __init__(
        self,
        control_system: ControlSystem,
        actuator_index: int,
        actuator_name: str,
        slider_sync: Callable[[float], None] | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
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

    def _type_to_index(self, control_type: ControlType) -> int:
        """Map ControlType to combo-box index."""
        mapping = {
            ControlType.CONSTANT: 0,
            ControlType.POLYNOMIAL: 1,
            ControlType.SINE_WAVE: 2,
            ControlType.STEP: 3,
        }
        return mapping.get(control_type, 0)

    def _index_to_type(self, index: int) -> ControlType:
        """Map combo-box index back to ControlType."""
        mapping = {
            0: ControlType.CONSTANT,
            1: ControlType.POLYNOMIAL,
            2: ControlType.SINE_WAVE,
            3: ControlType.STEP,
        }
        return mapping.get(index, ControlType.CONSTANT)

    def _on_type_changed(self, index: int) -> None:
        """Handle control type selection."""
        control_type = self._index_to_type(index)
        self.control_system.set_control_type(self.actuator_index, control_type)
        self._update_visibility()

    def _on_constant_changed(self, value: float) -> None:
        """Update constant torque and sync slider."""
        self.control_system.set_constant_value(self.actuator_index, value)
        if self.slider_sync is not None:
            self.slider_sync(value)

    def _on_damping_changed(self, value: float) -> None:
        """Update damping coefficient."""
        self.control_system.set_damping(self.actuator_index, value)

    def _on_polynomial_changed(self, coeff_index: int, value: float) -> None:
        """Update a single polynomial coefficient."""
        coeffs = self.control.get_polynomial_coeffs()
        coeffs[coeff_index] = value
        self.control_system.set_polynomial_coeffs(self.actuator_index, coeffs)
        self.control_system.set_control_type(
            self.actuator_index,
            ControlType.POLYNOMIAL,
        )
        self.control_type_combo.setCurrentIndex(
            self._type_to_index(ControlType.POLYNOMIAL),
        )

    def _on_sine_changed(self, param: str, value: float) -> None:
        """Update sine-wave parameters."""
        control = self.control_system.get_actuator_control(self.actuator_index)
        if param == "amplitude":
            control.sine_amplitude = value
        elif param == "frequency":
            control.sine_frequency = value
        elif param == "phase":
            control.sine_phase = value
        self.control_system.set_control_type(self.actuator_index, ControlType.SINE_WAVE)
        self.control_type_combo.setCurrentIndex(
            self._type_to_index(ControlType.SINE_WAVE),
        )

    def _on_step_changed(self, param: str, value: float) -> None:
        """Update step function parameters."""
        control = self.control_system.get_actuator_control(self.actuator_index)
        if param == "time":
            control.step_time = value
        elif param == "value":
            control.step_value = value
        self.control_system.set_control_type(self.actuator_index, ControlType.STEP)
        self.control_type_combo.setCurrentIndex(self._type_to_index(ControlType.STEP))

    def _update_visibility(self) -> None:
        """Show or hide parameter sections based on control type."""
        index = self.control_type_combo.currentIndex()
        self.poly_widget.setVisible(index == 1)
        self.sine_widget.setVisible(index == 2)
        self.step_widget.setVisible(index == 3)
