"""Advanced professional GUI for golf swing analysis.

This module provides a comprehensive interface with:
- Simulation controls and visualization
- Real-time biomechanical analysis
- Advanced plotting and data export
- Force/torque vector visualization
- Camera controls
"""

from __future__ import annotations

import logging
import typing

import mujoco
from PyQt6 import QtCore, QtGui, QtWidgets

from .advanced_gui_methods import AdvancedGuiMethodsMixin
from .grip_modelling_tab import GripModellingTab
from .gui.tabs.analysis_tab import AnalysisTab
from .gui.tabs.controls_tab import ControlsTab
from .gui.tabs.physics_tab import PhysicsTab
from .gui.tabs.visualization_tab import VisualizationTab
from .interactive_manipulation import ConstraintType
from .plotting import GolfSwingPlotter, MplCanvas
from .sim_widget import MuJoCoSimWidget

logger = logging.getLogger(__name__)


class AdvancedGolfAnalysisWindow(QtWidgets.QMainWindow, AdvancedGuiMethodsMixin):
    """Professional golf swing analysis application with comprehensive features."""

    SIMPLIFIED_ACTUATOR_THRESHOLD: typing.Final[int] = 60

    def __init__(self) -> None:
        """Initialize the AdvancedGolfAnalysisWindow.

        Sets up the main window, models, and UI components including
        controls, visualization, analysis, and plotting tabs.
        """
        super().__init__()

        self.setWindowTitle("Golf Swing Biomechanical Analysis Suite")
        self.resize(1600, 900)

        # Apply Global Stylesheet
        self.setStyleSheet(
            """
            QPushButton {
                border-radius: 5px;
                padding: 5px;
            }
            QPushButton:hover {
                border: 1px solid #4a90e2;
            }
            QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
                border-radius: 4px;
                padding: 4px;
                background-color: #333;
                color: white;
            }
            QGroupBox {
                border: 1px solid #555;
                border-radius: 5px;
                margin-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """
        )

        # Model configurations

        # Create central tab widget
        self.main_tab_widget = QtWidgets.QTabWidget()
        self.setCentralWidget(self.main_tab_widget)

        # --- Tab 1: Golf Swing Analysis ---
        self.golf_analysis_widget = QtWidgets.QWidget()
        self.main_tab_widget.addTab(self.golf_analysis_widget, "Golf Swing Analysis")

        main_layout = QtWidgets.QHBoxLayout(self.golf_analysis_widget)

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

        # --- Tab 2: Grip Modelling ---
        self.grip_modelling_tab = GripModellingTab()
        self.main_tab_widget.addTab(self.grip_modelling_tab, "Grip Modelling")

        # Create tabs
        self._create_control_tab()
        self._create_visualization_tab()
        self.analysis_tab = AnalysisTab(self.sim_widget, self)
        self.tab_widget.addTab(self.analysis_tab, "Analysis")
        self._create_plotting_tab()
        self._create_manipulation_tab()

        # Connect grip modelling tab to simulation widget
        self.grip_modelling_tab.connect_sim_widget(self.sim_widget)

        # Current plot canvas
        self.current_plot_canvas: MplCanvas | None = None

        self.sim_widget.reset_state()

        # Apply professional styling
        self._apply_styling()

        # Auto-load configuration if present (overrides defaults if config found)
        self._load_launch_config()

        # Create status bar
        self._create_status_bar()

        # Start status bar update timer
        self.status_timer = QtCore.QTimer(self)
        self.status_timer.timeout.connect(self._update_status_bar)
        self.status_timer.start(200)  # Update every 200ms

    @property
    def model_configs(self) -> list[dict]:
        """Expose model configs from PhysicsTab for mixin compatibility."""
        if hasattr(self, "physics_tab"):
            return self.physics_tab.model_configs
        return []

    @property
    def model_combo(self) -> QtWidgets.QComboBox | None:
        """Expose model combo from PhysicsTab for mixin compatibility."""
        if hasattr(self, "physics_tab"):
            return self.physics_tab.model_combo
        return None

    def _create_control_tab(self) -> None:
        """Create the simulation tabs (Physics, Controls)."""
        # Physics Configuration Tab
        self.physics_tab = PhysicsTab(self.sim_widget, self)
        self.tab_widget.addTab(self.physics_tab, "Physics")

        # Simulation Controls Tab
        self.controls_tab = ControlsTab(self.sim_widget, self)
        self.tab_widget.addTab(self.controls_tab, "Controls")

        # Connect signals
        self.physics_tab.model_changed.connect(self.controls_tab.on_model_loaded)
        self.physics_tab.model_changed.connect(self.on_model_changed_signal)
        self.physics_tab.mode_changed.connect(self.controls_tab.on_mode_changed)

    def on_model_changed_signal(self, model_name: str, config: dict) -> None:
        """Handle model change signal from ControlsTab."""
        # Update body lists for interactive manipulation
        self.update_body_lists()

        # Update camera controls to match new model
        if hasattr(self, "visualization_tab"):
            self.visualization_tab._update_camera_sliders()

        # Update status bar immediately
        self._update_status_bar()

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
        if hasattr(self, "visualization_tab"):
            self.visualization_tab._on_preset_clicked(preset_name)

    def on_reset_camera(self) -> None:
        """Reset camera to default position."""
        if hasattr(self, "visualization_tab"):
            self.visualization_tab.on_reset_camera()
        else:
            self.sim_widget.reset_camera()

    def keyPressEvent(self, event: QtGui.QKeyEvent | None) -> None:  # type: ignore[override]
        """Handle keyboard shortcuts."""
        if event is None:
            return
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
            # Enum key lookup issue with Int key
            preset = camera_shortcuts[key]  # type: ignore[index]
            self._on_quick_camera_clicked(preset)
            return

        # Space key: Play/Pause
        if key == QtCore.Qt.Key.Key_Space:
            if hasattr(self.controls_tab, "play_pause_btn"):
                self.controls_tab.play_pause_btn.toggle()
            return

        # R key: Reset
        if key == QtCore.Qt.Key.Key_R:
            if hasattr(self.controls_tab, "reset_btn"):
                self.controls_tab.reset_btn.click()
            return

        # H key: Toggle help panel
        if key == QtCore.Qt.Key.Key_H:
            # Find the help group and toggle it
            if hasattr(self.controls_tab, "help_group"):
                self.controls_tab.help_group.setChecked(
                    not self.controls_tab.help_group.isChecked(),
                )
            return

        super().keyPressEvent(event)

    def _create_status_bar(self) -> None:
        """Create a status bar showing simulation information."""
        status_bar = self.statusBar()
        if status_bar is None:
            return
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
        if hasattr(self, "physics_tab") and hasattr(self.physics_tab, "model_configs"):
            config_idx = self.physics_tab.model_combo.currentIndex()
            if config_idx < len(self.physics_tab.model_configs):
                model_name = self.physics_tab.model_configs[config_idx]["name"]
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
        self.visualization_tab = VisualizationTab(self.sim_widget, self)
        self.tab_widget.addTab(self.visualization_tab, "Visualization")

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
                "Swing Plane Analysis",
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

        # Global Body Selection
        sel_group = QtWidgets.QGroupBox("Target Selection")
        sel_layout = QtWidgets.QHBoxLayout(sel_group)
        sel_layout.addWidget(QtWidgets.QLabel("Select Body:"))
        self.manip_body_combo = QtWidgets.QComboBox()
        self.manip_body_combo.currentIndexChanged.connect(self.on_manip_body_selected)
        sel_layout.addWidget(self.manip_body_combo, stretch=1)
        manip_layout.addWidget(sel_group)

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

        # Manual Transform Controls
        transform_group = QtWidgets.QGroupBox("Manual Transform (Selected Body)")
        transform_layout = QtWidgets.QVBoxLayout(transform_group)

        # Position
        pos_layout = QtWidgets.QHBoxLayout()
        pos_layout.addWidget(QtWidgets.QLabel("Pos:"))
        self.trans_x = QtWidgets.QDoubleSpinBox()
        self.trans_x.setRange(-10, 10)
        self.trans_x.setSingleStep(0.01)
        self.trans_x.valueChanged.connect(
            lambda v: self.on_manual_transform("pos", 0, v)
        )
        pos_layout.addWidget(self.trans_x)

        self.trans_y = QtWidgets.QDoubleSpinBox()
        self.trans_y.setRange(-10, 10)
        self.trans_y.setSingleStep(0.01)
        self.trans_y.valueChanged.connect(
            lambda v: self.on_manual_transform("pos", 1, v)
        )
        pos_layout.addWidget(self.trans_y)

        self.trans_z = QtWidgets.QDoubleSpinBox()
        self.trans_z.setRange(-10, 10)
        self.trans_z.setSingleStep(0.01)
        self.trans_z.valueChanged.connect(
            lambda v: self.on_manual_transform("pos", 2, v)
        )
        pos_layout.addWidget(self.trans_z)
        transform_layout.addLayout(pos_layout)

        # Rotation (Euler)
        rot_layout = QtWidgets.QHBoxLayout()
        rot_layout.addWidget(QtWidgets.QLabel("Rot:"))
        self.trans_roll = QtWidgets.QDoubleSpinBox()  # X
        self.trans_roll.setRange(-180, 180)
        self.trans_roll.valueChanged.connect(
            lambda v: self.on_manual_transform("rot", 0, v)
        )
        rot_layout.addWidget(self.trans_roll)

        self.trans_pitch = QtWidgets.QDoubleSpinBox()  # Y
        self.trans_pitch.setRange(-180, 180)
        self.trans_pitch.valueChanged.connect(
            lambda v: self.on_manual_transform("rot", 1, v)
        )
        rot_layout.addWidget(self.trans_pitch)

        self.trans_yaw = QtWidgets.QDoubleSpinBox()  # Z
        self.trans_yaw.setRange(-180, 180)
        self.trans_yaw.valueChanged.connect(
            lambda v: self.on_manual_transform("rot", 2, v)
        )
        rot_layout.addWidget(self.trans_yaw)
        transform_layout.addLayout(rot_layout)

        # Helper button to get current values from selection
        self.refresh_trans_btn = QtWidgets.QPushButton("Refresh from Selection")
        self.refresh_trans_btn.clicked.connect(self.update_manual_transform_values)
        transform_layout.addWidget(self.refresh_trans_btn)

        manip_layout.addWidget(transform_group)

        # Constraint controls
        constraint_group = QtWidgets.QGroupBox("Body Constraints")
        constraint_layout = QtWidgets.QVBoxLayout(constraint_group)

        # Body selection
        body_select_layout = QtWidgets.QHBoxLayout()
        body_label = QtWidgets.QLabel("Body:")
        self.constraint_body_combo = QtWidgets.QComboBox()
        body_label.setBuddy(self.constraint_body_combo)
        body_select_layout.addWidget(body_label)
        self.constraint_body_combo.setMinimumWidth(150)
        body_select_layout.addWidget(self.constraint_body_combo, stretch=1)
        constraint_layout.addLayout(body_select_layout)

        # Constraint type
        type_layout = QtWidgets.QHBoxLayout()
        type_label = QtWidgets.QLabel("Type:")
        self.constraint_type_combo = QtWidgets.QComboBox()
        type_label.setBuddy(self.constraint_type_combo)
        type_layout.addWidget(type_label)
        self.constraint_type_combo.addItems(["Fixed in Space", "Relative to Body"])
        type_layout.addWidget(self.constraint_type_combo, stretch=1)
        constraint_layout.addLayout(type_layout)

        # Reference body (for relative constraints)
        self.ref_body_layout = QtWidgets.QHBoxLayout()
        ref_label = QtWidgets.QLabel("Reference:")
        self.ref_body_combo = QtWidgets.QComboBox()
        ref_label.setBuddy(self.ref_body_combo)
        self.ref_body_layout.addWidget(ref_label)
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
        self.pose_name_input.setClearButtonEnabled(True)
        self.pose_name_input.setAccessibleName("Pose Name")
        save_layout.addWidget(self.pose_name_input)
        self.save_pose_btn = QtWidgets.QPushButton("Save Pose")
        self.save_pose_btn.setToolTip(
            "Save the current body configuration to the library",
        )
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
        self.load_pose_btn.setToolTip("Apply the selected pose to the model")
        self.load_pose_btn.clicked.connect(self.on_load_pose)
        self.delete_pose_btn = QtWidgets.QPushButton("Delete")
        self.delete_pose_btn.setToolTip("Remove the selected pose from the library")
        self.delete_pose_btn.clicked.connect(self.on_delete_pose)
        self.export_poses_btn = QtWidgets.QPushButton("Export Library")
        self.export_poses_btn.setToolTip("Save all poses to a JSON file")
        self.export_poses_btn.clicked.connect(self.on_export_poses)
        self.import_poses_btn = QtWidgets.QPushButton("Import Library")
        self.import_poses_btn.setToolTip("Load poses from a JSON file")
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
        self.ik_damping_slider.setToolTip(
            "Adjust damping factor for IK solver to improve stability",
        )
        self.ik_damping_slider.setAccessibleName("IK Damping")
        self.ik_damping_slider.valueChanged.connect(self.on_ik_damping_changed)
        self.ik_damping_label = QtWidgets.QLabel("0.05")
        ik_layout.addRow("Damping:", self.ik_damping_slider)
        ik_layout.addRow("", self.ik_damping_label)

        self.ik_step_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.ik_step_slider.setMinimum(1)
        self.ik_step_slider.setMaximum(100)
        self.ik_step_slider.setValue(30)
        self.ik_step_slider.setToolTip("Adjust step size for IK solver convergence")
        self.ik_step_slider.setAccessibleName("IK Step Size")
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
```
        """,
        )

    # -------- Model management --------

    def _update_camera_sliders(self) -> None:
        """Update camera control sliders to match current camera state."""
        if hasattr(self, "visualization_tab"):
            self.visualization_tab._update_camera_sliders()

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

        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
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
            elif plot_type == "Swing Plane Analysis":
                plotter.plot_swing_plane(canvas.fig)
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
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

    # -------- Interactive manipulation event handlers --------

    def update_body_lists(self) -> None:
        """Update body selection combo boxes."""
        if self.sim_widget.model is None:
            return

        # Clear existing items
        self.constraint_body_combo.clear()
        self.ref_body_combo.clear()
        if hasattr(self, "visualization_tab"):
            self.visualization_tab.viz_body_combo.clear()
        if hasattr(self, "manip_body_combo"):
            self.manip_body_combo.clear()

        # Add all bodies
        for body_id in range(1, self.sim_widget.model.nbody):  # Skip world (0)
            body_name = mujoco.mj_id2name(
                self.sim_widget.model,
                mujoco.mjtObj.mjOBJ_BODY,
                body_id,
            )
            if body_name:
                item_text = f"{body_id}: {body_name}"
                self.constraint_body_combo.addItem(item_text)
                self.ref_body_combo.addItem(item_text)
                if hasattr(self, "visualization_tab"):
                    self.visualization_tab.viz_body_combo.addItem(item_text)
                if hasattr(self, "manip_body_combo"):
                    self.manip_body_combo.addItem(item_text)

    def on_manip_body_selected(self, index: int) -> None:
        """Handle body selection from combo box."""
        if index < 0:
            return

        text = self.manip_body_combo.itemText(index)
        try:
            body_id = int(text.split(":")[0])
            manipulator = self.sim_widget.get_manipulator()
            if manipulator:
                # Manually set selection
                manipulator.selected_body_id = body_id
                # Trigger update of manual transform values
                self.update_manual_transform_values()
                self.sim_widget._render_once()
        except ValueError:
            pass

    def on_manual_transform(self, type_: str, axis: int, value: float) -> None:
        """Handle manual transform changes."""
        manipulator = self.sim_widget.get_manipulator()
        if not manipulator or manipulator.selected_body_id is None:
            return

        # We need to know if it's a mocap body or free joint
        body_id = manipulator.selected_body_id
        model = self.sim_widget.model
        data = self.sim_widget.data

        if model is None or data is None:
            return

        mocap_id = model.body_mocapid[body_id]

        if mocap_id != -1:
            # It's a mocap body
            if type_ == "pos":
                data.mocap_pos[mocap_id][axis] = value
            elif type_ == "rot":
                # Warn user about unimplemented rotation
                QtWidgets.QMessageBox.warning(
                    self,
                    "Not Implemented",
                    "Rotation transformation for manual sliders is not implemented.",
                )
        else:
            # Not a mocap body
            status_bar = self.statusBar()
            if status_bar:
                status_bar.showMessage(
                    "Manual transform only available for mocap bodies", 3000
                )

        self.sim_widget._render_once()

    def update_manual_transform_values(self) -> None:
        """Update sliders from current selection."""
        manipulator = self.sim_widget.get_manipulator()
        if not manipulator or manipulator.selected_body_id is None:
            return

        body_id = manipulator.selected_body_id
        model = self.sim_widget.model
        data = self.sim_widget.data

        if model is None or data is None:
            return

        # Determine if the body is controlled by mocap
        mocap_id = model.body_mocapid[body_id]
        if mocap_id != -1:
            # Mocap body: get pose from mocap_pos and mocap_quat
            pos = data.mocap_pos[mocap_id]
            # quat = data.mocap_quat[mocap_id]
        else:
            # Free body: get pose from xpos and xquat
            pos = data.xpos[body_id]
            # quat = data.xquat[body_id]

        # Update the UI sliders/spinboxes for translation
        if (
            hasattr(self, "trans_x")
            and hasattr(self, "trans_y")
            and hasattr(self, "trans_z")
        ):
            pos_sliders = [self.trans_x, self.trans_y, self.trans_z]
            for i, slider in enumerate(pos_sliders):
                if slider is not None:
                    # Set value, block signals to avoid feedback loop
                    if hasattr(slider, "blockSignals"):
                        slider.blockSignals(True)
                    if hasattr(slider, "setValue"):
                        # Assume slider expects float in meters
                        slider.setValue(float(pos[i]))
                    elif hasattr(slider, "setText"):
                        slider.setText(str(float(pos[i])))
                    if hasattr(slider, "blockSignals"):
                        slider.blockSignals(False)

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

        logger.info("Pose '%s' saved successfully", pose_name)
        status_bar = self.statusBar()
        if status_bar:
            status_bar.showMessage(f"Pose '{pose_name}' saved successfully.", 3000)

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
            logger.info("Pose '%s' loaded successfully", pose_name)
            status_bar = self.statusBar()
            if status_bar:
                status_bar.showMessage(
                    f"Pose '{pose_name}' loaded successfully.",
                    3000,
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

        if reply == QtWidgets.QMessageBox.StandardButton.Yes:
            if manipulator.delete_pose(pose_name):
                self.update_pose_list()
                logger.info("Pose '%s' deleted successfully", pose_name)
                status_bar = self.statusBar()
                if status_bar:
                    status_bar.showMessage(
                        f"Pose '{pose_name}' deleted successfully.",
                        3000,
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
