"""Drake GUI UI setup mixin.

Extracts UI construction, kinematic controls, and mode handling
from DrakeSimApp (drake_gui_app.py).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.shared.python.engine_availability import (
    MATPLOTLIB_AVAILABLE,
    PYQT6_AVAILABLE,
)
from src.shared.python.logging_config import get_logger

HAS_QT = PYQT6_AVAILABLE
HAS_MATPLOTLIB = MATPLOTLIB_AVAILABLE

if HAS_QT:
    from PyQt6 import QtCore, QtGui, QtWidgets

# Drake imports
if TYPE_CHECKING or HAS_QT:
    try:
        from pydrake.all import (
            BodyIndex,
            JointIndex,
            PrismaticJoint,
            RevoluteJoint,
        )
    except ImportError:
        BodyIndex = None  # type: ignore[misc, assignment]
        JointIndex = None  # type: ignore[misc, assignment]
        PrismaticJoint = None  # type: ignore[misc, assignment]
        RevoluteJoint = None  # type: ignore[misc, assignment]

# Shared imports
try:
    from shared.python.dashboard.widgets import LivePlotWidget
except ImportError:
    LivePlotWidget = None  # type: ignore[misc, assignment]

# Constants
JOINT_ANGLE_MIN_RAD = -3.141592653589793
JOINT_ANGLE_MAX_RAD = 3.141592653589793
SPINBOX_STEP_RAD = 0.01
SLIDER_TO_RADIAN = 0.01
SLIDER_RANGE_MIN = -314
SLIDER_RANGE_MAX = 314
STYLE_BUTTON_RUN = "QPushButton { background-color: #4CAF50; color: white; }"
STYLE_BUTTON_STOP = "QPushButton { background-color: #f44336; color: white; }"
MS_PER_SECOND = 1000

LOGGER = get_logger(__name__)


class DrakeUIMixin:
    """Mixin for Drake GUI UI construction and control handling.

    Provides:
    - ``_setup_ui``: Full PyQt6 interface construction
    - ``_build_kinematic_controls``: Joint slider/spinbox creation
    - Mode change, slider/spin change, toggle, reset handlers
    - Overlay dialog for body frame/COM toggles
    """

    def _setup_ui(self: Any) -> None:  # noqa: PLR0915
        """Build the PyQt Interface."""
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        layout = QtWidgets.QVBoxLayout(central_widget)

        # 0. Model Selector
        model_group = QtWidgets.QGroupBox("Model Selection")
        model_layout = QtWidgets.QHBoxLayout()
        self.model_combo = QtWidgets.QComboBox()
        for model in self.available_models:
            self.model_combo.addItem(model["name"])
        self.model_combo.currentIndexChanged.connect(self._on_model_changed)
        model_layout.addWidget(QtWidgets.QLabel("Model:"))
        model_layout.addWidget(self.model_combo)
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)

        # 1. Mode Selector
        mode_group = QtWidgets.QGroupBox("Operating Mode")
        mode_layout = QtWidgets.QHBoxLayout()
        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItems(["Dynamic (Physics)", "Kinematic (Pose)"])
        self.mode_combo.setToolTip(
            "Select between physics simulation or manual pose control"
        )
        self.mode_combo.setStatusTip(
            "Select between physics simulation or manual pose control"
        )
        self.mode_combo.currentTextChanged.connect(self._on_mode_changed)
        mode_layout.addWidget(QtWidgets.QLabel("Mode:"))
        mode_layout.addWidget(self.mode_combo)
        mode_group.setLayout(mode_layout)
        layout.addWidget(mode_group)

        # 2. Controls Area (Stack)
        self.main_tab_widget = QtWidgets.QTabWidget()
        layout.addWidget(self.main_tab_widget)

        # Tab 1: Simulation Controls
        self.sim_tab = QtWidgets.QWidget()
        sim_tab_layout = QtWidgets.QVBoxLayout(self.sim_tab)

        self.controls_stack = QtWidgets.QStackedWidget()
        sim_tab_layout.addWidget(self.controls_stack)

        # -- Page 1: Dynamic Controls
        dynamic_page = QtWidgets.QWidget()
        dyn_layout = QtWidgets.QVBoxLayout(dynamic_page)

        self.btn_run = QtWidgets.QPushButton("▶ Run Simulation")
        self.btn_run.setCheckable(True)
        self.btn_run.setToolTip("Start or stop the physics simulation (Space)")
        self.btn_run.setStatusTip("Start or stop the physics simulation (Space)")
        self.btn_run.setShortcut(QtGui.QKeySequence(QtCore.Qt.Key.Key_Space))
        self.btn_run.setStyleSheet(STYLE_BUTTON_RUN)
        self.btn_run.clicked.connect(self._toggle_run)
        dyn_layout.addWidget(self.btn_run)

        self.btn_reset = QtWidgets.QPushButton("Reset")
        self.btn_reset.setToolTip("Reset the simulation to the initial state (Ctrl+R)")
        self.btn_reset.setStatusTip(
            "Reset the simulation to the initial state (Ctrl+R)"
        )
        self.btn_reset.setShortcut(QtGui.QKeySequence("Ctrl+R"))
        self.btn_reset.clicked.connect(self._reset_simulation)
        dyn_layout.addWidget(self.btn_reset)

        # Recording & Analysis
        analysis_group = QtWidgets.QGroupBox("Recording & Post-Hoc Analysis")
        analysis_layout = QtWidgets.QVBoxLayout()

        rec_row = QtWidgets.QHBoxLayout()
        self.btn_record = QtWidgets.QPushButton("Record")
        self.btn_record.setCheckable(True)
        self.btn_record.clicked.connect(self._toggle_recording)
        self.lbl_rec_status = QtWidgets.QLabel("Frames: 0")
        rec_row.addWidget(self.btn_record)
        rec_row.addWidget(self.lbl_rec_status)
        analysis_layout.addLayout(rec_row)

        # Induced Accel Plot
        ind_layout = QtWidgets.QHBoxLayout()
        self.btn_induced_acc = QtWidgets.QPushButton("Show Induced Acceleration")
        self.btn_induced_acc.setToolTip(
            "Analyze Gravity/Velocity/Control contributions to Acceleration"
        )
        self.btn_induced_acc.clicked.connect(self._show_induced_acceleration_plot)
        self.btn_induced_acc.setEnabled(HAS_MATPLOTLIB)
        ind_layout.addWidget(self.btn_induced_acc)

        analysis_layout.addLayout(ind_layout)

        self.btn_counterfactuals = QtWidgets.QPushButton(
            "Show Counterfactuals (ZTCF/ZVCF)"
        )
        self.btn_counterfactuals.setToolTip(
            "Show Zero Torque (ZTCF) and Zero Velocity (ZVCF) analysis"
        )
        self.btn_counterfactuals.clicked.connect(self._show_counterfactuals_plot)
        self.btn_counterfactuals.setEnabled(HAS_MATPLOTLIB)
        analysis_layout.addWidget(self.btn_counterfactuals)

        self.btn_swing_plane = QtWidgets.QPushButton("Show Swing Plane Analysis")
        self.btn_swing_plane.setToolTip("Analyze the swing plane and deviation")
        self.btn_swing_plane.clicked.connect(self._show_swing_plane_analysis)
        self.btn_swing_plane.setEnabled(HAS_MATPLOTLIB)
        analysis_layout.addWidget(self.btn_swing_plane)

        self.btn_advanced_plots = QtWidgets.QPushButton("Show Advanced Plots")
        self.btn_advanced_plots.setToolTip(
            "Show Radar Chart, CoP Field, and Power Flow"
        )
        self.btn_advanced_plots.clicked.connect(self._show_advanced_plots)
        self.btn_advanced_plots.setEnabled(HAS_MATPLOTLIB)
        analysis_layout.addWidget(self.btn_advanced_plots)

        self.btn_export = QtWidgets.QPushButton("Export Analysis Data (CSV)")
        self.btn_export.setToolTip("Export all recorded data and computed metrics")
        self.btn_export.clicked.connect(self._export_data)
        analysis_layout.addWidget(self.btn_export)

        analysis_group.setLayout(analysis_layout)
        dyn_layout.addWidget(analysis_group)

        dyn_layout.addStretch()
        self.controls_stack.addWidget(dynamic_page)

        # -- Page 2: Kinematic Controls
        kinematic_page = QtWidgets.QWidget()
        kin_layout = QtWidgets.QVBoxLayout(kinematic_page)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        self.kinematic_content = QtWidgets.QWidget()
        self.kinematic_layout = QtWidgets.QVBoxLayout(self.kinematic_content)
        scroll.setWidget(self.kinematic_content)

        kin_layout.addWidget(scroll)
        self.controls_stack.addWidget(kinematic_page)

        self.main_tab_widget.addTab(self.sim_tab, "Simulation Control")

        # Tab 2: Live Analysis (LivePlotWidget)
        if LivePlotWidget is not None:
            self.live_tab = QtWidgets.QWidget()
            live_layout = QtWidgets.QVBoxLayout(self.live_tab)
            self.live_plot = LivePlotWidget(self.recorder)
            # Pre-populate joint names
            self.live_plot.set_joint_names(self.get_joint_names())
            live_layout.addWidget(self.live_plot)
            self.main_tab_widget.addTab(self.live_tab, "Live Analysis")

        # 3. Visualization Toggles
        vis_group = QtWidgets.QGroupBox("Visualization")
        vis_layout = QtWidgets.QVBoxLayout()

        self.btn_overlays = QtWidgets.QPushButton("Manage Body Overlays")
        self.btn_overlays.setToolTip(
            "Toggle visibility of reference frames and centers of mass"
        )
        self.btn_overlays.setStatusTip(
            "Toggle visibility of reference frames and centers of mass"
        )
        self.btn_overlays.clicked.connect(self._show_overlay_dialog)
        vis_layout.addWidget(self.btn_overlays)

        # Force/Torque Toggles
        ft_grid = QtWidgets.QGridLayout()
        self.chk_show_forces = QtWidgets.QCheckBox("Show Forces")
        self.chk_show_forces.toggled.connect(self._on_visualization_changed)
        self.chk_show_torques = QtWidgets.QCheckBox("Show Torques")
        self.chk_show_torques.toggled.connect(self._on_visualization_changed)
        ft_grid.addWidget(self.chk_show_forces, 0, 0)
        ft_grid.addWidget(self.chk_show_torques, 0, 1)
        vis_layout.addLayout(ft_grid)

        # Ellipsoid Toggles
        self.chk_mobility = QtWidgets.QCheckBox("Show Mobility Ellipsoid (Green)")
        self.chk_mobility.toggled.connect(self._on_visualization_changed)
        vis_layout.addWidget(self.chk_mobility)

        self.chk_force_ellip = QtWidgets.QCheckBox("Show Force Ellipsoid (Red)")
        self.chk_force_ellip.toggled.connect(self._on_visualization_changed)
        vis_layout.addWidget(self.chk_force_ellip)

        self.chk_live_analysis = QtWidgets.QCheckBox("Live Analysis (Induced/CF)")
        self.chk_live_analysis.setToolTip(
            "Compute Induced Accelerations and Counterfactuals in real-time "
            "(Can slow down sim)"
        )
        vis_layout.addWidget(self.chk_live_analysis)

        # Manipulability Body Grid
        manip_group = QtWidgets.QGroupBox("Manipulability Targets")
        self.manip_body_layout = QtWidgets.QGridLayout()
        manip_group.setLayout(self.manip_body_layout)
        vis_layout.addWidget(manip_group)

        # Advanced Vectors
        vec_grid = QtWidgets.QGridLayout()

        self.chk_induced_vec = QtWidgets.QCheckBox("Induced Vectors")
        self.chk_induced_vec.toggled.connect(self._on_visualization_changed)

        self.combo_induced_source = QtWidgets.QComboBox()
        self.combo_induced_source.setEditable(True)
        self.combo_induced_source.addItems(["gravity", "velocity", "total"])
        self.combo_induced_source.setToolTip(
            "Select source (e.g. gravity) or type specific actuator index"
        )
        # Use lineEdit().editingFinished to avoid performance issues
        if line_edit := self.combo_induced_source.lineEdit():
            line_edit.editingFinished.connect(self._on_visualization_changed)
        # Also connect index changed for dropdown selection
        self.combo_induced_source.currentIndexChanged.connect(
            self._on_visualization_changed
        )

        self.chk_cf_vec = QtWidgets.QCheckBox("CF Vectors")
        self.chk_cf_vec.toggled.connect(self._on_visualization_changed)

        self.combo_cf_type = QtWidgets.QComboBox()
        self.combo_cf_type.addItems(["ztcf_accel", "zvcf_torque"])
        self.combo_cf_type.currentIndexChanged.connect(self._on_visualization_changed)

        vec_grid.addWidget(self.chk_induced_vec, 0, 0)
        vec_grid.addWidget(self.combo_induced_source, 0, 1)
        vec_grid.addWidget(self.chk_cf_vec, 1, 0)
        vec_grid.addWidget(self.combo_cf_type, 1, 1)

        vis_layout.addLayout(vec_grid)

        vis_group.setLayout(vis_layout)
        layout.addWidget(vis_group)

        # Matrix Analysis
        matrix_group = QtWidgets.QGroupBox("Matrix Analysis")
        matrix_layout = QtWidgets.QFormLayout(matrix_group)
        self.lbl_cond = QtWidgets.QLabel("--")
        self.lbl_rank = QtWidgets.QLabel("--")
        matrix_layout.addRow("Jacobian Cond:", self.lbl_cond)
        matrix_layout.addRow("Constraint Rank:", self.lbl_rank)
        layout.addWidget(matrix_group)

        # Status Bar
        self._update_status("Ready")

        # Populate Kinematic Sliders
        self._build_kinematic_controls()

    def _build_kinematic_controls(self: Any) -> None:  # noqa: PLR0915
        """Create sliders for all joints."""
        plant = self.plant
        if not plant:
            return

        self.sliders.clear()
        self.spinboxes.clear()

        # Update Live Plot joint names if initialized
        if hasattr(self, "live_plot"):
            self.live_plot.set_joint_names(self.get_joint_names())

        # Populate induced source combo with joint names
        current_text = self.combo_induced_source.currentText()
        self.combo_induced_source.clear()
        self.combo_induced_source.addItems(["gravity", "velocity", "total"])
        for i in range(plant.num_joints()):
            joint = plant.get_joint(JointIndex(i))
            if joint.num_velocities() == 1:
                self.combo_induced_source.addItem(joint.name())
        if current_text:
            self.combo_induced_source.setCurrentText(current_text)

        # Iterate over joints
        for i in range(plant.num_joints()):
            joint = plant.get_joint(JointIndex(i))

            # Skip welds (0 DOF) and multi-DOF joints (not yet supported)
            if joint.num_positions() != 1:
                continue

            # Create control row
            row = QtWidgets.QWidget()
            row_layout = QtWidgets.QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)

            label = QtWidgets.QLabel(f"{joint.name()}:")
            label.setMinimumWidth(120)
            row_layout.addWidget(label)

            # Slider
            slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
            label.setBuddy(slider)
            slider.setRange(SLIDER_RANGE_MIN, SLIDER_RANGE_MAX)
            slider.setValue(0)

            # Determine joint limits for tooltip: prefer physical, else UI
            try:
                # For single-DOF joints, these are length-1 arrays
                joint_min = float(joint.position_lower_limits()[0])
                joint_max = float(joint.position_upper_limits()[0])
            except ImportError:
                # Fallback to UI limits if joint does not provide limits
                joint_min = JOINT_ANGLE_MIN_RAD
                joint_max = JOINT_ANGLE_MAX_RAD

            slider.setToolTip(
                f"Adjust angle for {joint.name()} (radians, "
                f"{joint_min:.2f} to {joint_max:.2f})"
            )

            # Spinbox
            spin = QtWidgets.QDoubleSpinBox()
            spin.setRange(JOINT_ANGLE_MIN_RAD, JOINT_ANGLE_MAX_RAD)
            spin.setSingleStep(SPINBOX_STEP_RAD)
            spin.setDecimals(3)

            # Connect
            j_idx = int(joint.index())
            if not (0 <= j_idx < plant.num_joints()):
                msg = (
                    f"Joint index {j_idx} out of bounds for plant with "
                    f"{plant.num_joints()} joints."
                )
                raise ValueError(msg)

            slider.valueChanged.connect(
                lambda val, s=spin, idx=j_idx: self._on_slider_change(val, s, idx)
            )
            spin.valueChanged.connect(
                lambda val, s=slider, idx=j_idx: self._on_spin_change(val, s, idx)
            )

            row_layout.addWidget(slider)
            row_layout.addWidget(spin)

            self.kinematic_layout.addWidget(row)

            self.sliders[j_idx] = slider
            self.spinboxes[j_idx] = spin

    def _on_slider_change(  # type: ignore[no-any-unimported]
        self: Any, val: int, spin: QtWidgets.QDoubleSpinBox, joint_idx: int
    ) -> None:
        radian = val * SLIDER_TO_RADIAN
        with QtCore.QSignalBlocker(spin):
            spin.setValue(radian)
        self._update_joint_pos(joint_idx, radian)

    def _on_spin_change(  # type: ignore[no-any-unimported]
        self: Any, val: float, slider: QtWidgets.QSlider, joint_idx: int
    ) -> None:
        with QtCore.QSignalBlocker(slider):
            slider.setValue(int(val / SLIDER_TO_RADIAN))
        self._update_joint_pos(joint_idx, val)

    def _update_joint_pos(self: Any, joint_idx: int, angle: float) -> None:
        """Update joint position in plant context."""
        if self.operating_mode != "kinematic":
            return

        plant = self.plant
        context = self.context
        diagram = self.diagram

        if not plant or not context or not diagram:
            return

        plant_context = plant.GetMyContextFromRoot(context)

        joint = plant.get_joint(JointIndex(joint_idx))

        # Assuming single DOF revolute/prismatic for now
        if joint.num_positions() == 1:
            if isinstance(joint, RevoluteJoint):
                joint.set_angle(plant_context, angle)
            elif isinstance(joint, PrismaticJoint):
                joint.set_translation(plant_context, angle)

        diagram.ForcedPublish(context)

        # Update overlays
        if self.visualizer:
            self.visualizer.update_frame_transforms(context)
            self.visualizer.update_com_transforms(context)

        self._update_visualization()

    def _sync_kinematic_sliders(self: Any) -> None:
        """Read current plant state and update sliders."""
        plant = self.plant
        context = self.context
        if not plant or not context:
            return

        plant_context = plant.GetMyContextFromRoot(context)

        for j_idx, spin in self.spinboxes.items():
            joint = plant.get_joint(JointIndex(j_idx))
            if joint.num_positions() == 1:
                val = joint.GetOnePosition(plant_context)
                spin.setValue(val)

    def _update_status(self: Any, message: str) -> None:
        """Update status bar message safely."""
        status_bar = self.statusBar()
        if status_bar:
            status_bar.showMessage(message)

    def _on_mode_changed(self: Any, text: str) -> None:
        if "Kinematic" in text:
            self.operating_mode = "kinematic"
            self.controls_stack.setCurrentIndex(1)
            self.is_running = False
            self.btn_run.setChecked(False)
            self.btn_run.setStyleSheet(STYLE_BUTTON_RUN)
            self.btn_run.setText("▶ Run Simulation")
            self._update_status("Mode: Kinematic Control")
            self._sync_kinematic_sliders()
            # Stop physics, allow manual
        else:
            self.operating_mode = "dynamic"
            self.controls_stack.setCurrentIndex(0)
            self._update_status("Mode: Dynamic Simulation")
            # Ensure simulation resumes or is stopped
            if self.is_running:
                self.btn_run.setText("■ Stop Simulation")
                self.btn_run.setChecked(True)
                self.btn_run.setStyleSheet(STYLE_BUTTON_STOP)
            else:
                self.btn_run.setText("▶ Run Simulation")
                self.btn_run.setChecked(False)
                self.btn_run.setStyleSheet(STYLE_BUTTON_RUN)

    def _toggle_run(self: Any, checked: bool) -> None:  # noqa: FBT001
        self.is_running = checked
        if checked:
            self.btn_run.setText("■ Stop Simulation")
            self.btn_run.setStyleSheet(STYLE_BUTTON_STOP)
            self._update_status("Simulation Running...")
        else:
            self.btn_run.setText("▶ Run Simulation")
            self.btn_run.setStyleSheet(STYLE_BUTTON_RUN)
            self._update_status("Simulation Stopped.")

    def _reset_simulation(self: Any) -> None:
        self.is_running = False
        self.btn_run.setChecked(False)
        self.btn_run.setText("▶ Run Simulation")
        self.btn_run.setStyleSheet(STYLE_BUTTON_RUN)
        self._update_status("Simulation Reset.")
        self._reset_state()

    def _toggle_recording(self: Any, checked: bool) -> None:  # type: ignore[override]  # noqa: FBT001
        if checked:
            self.recorder.start()
            self.btn_record.setText("Stop Recording")
            self._update_status("Recording started...")
        else:
            self.recorder.stop()
            self.btn_record.setText("Record")
            self._update_status(
                f"Recording stopped. Total Frames: {len(self.recorder.times)}"
            )

    def _show_overlay_dialog(self: Any) -> None:  # noqa: PLR0915
        """Show dialog to toggle overlays for specific bodies."""
        plant = self.plant
        diagram = self.diagram
        context = self.context
        visualizer = self.visualizer

        if not plant or not diagram or not context or not visualizer:
            return

        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Manage Overlays")
        layout = QtWidgets.QVBoxLayout(dialog)

        scroll = QtWidgets.QScrollArea()
        content = QtWidgets.QWidget()
        c_layout = QtWidgets.QVBoxLayout(content)

        # List all bodies
        for i in range(plant.num_bodies()):
            body = plant.get_body(BodyIndex(i))
            name = body.name()
            if name == "world":
                continue

            b_row = QtWidgets.QHBoxLayout()
            lbl = QtWidgets.QLabel(name)

            chk_frame = QtWidgets.QCheckBox("Frame")
            is_vis_f = name in visualizer.visible_frames
            chk_frame.setChecked(is_vis_f)
            chk_frame.toggled.connect(lambda c, n=name: visualizer.toggle_frame(n, c))

            chk_com = QtWidgets.QCheckBox("COM")
            is_vis_c = name in visualizer.visible_coms
            chk_com.setChecked(is_vis_c)
            chk_com.toggled.connect(lambda c, n=name: visualizer.toggle_com(n, c))

            b_row.addWidget(lbl)
            b_row.addWidget(chk_frame)
            b_row.addWidget(chk_com)
            c_layout.addLayout(b_row)

        scroll.setWidget(content)
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll)

        close = QtWidgets.QPushButton("Close")
        close.clicked.connect(dialog.accept)
        layout.addWidget(close)

        dialog.exec()

        if self.operating_mode == "kinematic":
            diagram.ForcedPublish(context)
            if self.visualizer:
                self.visualizer.update_frame_transforms(context)
                self.visualizer.update_com_transforms(context)
