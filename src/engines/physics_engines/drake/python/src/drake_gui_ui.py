"""Drake GUI UI Setup Mixin.

Contains all widget/layout construction methods for DrakeSimApp.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from src.shared.python.engine_core.engine_availability import (
    MATPLOTLIB_AVAILABLE,
    PYQT6_AVAILABLE,
)
from src.shared.python.theme.style_constants import Styles

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
        )
    except ImportError:
        BodyIndex = None  # type: ignore[misc, assignment]
        JointIndex = None  # type: ignore[misc, assignment]

# Shared imports
try:
    from shared.python.dashboard.widgets import LivePlotWidget
except ImportError:
    LivePlotWidget = None  # type: ignore[misc, assignment]

# Constants (duplicated from main module to avoid circular import)
JOINT_ANGLE_MIN_RAD = -np.pi
JOINT_ANGLE_MAX_RAD = np.pi
SPINBOX_STEP_RAD = 0.01
SLIDER_RANGE_MIN = -314
SLIDER_RANGE_MAX = 314
STYLE_BUTTON_RUN = Styles.BTN_RUN
STYLE_BUTTON_STOP = Styles.BTN_STOP


class UISetupMixin:
    """Mixin providing all UI construction methods for DrakeSimApp."""

    def _setup_ui(self) -> None:
        """Build the PyQt Interface."""
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)  # type: ignore[attr-defined]
        layout = QtWidgets.QVBoxLayout(central_widget)

        # 0. Model Selector
        self._setup_model_selector(layout)

        # 1. Mode Selector
        self._setup_mode_selector(layout)

        # 2. Controls Area (Tabs + Stack)
        self._setup_controls_tabs(layout)

        # 3. Visualization Toggles
        self._setup_visualization_panel(layout)

        # 4. Matrix Analysis
        self._setup_matrix_analysis_panel(layout)

        # Status Bar
        self._update_status("Ready")  # type: ignore[attr-defined]

        # Populate Kinematic Sliders
        self._build_kinematic_controls()

    def _setup_model_selector(self, layout: QtWidgets.QVBoxLayout) -> None:
        """Build the model selection group box."""
        model_group = QtWidgets.QGroupBox("Model Selection")
        model_layout = QtWidgets.QHBoxLayout()
        self.model_combo = QtWidgets.QComboBox()
        for model in self.available_models:  # type: ignore[attr-defined]
            self.model_combo.addItem(model["name"])
        self.model_combo.currentIndexChanged.connect(self._on_model_changed)  # type: ignore[attr-defined]
        model_layout.addWidget(QtWidgets.QLabel("Model:"))
        model_layout.addWidget(self.model_combo)
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)

    def _setup_mode_selector(self, layout: QtWidgets.QVBoxLayout) -> None:
        """Build the operating mode group box."""
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
        self.mode_combo.currentTextChanged.connect(self._on_mode_changed)  # type: ignore[attr-defined]
        mode_layout.addWidget(QtWidgets.QLabel("Mode:"))
        mode_layout.addWidget(self.mode_combo)
        mode_group.setLayout(mode_layout)
        layout.addWidget(mode_group)

    def _setup_controls_tabs(self, layout: QtWidgets.QVBoxLayout) -> None:
        """Build the main tab widget with simulation controls and live analysis."""
        self.main_tab_widget = QtWidgets.QTabWidget()
        layout.addWidget(self.main_tab_widget)

        # Tab 1: Simulation Controls
        self.sim_tab = QtWidgets.QWidget()
        sim_tab_layout = QtWidgets.QVBoxLayout(self.sim_tab)

        self.controls_stack = QtWidgets.QStackedWidget()
        sim_tab_layout.addWidget(self.controls_stack)

        self._setup_dynamic_controls_page()
        self._setup_kinematic_controls_page()

        self.main_tab_widget.addTab(self.sim_tab, "Simulation Control")

        # Tab 2: Live Analysis (LivePlotWidget)
        if LivePlotWidget is not None:
            self.live_tab = QtWidgets.QWidget()
            live_layout = QtWidgets.QVBoxLayout(self.live_tab)
            self.live_plot = LivePlotWidget(self.recorder)  # type: ignore[attr-defined]
            # Pre-populate joint names
            self.live_plot.set_joint_names(self.get_joint_names())  # type: ignore[attr-defined]
            live_layout.addWidget(self.live_plot)
            self.main_tab_widget.addTab(self.live_tab, "Live Analysis")

    def _setup_dynamic_controls_page(self) -> None:
        """Build the dynamic simulation controls page."""
        dynamic_page = QtWidgets.QWidget()
        dyn_layout = QtWidgets.QVBoxLayout(dynamic_page)

        self.btn_run = QtWidgets.QPushButton("▶ Run Simulation")
        self.btn_run.setCheckable(True)
        self.btn_run.setToolTip("Start or stop the physics simulation (Space)")
        self.btn_run.setStatusTip("Start or stop the physics simulation (Space)")
        self.btn_run.setShortcut(QtGui.QKeySequence(QtCore.Qt.Key.Key_Space))
        self.btn_run.setStyleSheet(STYLE_BUTTON_RUN)
        self.btn_run.clicked.connect(self._toggle_run)  # type: ignore[attr-defined]
        dyn_layout.addWidget(self.btn_run)

        self.btn_reset = QtWidgets.QPushButton("Reset")
        self.btn_reset.setToolTip("Reset the simulation to the initial state (Ctrl+R)")
        self.btn_reset.setStatusTip(
            "Reset the simulation to the initial state (Ctrl+R)"
        )
        self.btn_reset.setShortcut(QtGui.QKeySequence("Ctrl+R"))
        self.btn_reset.clicked.connect(self._reset_simulation)  # type: ignore[attr-defined]
        dyn_layout.addWidget(self.btn_reset)

        # Recording & Analysis
        self._setup_analysis_group(dyn_layout)

        dyn_layout.addStretch()
        self.controls_stack.addWidget(dynamic_page)

    def _setup_analysis_group(self, parent_layout: QtWidgets.QVBoxLayout) -> None:
        """Build the recording and post-hoc analysis group box."""
        analysis_group = QtWidgets.QGroupBox("Recording & Post-Hoc Analysis")
        analysis_layout = QtWidgets.QVBoxLayout()

        rec_row = QtWidgets.QHBoxLayout()
        self.btn_record = QtWidgets.QPushButton("Record")
        self.btn_record.setCheckable(True)
        self.btn_record.clicked.connect(self._toggle_recording)  # type: ignore[attr-defined]
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
        self.btn_induced_acc.clicked.connect(self._show_induced_acceleration_plot)  # type: ignore[attr-defined]
        self.btn_induced_acc.setEnabled(HAS_MATPLOTLIB)
        ind_layout.addWidget(self.btn_induced_acc)

        analysis_layout.addLayout(ind_layout)

        self.btn_counterfactuals = QtWidgets.QPushButton(
            "Show Counterfactuals (ZTCF/ZVCF)"
        )
        self.btn_counterfactuals.setToolTip(
            "Show Zero Torque (ZTCF) and Zero Velocity (ZVCF) analysis"
        )
        self.btn_counterfactuals.clicked.connect(self._show_counterfactuals_plot)  # type: ignore[attr-defined]
        self.btn_counterfactuals.setEnabled(HAS_MATPLOTLIB)
        analysis_layout.addWidget(self.btn_counterfactuals)

        self.btn_swing_plane = QtWidgets.QPushButton("Show Swing Plane Analysis")
        self.btn_swing_plane.setToolTip("Analyze the swing plane and deviation")
        self.btn_swing_plane.clicked.connect(self._show_swing_plane_analysis)  # type: ignore[attr-defined]
        self.btn_swing_plane.setEnabled(HAS_MATPLOTLIB)
        analysis_layout.addWidget(self.btn_swing_plane)

        self.btn_advanced_plots = QtWidgets.QPushButton("Show Advanced Plots")
        self.btn_advanced_plots.setToolTip(
            "Show Radar Chart, CoP Field, and Power Flow"
        )
        self.btn_advanced_plots.clicked.connect(self._show_advanced_plots)  # type: ignore[attr-defined]
        self.btn_advanced_plots.setEnabled(HAS_MATPLOTLIB)
        analysis_layout.addWidget(self.btn_advanced_plots)

        self.btn_export = QtWidgets.QPushButton("Export Analysis Data (CSV)")
        self.btn_export.setToolTip("Export all recorded data and computed metrics")
        self.btn_export.clicked.connect(self._export_data)  # type: ignore[attr-defined]
        analysis_layout.addWidget(self.btn_export)

        analysis_group.setLayout(analysis_layout)
        parent_layout.addWidget(analysis_group)

    def _setup_kinematic_controls_page(self) -> None:
        """Build the kinematic controls page with joint sliders scroll area."""
        kinematic_page = QtWidgets.QWidget()
        kin_layout = QtWidgets.QVBoxLayout(kinematic_page)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        self.kinematic_content = QtWidgets.QWidget()
        self.kinematic_layout = QtWidgets.QVBoxLayout(self.kinematic_content)
        scroll.setWidget(self.kinematic_content)

        kin_layout.addWidget(scroll)
        self.controls_stack.addWidget(kinematic_page)

    def _setup_visualization_panel(self, layout: QtWidgets.QVBoxLayout) -> None:
        """Build the visualization toggles group box."""
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
        self.chk_show_forces.toggled.connect(self._on_visualization_changed)  # type: ignore[attr-defined]
        self.chk_show_torques = QtWidgets.QCheckBox("Show Torques")
        self.chk_show_torques.toggled.connect(self._on_visualization_changed)  # type: ignore[attr-defined]
        ft_grid.addWidget(self.chk_show_forces, 0, 0)
        ft_grid.addWidget(self.chk_show_torques, 0, 1)
        vis_layout.addLayout(ft_grid)

        # Ellipsoid Toggles
        self.chk_mobility = QtWidgets.QCheckBox("Show Mobility Ellipsoid (Green)")
        self.chk_mobility.toggled.connect(self._on_visualization_changed)  # type: ignore[attr-defined]
        vis_layout.addWidget(self.chk_mobility)

        self.chk_force_ellip = QtWidgets.QCheckBox("Show Force Ellipsoid (Red)")
        self.chk_force_ellip.toggled.connect(self._on_visualization_changed)  # type: ignore[attr-defined]
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
        self._setup_advanced_vectors(vis_layout)

        vis_group.setLayout(vis_layout)
        layout.addWidget(vis_group)

    def _setup_advanced_vectors(self, vis_layout: QtWidgets.QVBoxLayout) -> None:
        """Build the advanced vector controls (induced accel, counterfactual combos)."""
        vec_grid = QtWidgets.QGridLayout()

        self.chk_induced_vec = QtWidgets.QCheckBox("Induced Vectors")
        self.chk_induced_vec.toggled.connect(self._on_visualization_changed)  # type: ignore[attr-defined]

        self.combo_induced_source = QtWidgets.QComboBox()
        self.combo_induced_source.setEditable(True)
        self.combo_induced_source.addItems(["gravity", "velocity", "total"])
        self.combo_induced_source.setToolTip(
            "Select source (e.g. gravity) or type specific actuator index"
        )
        # Use lineEdit().editingFinished to avoid performance issues
        if line_edit := self.combo_induced_source.lineEdit():
            line_edit.editingFinished.connect(self._on_visualization_changed)  # type: ignore[attr-defined]
        # Also connect index changed for dropdown selection
        self.combo_induced_source.currentIndexChanged.connect(
            self._on_visualization_changed  # type: ignore[attr-defined]
        )

        self.chk_cf_vec = QtWidgets.QCheckBox("CF Vectors")
        self.chk_cf_vec.toggled.connect(self._on_visualization_changed)  # type: ignore[attr-defined]

        self.combo_cf_type = QtWidgets.QComboBox()
        self.combo_cf_type.addItems(["ztcf_accel", "zvcf_torque"])
        self.combo_cf_type.currentIndexChanged.connect(self._on_visualization_changed)  # type: ignore[attr-defined]

        vec_grid.addWidget(self.chk_induced_vec, 0, 0)
        vec_grid.addWidget(self.combo_induced_source, 0, 1)
        vec_grid.addWidget(self.chk_cf_vec, 1, 0)
        vec_grid.addWidget(self.combo_cf_type, 1, 1)

        vis_layout.addLayout(vec_grid)

    def _setup_matrix_analysis_panel(self, layout: QtWidgets.QVBoxLayout) -> None:
        """Build the matrix analysis group box (Jacobian condition, constraint rank)."""
        matrix_group = QtWidgets.QGroupBox("Matrix Analysis")
        matrix_layout = QtWidgets.QFormLayout(matrix_group)
        self.lbl_cond = QtWidgets.QLabel("--")
        self.lbl_rank = QtWidgets.QLabel("--")
        matrix_layout.addRow("Jacobian Cond:", self.lbl_cond)
        matrix_layout.addRow("Constraint Rank:", self.lbl_rank)
        layout.addWidget(matrix_group)

    def _build_kinematic_controls(self) -> None:  # noqa: PLR0915
        """Create sliders for all joints."""
        plant = self.plant  # type: ignore[attr-defined]
        if not plant:
            return

        self.sliders.clear()  # type: ignore[attr-defined]
        self.spinboxes.clear()  # type: ignore[attr-defined]

        # Update Live Plot joint names if initialized
        if hasattr(self, "live_plot"):
            self.live_plot.set_joint_names(self.get_joint_names())  # type: ignore[attr-defined]

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
                joint_min = float(joint.position_lower_limits()[0])
                joint_max = float(joint.position_upper_limits()[0])
            except ImportError:
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
                lambda val, s=spin, idx=j_idx: self._on_slider_change(val, s, idx)  # type: ignore[attr-defined]
            )
            spin.valueChanged.connect(
                lambda val, s=slider, idx=j_idx: self._on_spin_change(val, s, idx)  # type: ignore[attr-defined]
            )

            row_layout.addWidget(slider)
            row_layout.addWidget(spin)

            self.kinematic_layout.addWidget(row)

            self.sliders[j_idx] = slider  # type: ignore[attr-defined]
            self.spinboxes[j_idx] = spin  # type: ignore[attr-defined]

    def _show_overlay_dialog(self) -> None:  # noqa: PLR0915
        """Show dialog to toggle overlays for specific bodies."""
        plant = self.plant  # type: ignore[attr-defined]
        diagram = self.diagram  # type: ignore[attr-defined]
        context = self.context  # type: ignore[attr-defined]
        visualizer = self.visualizer  # type: ignore[attr-defined]

        if not plant or not diagram or not context or not visualizer:
            return

        dialog = QtWidgets.QDialog(self)  # type: ignore[arg-type]
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
            is_vis_f = name in visualizer.visible_frames  # type: ignore[attr-defined]
            chk_frame.setChecked(is_vis_f)
            chk_frame.toggled.connect(lambda c, n=name: visualizer.toggle_frame(n, c))  # type: ignore[attr-defined]

            chk_com = QtWidgets.QCheckBox("COM")
            is_vis_c = name in visualizer.visible_coms  # type: ignore[attr-defined]
            chk_com.setChecked(is_vis_c)
            chk_com.toggled.connect(lambda c, n=name: visualizer.toggle_com(n, c))  # type: ignore[attr-defined]

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

        if self.operating_mode == "kinematic":  # type: ignore[attr-defined]
            diagram.ForcedPublish(context)
            if self.visualizer:  # type: ignore[attr-defined]
                self.visualizer.update_frame_transforms(context)  # type: ignore[attr-defined]
                self.visualizer.update_com_transforms(context)  # type: ignore[attr-defined]

    def _populate_manip_checkboxes(self) -> None:
        """Populate checkboxes for manipulability analysis."""
        if not self.manip_analyzer or not self.manip_body_layout:  # type: ignore[attr-defined]
            return

        # Clear existing
        while self.manip_body_layout.count():
            item = self.manip_body_layout.takeAt(0)
            if item is None:
                continue
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self.manip_checkboxes.clear()  # type: ignore[attr-defined]

        bodies = self.manip_analyzer.find_potential_bodies()  # type: ignore[attr-defined]

        cols = 3
        for i, name in enumerate(bodies):
            chk = QtWidgets.QCheckBox(name)
            chk.toggled.connect(self._on_visualization_changed)  # type: ignore[attr-defined]
            self.manip_checkboxes[name] = chk  # type: ignore[attr-defined]
            self.manip_body_layout.addWidget(chk, i // cols, i % cols)

            # Default check relevant parts
            if any(x in name.lower() for x in ["club", "hand", "wrist"]):
                chk.setChecked(True)
